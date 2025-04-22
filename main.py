import asyncio
import base64
import csv
import json
import logging
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from uuid import uuid4

from cachetools import TTLCache
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Body, Query, UploadFile, File, HTTPException, status, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fuzzywuzzy import fuzz
from google.cloud import texttospeech
from openai import OpenAI
from pydantic import BaseModel
import sqlite3

# 🌍 Logging yapılandırması
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 🌍 Ortam değişkenleri
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CREDS_BASE64 = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_BASE64")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

if not all([OPENAI_API_KEY, ADMIN_USERNAME, ADMIN_PASSWORD]):
    raise ValueError("OPENAI_API_KEY, ADMIN_USERNAME ve ADMIN_PASSWORD ortam değişkenleri zorunludur.")

# Google Cloud kimlik bilgileri
if GOOGLE_CREDS_BASE64:
    creds_path = os.path.join(os.getcwd(), "google_creds.json")
    if not os.path.exists(creds_path):
        decoded = base64.b64decode(GOOGLE_CREDS_BASE64)
        with open(creds_path, "wb") as f:
            f.write(decoded)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

# FastAPI ve istemci başlatma
app = FastAPI()
client = OpenAI(api_key=OPENAI_API_KEY)
security = HTTPBasic()

# Menü önbelleği (5 dakika TTL)
menu_cache = TTLCache(maxsize=1, ttl=300)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Aktif WebSocket'ler ve kullanıcılar
aktif_mutfak_websocketleri = []
aktif_kullanicilar = {}

# Pydantic modelleri
class Siparis(BaseModel):
    masa: str
    yanit: Optional[str] = None
    sepet: List[Dict]

class MenuItem(BaseModel):
    ad: str
    fiyat: float
    kategori: str

# Veritabanı bağlam yöneticisi
@contextmanager
def get_db(db_name: str):
    """Veritabanı bağlantısını yönetir."""
    conn = sqlite3.connect(db_name)
    try:
        yield conn
    finally:
        conn.close()

@app.middleware("http")
async def aktif_kullanici_takibi(request: Request, call_next):
    """Kullanıcı oturumlarını takip eder."""
    ip = request.client.host
    agent = request.headers.get("user-agent", "")
    kimlik = f"{ip}_{agent}"
    aktif_kullanicilar[kimlik] = datetime.now()
    response = await call_next(request)
    return response

@app.get("/istatistik/online")
def online_kullanici_sayisi():
    """Çevrimiçi kullanıcı sayısını döndürür."""
    su_an = datetime.now()
    aktifler = [kimlik for kimlik, zaman in aktif_kullanicilar.items() if (su_an - zaman).seconds < 300]
    return {"count": len(aktifler)}

@app.websocket("/ws/mutfak")
async def websocket_mutfak(websocket: WebSocket):
    """Mutfak WebSocket bağlantısını yönetir."""
    await websocket.accept()
    aktif_mutfak_websocketleri.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                json.loads(data)  # Mesaj formatını kontrol et
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Geçersiz JSON formatı"}))
    except WebSocketDisconnect:
        aktif_mutfak_websocketleri.remove(websocket)

async def mutfaga_gonder(siparis: dict):
    """Siparişi mutfak WebSocket'lerine gönderir."""
    for ws in aktif_mutfak_websocketleri[:]:  # Kopya liste ile iterasyon
        try:
            await ws.send_text(json.dumps(siparis))
        except Exception as e:
            logger.error(f"WebSocket gönderim hatası: {e}")
            aktif_mutfak_websocketleri.remove(ws)

@app.post("/siparis-ekle")
async def siparis_ekle(data: Siparis):
    """Yeni sipariş kaydeder ve mutfağa iletir."""
    logger.info(f"Yeni sipariş geldi: {data.dict()}")
    zaman = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        istek = ", ".join([f"{item.get('urun', '').strip()} ({item.get('adet', 1)} adet)" for item in data.sepet])
    except Exception as e:
        logger.warning(f"İstek oluşturma hatası: {e}")
        istek = "Tanımsız"

    try:
        with get_db("neso.db") as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO siparisler (masa, istek, yanit, sepet, zaman)
                VALUES (?, ?, ?, ?, ?)
                """,
                (data.masa, istek, data.yanit, json.dumps(data.sepet), zaman),
            )
            conn.commit()

        await mutfaga_gonder({
            "masa": data.masa,
            "istek": istek,
            "yanit": data.yanit,
            "sepet": data.sepet,
            "zaman": zaman
        })

        return {"mesaj": "Sipariş başarıyla kaydedildi ve mutfağa iletildi."}
    except Exception as e:
        logger.error(f"Sipariş ekleme hatası: {e}")
        raise HTTPException(status_code=500, detail="Sipariş eklenemedi.")

def init_db():
    """Sipariş veritabanını başlatır."""
    with get_db("neso.db") as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS siparisler (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                masa TEXT,
                istek TEXT,
                yanit TEXT,
                zaman TEXT,
                sepet TEXT
            )
        """)
        conn.commit()

def init_menu_db():
    """Menü veritabanını başlatır ve CSV'den veri yükler."""
    yeni_olustu = not os.path.exists("neso_menu.db")
    with get_db("neso_menu.db") as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kategoriler (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                isim TEXT UNIQUE NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS menu (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ad TEXT NOT NULL,
                fiyat REAL NOT NULL,
                kategori_id INTEGER NOT NULL,
                FOREIGN KEY (kategori_id) REFERENCES kategoriler(id)
            )
        """)
        conn.commit()

    if yeni_olustu and os.path.exists("menu.csv"):
        try:
            with open("menu.csv", "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                with get_db("neso_menu.db") as conn:
                    cursor = conn.cursor()
                    for row in reader:
                        urun = row["urun"]
                        fiyat = float(row["fiyat"])
                        kategori = row["kategori"]
                        cursor.execute("INSERT OR IGNORE INTO kategoriler (isim) VALUES (?)", (kategori,))
                        cursor.execute("SELECT id FROM kategoriler WHERE isim = ?", (kategori,))
                        kategori_id = cursor.fetchone()[0]
                        cursor.execute("INSERT INTO menu (ad, fiyat, kategori_id) VALUES (?, ?, ?)", (urun, fiyat, kategori_id))
                    conn.commit()
        except Exception as e:
            logger.error(f"CSV yükleme hatası: {e}")

init_db()
init_menu_db()

def urun_bul_ve_duzelt(gelen_urun: str, menu_urunler: List[str]) -> Optional[str]:
    """Yazım hatalı ürünleri menüyle eşleştirir."""
    max_oran = 0
    en_benzer = None
    for menu_urunu in menu_urunler:
        oran = fuzz.token_sort_ratio(gelen_urun.lower(), menu_urunu.lower())
        if oran > max_oran:
            max_oran = oran
            en_benzer = menu_urunu
    return en_benzer if max_oran >= 80 else None

def menu_aktar() -> str:
    """Menü bilgisini önbellekten veya veritabanından alır."""
    if "menu" in menu_cache:
        return menu_cache["menu"]

    try:
        with get_db("neso_menu.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT k.isim, m.ad FROM menu m JOIN kategoriler k ON m.kategori_id = k.id")
            urunler = cursor.fetchall()

        kategorili_menu = {}
        for kategori, urun in urunler:
            kategorili_menu.setdefault(kategori, []).append(urun)

        menu_aciklama = "\n".join([
            f"{kategori}: {', '.join(urunler)}" for kategori, urunler in kategorili_menu.items()
        ])
        result = "Menüde şu ürünler bulunmaktadır:\n" + menu_aciklama
        menu_cache["menu"] = result
        return result
    except Exception as e:
        logger.error(f"Menü yükleme hatası: {e}")
        return "Menü bilgisi şu anda yüklenemedi."

def check_admin(credentials: HTTPBasicCredentials = Depends(security)):
    """Admin kimlik doğrulamasını yapar."""
    if credentials.username != ADMIN_USERNAME or credentials.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Yetkisiz erişim")
    return True

@app.get("/siparisler")
def get_orders(auth: bool = Depends(check_admin)):
    """Tüm siparişleri listeler."""
    with get_db("neso.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT masa, istek, yanit, sepet, zaman FROM siparisler ORDER BY id DESC")
        rows = cursor.fetchall()
    return {
        "orders": [
            {
                "masa": r[0],
                "istek": r[1],
                "yanit": r[2],
                "sepet": r[3],
                "zaman": r[4]
            } for r in rows
        ]
    }

SISTEM_MESAJI = {
    "role": "system",
    "content": (
        "Sen Neso adında Fıstık Kafe için tasarlanmış sesli ve yazılı bir yapay zeka modelisin. "
        "Amacın masalardaki müşterilerin söylediklerinden ne sipariş etmek istediklerini anlamak, ürünleri menüye göre eşleştirerek adetleriyle birlikte kayıt altına almak ve mutfağa iletmektir. "
        "Siparişleri sen hazırlamıyorsun ama doğru şekilde alır ve iletişim kurarsın. "
        "Müşteri '1 saleep', '2 menengiş kahvesi', 'orta şekerli Türk kahvesi istiyorum' gibi ifadeler kullandığında, yazım hatalarını da anlayarak ne istediklerini çıkar ve yanıtla. "
        "Menüde olmayan ürünler için 'üzgünüm menümüzde bu ürün yok' gibi kibar ve bilgilendirici cevaplar ver. "
        "Genel kültür, tarih, siyaset gibi konular sorulursa, 'Ben bir restoran sipariş asistanıyım, bu konuda yardımcı olamam 😊' şeklinde yanıt ver. "
        "Her zaman sıcak, kibar, çözüm odaklı ve samimi ol. Menü şu şekildedir:\n\n"
    )
}

@app.post("/yanitla")
async def yanitla(data: Dict = Body(...)):
    """Müşteri mesajına OpenAI ile yanıt üretir."""
    mesaj = data.get("text", "")
    masa = data.get("masa", "bilinmiyor")
    logger.info(f"[Masa {masa}] Mesaj geldi: {mesaj}")

    try:
        messages = [
            {**SISTEM_MESAJI, "content": SISTEM_MESAJI["content"] + menu_aktar()},
            {"role": "user", "content": mesaj}
        ]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7
        )
        reply = response.choices[0].message.content.strip()
        return {"reply": reply}
    except Exception as e:
        logger.error(f"OpenAI yanıt hatası: {e}")
        return {"reply": "Bir hata oluştu, lütfen tekrar deneyin."}

@app.get("/menu")
def get_menu():
    """Menüyü JSON formatında döndürür."""
    try:
        with get_db("neso_menu.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, isim FROM kategoriler")
            kategoriler = cursor.fetchall()
            full_menu = []
            for kategori_id, kategori_adi in kategoriler:
                cursor.execute("SELECT ad, fiyat FROM menu WHERE kategori_id = ?", (kategori_id,))
                urunler = cursor.fetchall()
                full_menu.append({
                    "kategori": kategori_adi,
                    "urunler": [{"ad": u[0], "fiyat": u[1]} for u in urunler]
                })
        return {"menu": full_menu}
    except Exception as e:
        logger.error(f"Menü getirme hatası: {e}")
        return {"error": "Menü yüklenemedi."}

@app.post("/menu-yukle-csv")
async def menu_yukle_csv(dosya: UploadFile = File(...)):
    """CSV dosyasından menü yükler."""
    try:
        contents = await dosya.read()
        text = contents.decode("utf-8").splitlines()
        reader = csv.DictReader(text)
        with get_db("neso_menu.db") as conn:
            cursor = conn.cursor()
            for row in reader:
                urun = row["urun"]
                fiyat = float(row["fiyat"])
                kategori = row["kategori"]
                cursor.execute("INSERT OR IGNORE INTO kategoriler (isim) VALUES (?)", (kategori,))
                cursor.execute("SELECT id FROM kategoriler WHERE isim = ?", (kategori,))
                kategori_id = cursor.fetchone()[0]
                cursor.execute("INSERT INTO menu (ad, fiyat, kategori_id) VALUES (?, ?, ?)", (urun, fiyat, kategori_id))
            conn.commit()
        return {"mesaj": "CSV'den menü başarıyla yüklendi."}
    except Exception as e:
        logger.error(f"CSV yükleme hatası: {e}")
        return {"hata": "CSV yüklenemedi."}

@app.post("/menu/ekle")
async def menu_ekle(veri: MenuItem):
    """Menüye yeni ürün ekler."""
    try:
        with get_db("neso_menu.db") as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR IGNORE INTO kategoriler (isim) VALUES (?)", (veri.kategori,))
            cursor.execute("SELECT id FROM kategoriler WHERE isim = ?", (veri.kategori,))
            kategori_id = cursor.fetchone()[0]
            cursor.execute("INSERT INTO menu (ad, fiyat, kategori_id) VALUES (?, ?, ?)", (veri.ad, veri.fiyat, kategori_id))
            conn.commit()
        menu_cache.clear()  # Menü değişti, önbelleği temizle
        return {"mesaj": f"{veri.ad} başarıyla eklendi."}
    except Exception as e:
        logger.error(f"Menü ekleme hatası: {e}")
        return {"hata": "Ürün eklenemedi."}

@app.delete("/menu/sil")
async def menu_sil(urun_adi: str = Query(...)):
    """Menüden ürün siler."""
    try:
        with get_db("neso_menu.db") as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM menu WHERE ad = ?", (urun_adi,))
            conn.commit()
        menu_cache.clear()  # Menü değişti, önbelleği temizle
        return {"mesaj": f"{urun_adi} başarıyla silindi."}
    except Exception as e:
        logger.error(f"Menü silme hatası: {e}")
        return {"hata": "Ürün silinemedi."}

def istatistik_hesapla(veriler: List[tuple]) -> tuple[int, float]:
    """Sipariş istatistiklerini hesaplar."""
    fiyatlar = menu_fiyat_sozlugu()
    toplam_siparis = 0
    toplam_tutar = 0
    for (sepet_json,) in veriler:
        try:
            urunler = json.loads(sepet_json)
            for u in urunler:
                adet = u.get("adet", 1)
                urun_adi = u.get("urun", "").lower().strip()
                fiyat = fiyatlar.get(urun_adi, 0)
                toplam_siparis += adet
                toplam_tutar += adet * fiyat
        except Exception as e:
            logger.warning(f"İstatistik hesaplama hatası: {e}")
            continue
    return toplam_siparis, toplam_tutar

def menu_fiyat_sozlugu() -> Dict[str, float]:
    """Menü fiyatlarını bir sözlük olarak döndürür."""
    try:
        with get_db("neso_menu.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT LOWER(TRIM(ad)), fiyat FROM menu")
            veriler = cursor.fetchall()
        return {ad: fiyat for ad, fiyat in veriler}
    except Exception as e:
        logger.error(f"Menü fiyat sözlüğü hatası: {e}")
        return {}

@app.api_route("/siparisler/ornek", methods=["GET", "POST"])
def ornek_siparis_ekle():
    """Örnek bir sipariş ekler."""
    try:
        with get_db("neso.db") as conn:
            cursor = conn.cursor()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sepet = json.dumps([
                {"urun": "Çay", "adet": 2, "fiyat": 20},
                {"urun": "Türk Kahvesi", "adet": 1, "fiyat": 75}
            ])
            cursor.execute(
                """
                INSERT INTO siparisler (masa, istek, yanit, sepet, zaman)
                VALUES (?, ?, ?, ?, ?)
                """,
                ("1", "Çay ve kahve istiyoruz", "Siparişiniz alındı", sepet, now)
            )
            conn.commit()
        return {"mesaj": "✅ Örnek sipariş başarıyla eklendi."}
    except Exception as e:
        logger.error(f"Örnek sipariş ekleme hatası: {e}")
        raise HTTPException(status_code=500, detail="Örnek sipariş eklenemedi.")

@app.get("/istatistik/en-cok-satilan")
def populer_urunler():
    """En çok satılan ürünleri listeler."""
    try:
        with get_db("neso.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT sepet FROM siparisler")
            veriler = cursor.fetchall()
        sayac = {}
        for (sepet_json,) in veriler:
            if not sepet_json:
                continue
            try:
                urunler = json.loads(sepet_json)
                for u in urunler:
                    isim = u.get("urun")
                    if not isim:
                        continue
                    adet = u.get("adet", 1)
                    sayac[isim] = sayac.get(isim, 0) + adet
            except Exception as e:
                logger.warning(f"JSON parse hatası: {e}")
                continue
        en_cok = sorted(sayac.items(), key=lambda x: x[1], reverse=True)[:5]
        return [{"urun": u, "adet": a} for u, a in en_cok]
    except Exception as e:
        logger.error(f"Popüler ürünler hatası: {e}")
        raise HTTPException(status_code=500, detail="Popüler ürünler alınamadı.")

@app.get("/istatistik/gunluk")
def gunluk_istatistik():
    """Günlük istatistikleri döndürür."""
    bugun = datetime.now().strftime("%Y-%m-%d")
    with get_db("neso.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT sepet FROM siparisler WHERE zaman LIKE ?", (f"{bugun}%",))
        veriler = cursor.fetchall()
    siparis_sayisi, gelir = istatistik_hesapla(veriler)
    return {"tarih": bugun, "siparis_sayisi": siparis_sayisi, "gelir": gelir}

@app.get("/istatistik/aylik")
def aylik_istatistik():
    """Aylık istatistikleri döndürür."""
    baslangic = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    with get_db("neso.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT sepet FROM siparisler WHERE zaman >= ?", (baslangic,))
        veriler = cursor.fetchall()
    siparis_sayisi, gelir = istatistik_hesapla(veriler)
    return {"baslangic": baslangic, "siparis_sayisi": siparis_sayisi, "gelir": gelir}

@app.get("/istatistik/yillik")
def yillik_istatistik():
    """Yıllık istatistikleri döndürür."""
    with get_db("neso.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT zaman, sepet FROM siparisler")
        veriler = cursor.fetchall()
    aylik = {}
    for zaman, sepet_json in veriler:
        try:
            ay = zaman[:7]
            urunler = json.loads(sepet_json)
            adet = sum([u.get("adet", 1) for u in urunler])
            aylik[ay] = aylik.get(ay, 0) + adet
        except Exception as e:
            logger.warning(f"Yıllık istatistik hatası: {e}")
            continue
    return dict(sorted(aylik.items()))

@app.get("/istatistik/filtreli")
def filtreli_istatistik(baslangic: str = Query(...), bitis: str = Query(...)):
    """Belirtilen tarih aralığı için istatistikleri döndürür."""
    with get_db("neso.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT sepet FROM siparisler WHERE zaman BETWEEN ? AND ?", (baslangic, bitis))
        veriler = cursor.fetchall()
    siparis_sayisi, gelir = istatistik_hesapla(veriler)
    return {"aralik": f"{baslangic} → {bitis}", "siparis_sayisi": siparis_sayisi, "gelir": gelir}

@app.post("/sesli-yanit")
async def sesli_yanit(data: Dict = Body(...)):
    """Metni sese çevirir ve MP3 olarak döndürür."""
    metin = data.get("text", "")
    if not metin.strip():
        raise HTTPException(status_code=400, detail="Metin boş olamaz.")

    logger.info(f"Sesli yanıt istendi: {metin}")
    try:
        tts_client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=metin)
        voice = texttospeech.VoiceSelectionParams(
            language_code="tr-TR",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.3
        )
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        return Response(content=response.audio_content, media_type="audio/mpeg")
    except Exception as e:
        logger.error(f"Sesli yanıt hatası: {e}")
        raise HTTPException(status_code=500, detail="Sesli yanıt oluşturulamadı.")