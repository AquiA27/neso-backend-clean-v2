from fastapi import FastAPI, Request, Body, Query, UploadFile, File, HTTPException, status, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.middleware.sessions import SessionMiddleware
import os
import base64
import tempfile
import sqlite3
import json
import csv
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
from openai import OpenAI
from google.cloud import texttospeech
import logging

# 🌍 Ortam değişkenleri
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CREDS_BASE64 = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_BASE64")

# Hassas bilgileri doğrulama
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY ortam değişkeni eksik.")
if not GOOGLE_CREDS_BASE64:
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS_BASE64 ortam değişkeni eksik.")

# Google Cloud kimlik bilgilerini ayarla
decoded = base64.b64decode(GOOGLE_CREDS_BASE64)
with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
    tmp.write(decoded)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()
security = HTTPBasic()

# CORS yapılandırması
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # İdealde spesifik domainler eklenmeli
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Aktif kullanıcılar ve mutfak WebSocket bağlantıları
aktif_mutfak_websocketleri = set()
aktif_kullanicilar = {}

# Log yapılandırması
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Emoji temizleme fonksiyonu
def temizle_emoji(text):
    import re
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Middleware ile aktif kullanıcı takibi
@app.middleware("http")
async def aktif_kullanici_takibi(request: Request, call_next):
    ip = request.client.host
    agent = request.headers.get("user-agent", "")
    kimlik = f"{ip}_{agent}"
    aktif_kullanicilar[kimlik] = datetime.now()
    response = await call_next(request)
    return response

@app.get("/istatistik/online")
def online_kullanici_sayisi():
    su_an = datetime.now()
    aktifler = [kimlik for kimlik, zaman in aktif_kullanicilar.items() if (su_an - zaman).seconds < 300]
    return {"count": len(aktifler)}


@app.websocket("/ws/mutfak")
async def websocket_mutfak(websocket: WebSocket):
    await websocket.accept()
    aktif_mutfak_websocketleri.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        aktif_mutfak_websocketleri.discard(websocket)

# Siparişleri mutfağa gönder
async def mutfaga_gonder(siparis):
    for ws in list(aktif_mutfak_websocketleri):  # Set'i listeye çevirerek güvenli iterate
        try:
            await ws.send_text(json.dumps(siparis))
        except Exception as e:
            logging.warning(f"Mutfak WebSocket gönderim hatası: {e}")
            aktif_mutfak_websocketleri.discard(ws)

@app.post("/siparis-ekle")
async def siparis_ekle(data: dict = Body(...)):
    logging.info(f"📥 Yeni sipariş geldi: {data}")
    masa = data.get("masa")
    yanit = data.get("yanit")
    sepet_verisi = data.get("sepet", [])
    zaman = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not masa:
        raise HTTPException(status_code=400, detail="Masa bilgisi eksik.")

    # İstek metni sepetten oluşturulsun
    try:
        istek = ", ".join([f"{item.get('urun', '').strip()} ({item.get('adet', 1)} adet)" for item in sepet_verisi])
    except Exception as e:
        istek = "Tanımsız"

    try:
        sepet_json = json.dumps(sepet_verisi)
        with sqlite3.connect("neso.db") as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO siparisler (masa, istek, yanit, sepet, zaman)
                VALUES (?, ?, ?, ?, ?)
            """, (masa, istek, yanit, sepet_json, zaman))
            conn.commit()

        await mutfaga_gonder({
            "masa": masa,
            "istek": istek,
            "yanit": yanit,
            "sepet": sepet_json,
            "zaman": zaman
        })

        return {"mesaj": "Sipariş başarıyla kaydedildi ve mutfağa iletildi."}
    except Exception as e:
        logging.error(f"Sipariş ekleme hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Sipariş eklenemedi: {e}")


# Veritabanı başlatma
def init_db():
    with sqlite3.connect("neso.db") as conn:
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
    yeni_olustu = not os.path.exists("neso_menu.db")
    with sqlite3.connect("neso_menu.db") as conn:
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
                logging.error(f"❌ CSV otomatik yükleme hatası: {e}")

init_db()
init_menu_db()

# ✨ OpenAI modele menü aktarım fonksiyonu

# 🔍 Fuzzy ürün eşleştirme
def urun_bul_ve_duzelt(gelen_urun, menu_urunler):
    max_oran = 0
    en_benzer = None
    for menu_urunu in menu_urunler:
        oran = fuzz.token_sort_ratio(gelen_urun.lower(), menu_urunu.lower())
        if oran > max_oran:
            max_oran = oran
            en_benzer = menu_urunu
    if max_oran >= 80:
        return en_benzer
    return None

def menu_aktar():
    try:
        conn = sqlite3.connect("neso_menu.db")
        cursor = conn.cursor()
        cursor.execute("SELECT k.isim, m.ad FROM menu m JOIN kategoriler k ON m.kategori_id = k.id")
        urunler = cursor.fetchall()
        conn.close()
        kategorili_menu = {}
        for kategori, urun in urunler:
            kategorili_menu.setdefault(kategori, []).append(urun)

        menu_aciklama = "\n".join([
            f"{kategori}: {', '.join(urunler)}" for kategori, urunler in kategorili_menu.items()
        ])
        return "Menüde şu ürünler bulunmaktadır:\n" + menu_aciklama
    except:
        return "Menü bilgisi şu anda yüklenemedi."

# ✅ Admin Yetkisi Kontrol
def check_admin(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = os.getenv("ADMIN_USERNAME", "admin")
    correct_password = os.getenv("ADMIN_PASSWORD", "admin123")
    if credentials.username != correct_username or credentials.password != correct_password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Yetkisiz erişim")
    return True

# 🔍 Siparişleri Listele
@app.get("/siparisler")
def get_orders(auth: bool = Depends(check_admin)):
    conn = sqlite3.connect("neso.db")
    cursor = conn.cursor()
    cursor.execute("SELECT masa, istek, yanit, sepet, zaman FROM siparisler ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
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

# 🔊 OpenAI Yanıt Üretici
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
        + menu_aktar()
    )
}

@app.post("/yanitla")
async def yanitla(data: dict = Body(...)):
    mesaj = data.get("text", "")
    masa = data.get("masa", "bilinmiyor")
    print(f"[Masa {masa}] mesaj geldi: {mesaj}")
    reply = cevap_uret(mesaj)
    return {"reply": reply}

def cevap_uret(mesaj: str) -> str:
    try:
        messages = [
    SISTEM_MESAJI,
    {"role": "system", "content": menu_aktar()},
    {"role": "user", "content": mesaj}
]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "🚨 Bir hata oluştu: " + str(e)
# 🧾 Menü Getir
@app.get("/menu")
def get_menu():
    try:
        conn = sqlite3.connect("neso_menu.db")
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
        conn.close()
        return {"menu": full_menu}
    except Exception as e:
        return {"error": str(e)}

# 📥 Menü Yükle CSV
@app.post("/menu-yukle-csv")
async def menu_yukle_csv(dosya: UploadFile = File(...)):
    try:
        contents = await dosya.read()
        text = contents.decode("utf-8").splitlines()
        reader = csv.DictReader(text)
        conn = sqlite3.connect("neso_menu.db")
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
        conn.close()
        return {"mesaj": "CSV'den menü başarıyla yüklendi."}
    except Exception as e:
        return {"hata": str(e)}

# ➕ Menüye Ürün Ekle
@app.post("/menu/ekle")
async def menu_ekle(veri: dict = Body(...)):
    try:
        urun = veri.get("ad")
        fiyat = float(veri.get("fiyat"))
        kategori = veri.get("kategori")
        if not urun or not kategori:
            return {"hata": "Ürün adı ve kategori zorunludur."}
        conn = sqlite3.connect("neso_menu.db")
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO kategoriler (isim) VALUES (?)", (kategori,))
        cursor.execute("SELECT id FROM kategoriler WHERE isim = ?", (kategori,))
        kategori_id = cursor.fetchone()[0]
        cursor.execute("INSERT INTO menu (ad, fiyat, kategori_id) VALUES (?, ?, ?)", (urun, fiyat, kategori_id))
        conn.commit()
        conn.close()
        return {"mesaj": f"{urun} başarıyla eklendi."}
    except Exception as e:
        return {"hata": str(e)}

# ❌ Menüden Ürün Sil
@app.delete("/menu/sil")
async def menu_sil(urun_adi: str = Query(...)):
    try:
        conn = sqlite3.connect("neso_menu.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM menu WHERE ad = ?", (urun_adi,))
        conn.commit()
        conn.close()
        return {"mesaj": f"{urun_adi} başarıyla silindi."}
    except Exception as e:
        return {"hata": str(e)}

# 📊 Yardımcı İstatistik Hesaplayıcı
def istatistik_hesapla(veriler):
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
        except:
            continue
    return toplam_siparis, toplam_tutar

def menu_fiyat_sozlugu():
    try:
        conn = sqlite3.connect("neso_menu.db")
        cursor = conn.cursor()
        cursor.execute("SELECT LOWER(TRIM(ad)), fiyat FROM menu")
        veriler = cursor.fetchall()
        conn.close()
        return {ad: fiyat for ad, fiyat in veriler}
    except Exception as e:
        print("💥 Menü fiyat sözlüğü hatası:", e)
        return {}


@app.api_route("/siparisler/ornek", methods=["GET", "POST"])
def ornek_siparis_ekle():
    try:
        conn = sqlite3.connect("neso.db")
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sepet = json.dumps([
            {"urun": "Çay", "adet": 2, "fiyat": 20},
            {"urun": "Türk Kahvesi", "adet": 1, "fiyat": 75}
        ])
        cursor.execute("""
            INSERT INTO siparisler (masa, istek, yanit, sepet, zaman)
            VALUES (?, ?, ?, ?, ?)
        """, ("1", "Çay ve kahve istiyoruz", "Siparişiniz alındı", sepet, now))
        conn.commit()
        conn.close()
        return {"mesaj": "✅ Örnek sipariş başarıyla eklendi."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ En Çok Satılan Ürünler - Hatalara Dayanıklı
@app.get("/istatistik/en-cok-satilan")
def populer_urunler():
    try:
        conn = sqlite3.connect("neso.db")
        cursor = conn.cursor()
        cursor.execute("SELECT sepet FROM siparisler")
        veriler = cursor.fetchall()
        sayac = {}
        for (sepet_json,) in veriler:
            if not sepet_json:
                continue  # boş veri varsa geç
            try:
                urunler = json.loads(sepet_json)
                for u in urunler:
                    isim = u.get("urun")
                    if not isim:
                        continue
                    adet = u.get("adet", 1)
                    sayac[isim] = sayac.get(isim, 0) + adet
            except Exception as e:
                print("🚨 JSON parse hatası:", e)
                continue
        en_cok = sorted(sayac.items(), key=lambda x: x[1], reverse=True)[:5]
        return [{"urun": u, "adet": a} for u, a in en_cok]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hata: {e}")

@app.get("/istatistik/gunluk")
def gunluk_istatistik():
    bugun = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect("neso.db")
    cursor = conn.cursor()
    cursor.execute("SELECT sepet FROM siparisler WHERE zaman LIKE ?", (f"{bugun}%",))
    veriler = cursor.fetchall()
    siparis_sayisi, gelir = istatistik_hesapla(veriler)
    return {"tarih": bugun, "siparis_sayisi": siparis_sayisi, "gelir": gelir}

@app.get("/istatistik/aylik")
def aylik_istatistik():
    baslangic = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    conn = sqlite3.connect("neso.db")
    cursor = conn.cursor()
    cursor.execute("SELECT sepet FROM siparisler WHERE zaman >= ?", (baslangic,))
    veriler = cursor.fetchall()
    siparis_sayisi, gelir = istatistik_hesapla(veriler)
    return {"baslangic": baslangic, "siparis_sayisi": siparis_sayisi, "gelir": gelir}

@app.get("/istatistik/yillik")
def yillik_istatistik():
    conn = sqlite3.connect("neso.db")
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
        except:
            continue
    return dict(sorted(aylik.items()))

@app.get("/istatistik/filtreli")
def filtreli_istatistik(baslangic: str = Query(...), bitis: str = Query(...)):
    conn = sqlite3.connect("neso.db")
    cursor = conn.cursor()
    cursor.execute("SELECT sepet FROM siparisler WHERE zaman BETWEEN ? AND ?", (baslangic, bitis))
    veriler = cursor.fetchall()
    siparis_sayisi, gelir = istatistik_hesapla(veriler)
    return {"aralik": f"{baslangic} → {bitis}", "siparis_sayisi": siparis_sayisi, "gelir": gelir}

# 🔊 Google Text-to-Speech Sesli Yanıt
@app.post("/sesli-yanit")
async def sesli_yanit(data: dict = Body(...)):
    metin = data.get("text", "")
    try:
        if not metin.strip():
            raise ValueError("Metin boş geldi.")
        print("🟡 Sesli yanıt istendi. Metin:", metin)

        tts_client = texttospeech.TextToSpeechClient()
        cleaned_text = temizle_emoji(metin)
        synthesis_input = texttospeech.SynthesisInput(text=cleaned_text)
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

        print("✅ Sesli yanıt başarıyla oluşturuldu.")
        return Response(content=response.audio_content, media_type="audio/mpeg")

    except Exception as e:
        print("❌ SESLİ YANIT HATASI:", str(e))
        raise HTTPException(status_code=500, detail=f"Sesli yanıt hatası: {e}")
