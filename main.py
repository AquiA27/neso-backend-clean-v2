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
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
from openai import OpenAI
from google.cloud import texttospeech

# 🌍 Load environment variables and configure logging
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def temizle_emoji(text):
    import re
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CREDS_BASE64 = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_BASE64")

if GOOGLE_CREDS_BASE64:
    decoded = base64.b64decode(GOOGLE_CREDS_BASE64)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(decoded)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()
security = HTTPBasic()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", "your-secret-key-here"),
    session_cookie="session"
)

aktif_mutfak_websocketleri = []
aktif_kullanicilar = {}

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
    aktif_mutfak_websocketleri.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"WebSocket message received: {data}")
    except WebSocketDisconnect:
        aktif_mutfak_websocketleri.remove(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        if websocket in aktif_mutfak_websocketleri:
            aktif_mutfak_websocketleri.remove(websocket)

async def mutfaga_gonder(siparis):
    for ws in aktif_mutfak_websocketleri[:]:  # Copy list to avoid modification during iteration
        try:
            await ws.send_text(json.dumps(siparis))
        except Exception as e:
            logger.error(f"Failed to send order to kitchen: {str(e)}")
            if ws in aktif_mutfak_websocketleri:
                aktif_mutfak_websocketleri.remove(ws)

def check_admin(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = os.getenv("ADMIN_USERNAME", "admin")
    correct_password = os.getenv("ADMIN_PASSWORD", "admin123")
    if credentials.username != correct_username or credentials.password != correct_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True

@app.post("/siparis-ekle")
async def siparis_ekle(data: dict = Body(...)):
    logger.info(f"📥 New order received: {data}")
    masa = data.get("masa")
    yanit = data.get("yanit")
    sepet_verisi = data.get("sepet", [])
    zaman = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not masa:
        raise HTTPException(status_code=400, detail="Masa bilgisi eksik.")

    try:
        istek = ", ".join([f"{item.get('urun', '').strip()} ({item.get('adet', 1)} adet)" for item in sepet_verisi])
    except Exception as e:
        logger.error(f"Error creating order text: {str(e)}")
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
            "sepet": sepet_verisi,
            "zaman": zaman
        })

        return {"mesaj": "Sipariş başarıyla kaydedildi ve mutfağa iletildi."}
    except Exception as e:
        logger.error(f"Order creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Sipariş eklenemedi: {str(e)}")

@app.get("/siparisler")
def get_orders(auth: bool = Depends(check_admin)):
    try:
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
    except Exception as e:
        logger.error(f"Error fetching orders: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def init_db():
    try:
        conn = sqlite3.connect("neso.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS siparisler (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                masa TEXT,
                istek TEXT,
                yanit TEXT,
                sepet TEXT,
                zaman TEXT
            )
        """)
        conn.commit()
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        raise
    finally:
        conn.close()

def init_menu_db():
    try:
        conn = sqlite3.connect("neso_menu.db")
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
    except Exception as e:
        logger.error(f"Menu database initialization error: {str(e)}")
        raise
    finally:
        conn.close()

init_db()
init_menu_db()

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
    except Exception as e:
        logger.error(f"Menu transfer error: {str(e)}")
        return "Menü bilgisi şu anda yüklenemedi."

def menu_fiyat_sozlugu():
    try:
        conn = sqlite3.connect("neso_menu.db")
        cursor = conn.cursor()
        cursor.execute("SELECT LOWER(TRIM(ad)), fiyat FROM menu")
        veriler = cursor.fetchall()
        conn.close()
        return {ad: fiyat for ad, fiyat in veriler}
    except Exception as e:
        logger.error(f"Menu price dictionary error: {str(e)}")
        return {}

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

SISTEM_MESAJI = {
    "role": "system",
    "content": (
        "Sen Neso adında Fıstık Kafe için tasarlanmış sesli ve yazılı bir yapay zeka modelisin. "
        "Amacın masalardaki müşterilerin söylediklerinden ne sipariş etmek istediklerini anlamak, "
        "ürünleri menüye göre eşleştirerek adetleriyle birlikte kayıt altına almak ve mutfağa iletmektir. "
        "Siparişleri sen hazırlamıyorsun ama doğru şekilde alır ve iletişim kurarsın. "
        "Her zaman sıcak, kibar, çözüm odaklı ve samimi ol. Menü şu şekildedir:\n\n"
        + menu_aktar()
    )
}

@app.post("/yanitla")
async def yanitla(data: dict = Body(...)):
    mesaj = data.get("text", "")
    masa = data.get("masa", "bilinmiyor")
    logger.info(f"[Masa {masa}] New message: {mesaj}")
    reply = cevap_uret(mesaj)
    return {"reply": reply}

def cevap_uret(mesaj: str) -> str:
    try:
        messages = [
            SISTEM_MESAJI,
            {"role": "user", "content": mesaj}
        ]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Response generation error: {str(e)}")
        return "🚨 Bir hata oluştu: " + str(e)

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
        logger.error(f"Menu retrieval error: {str(e)}")
        return {"error": str(e)}

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
            cursor.execute("INSERT INTO menu (ad, fiyat, kategori_id) VALUES (?, ?, ?)", 
                         (urun, fiyat, kategori_id))
        
        conn.commit()
        conn.close()
        return {"mesaj": "CSV'den menü başarıyla yüklendi."}
    except Exception as e:
        logger.error(f"CSV menu upload error: {str(e)}")
        return {"hata": str(e)}

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
        cursor.execute("INSERT INTO menu (ad, fiyat, kategori_id) VALUES (?, ?, ?)", 
                     (urun, fiyat, kategori_id))
        
        conn.commit()
        conn.close()
        return {"mesaj": f"{urun} başarıyla eklendi."}
    except Exception as e:
        logger.error(f"Menu item addition error: {str(e)}")
        return {"hata": str(e)}

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
        logger.error(f"Menu item deletion error: {str(e)}")
        return {"hata": str(e)}

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
                logger.error(f"JSON parse error in popular items: {str(e)}")
                continue
                
        en_cok = sorted(sayac.items(), key=lambda x: x[1], reverse=True)[:5]
        return [{"urun": u, "adet": a} for u, a in en_cok]
    except Exception as e:
        logger.error(f"Popular items calculation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Hata: {str(e)}")

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
            ay = zaman[:7]  # YYYY-MM formatında ay bilgisi
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

@app.post("/sesli-yanit")
async def sesli_yanit(data: dict = Body(...)):
    metin = data.get("text", "")
    try:
        if not metin.strip():
            raise ValueError("Metin boş geldi.")
        logger.info("🟡 Voice response requested. Text: %s", metin)

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

        logger.info("✅ Voice response successfully created")
        return Response(content=response.audio_content, media_type="audio/mpeg")

    except Exception as e:
        logger.error("❌ VOICE RESPONSE ERROR: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Sesli yanıt hatası: {str(e)}")

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
        logger.error(f"Sample order creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))