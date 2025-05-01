from fastapi import FastAPI, Request, Body, Query, UploadFile, File, HTTPException, status, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.middleware.sessions import SessionMiddleware
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
from openai import OpenAI
from google.cloud import texttospeech
import os
import base64
import tempfile
import sqlite3
import json
import csv
import re
import io

# 🌍 Ortam değişkenleri
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CREDS_BASE64 = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_BASE64")

if GOOGLE_CREDS_BASE64:
    decoded = base64.b64decode(GOOGLE_CREDS_BASE64)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(decoded)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()

# 🌐 CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Gerekirse domain bazlı kısıtlanabilir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔐 Basic Auth
security = HTTPBasic()

# ✅ Emoji Temizleyici
def temizle_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# 🔊 Google TTS
def google_sesli_yanit(text):
    temiz_text = temizle_emoji(text)
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=temiz_text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="tr-TR",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.3,
        pitch=0,
    )
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    return response.audio_content

# 🔊 OpenAI TTS
def openai_sesli_yanit(text):
    temiz_text = temizle_emoji(text)
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=temiz_text,
    )
    return response.content

# 📦 Menü Veritabanı Bağlantısı
def get_menu_db():
    return sqlite3.connect("neso_menu.db")

# 📦 Ana Sipariş Veritabanı
def get_siparis_db():
    return sqlite3.connect("neso.db")

# 📂 Menü Listeleme
@app.get("/menu")
def menu_listele():
    conn = get_menu_db()
    cursor = conn.cursor()
    cursor.execute("SELECT urun, fiyat, kategori FROM menu")
    rows = cursor.fetchall()
    conn.close()

    menu = {}
    for urun, fiyat, kategori in rows:
        if kategori not in menu:
            menu[kategori] = []
        menu[kategori].append({"urun": urun, "fiyat": fiyat})
    return menu

# ➕ Ürün Ekle
@app.post("/menu/ekle")
def menu_ekle(veri: dict):
    urun = veri.get("urun")
    fiyat = veri.get("fiyat")
    kategori = veri.get("kategori")
    if not urun or not fiyat or not kategori:
        raise HTTPException(status_code=400, detail="Eksik bilgi")

    conn = get_menu_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO menu (urun, fiyat, kategori) VALUES (?, ?, ?)",
        (urun.strip(), float(fiyat), kategori.strip()),
    )
    conn.commit()
    conn.close()
    return {"mesaj": "Ürün eklendi"}

# ❌ Ürün Sil
@app.post("/menu/sil")
def menu_sil(veri: dict):
    urun = veri.get("urun")
    if not urun:
        raise HTTPException(status_code=400, detail="Ürün adı gerekli")

    conn = get_menu_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM menu WHERE urun = ?", (urun.strip(),))
    conn.commit()
    conn.close()
    return {"mesaj": "Ürün silindi"}

# 📝 Ürün Güncelle
@app.post("/menu/guncelle")
def menu_guncelle(veri: dict):
    eski_urun = veri.get("eski_urun")
    yeni_urun = veri.get("yeni_urun")
    fiyat = veri.get("fiyat")
    kategori = veri.get("kategori")

    if not eski_urun:
        raise HTTPException(status_code=400, detail="Eski ürün adı gerekli")

    conn = get_menu_db()
    cursor = conn.cursor()
    if yeni_urun:
        cursor.execute("UPDATE menu SET urun = ? WHERE urun = ?", (yeni_urun.strip(), eski_urun.strip()))
    if fiyat:
        cursor.execute("UPDATE menu SET fiyat = ? WHERE urun = ?", (float(fiyat), yeni_urun or eski_urun))
    if kategori:
        cursor.execute("UPDATE menu SET kategori = ? WHERE urun = ?", (kategori.strip(), yeni_urun or eski_urun))
    conn.commit()
    conn.close()
    return {"mesaj": "Ürün güncellendi"}

# 🧾 Kategori Listeleme
@app.get("/menu/kategoriler")
def kategori_listele():
    conn = get_menu_db()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT kategori FROM menu")
    kategoriler = [row[0] for row in cursor.fetchall()]
    conn.close()
    return {"kategoriler": kategoriler}

# 📁 CSV'den Menü Yükleme
@app.post("/menu-yukle-csv")
async def menu_yukle_csv(dosya: UploadFile = File(...)):
    conn = get_menu_db()
    cursor = conn.cursor()
    icerik = await dosya.read()
    satirlar = icerik.decode().splitlines()
    csv_reader = csv.DictReader(satirlar)

    for row in csv_reader:
        urun = row["urun"]
        fiyat = float(row["fiyat"])
        kategori = row["kategori"]
        cursor.execute(
            "INSERT OR IGNORE INTO menu (urun, fiyat, kategori) VALUES (?, ?, ?)",
            (urun.strip(), fiyat, kategori.strip()),
        )

    conn.commit()
    conn.close()
    return {"mesaj": "CSV'den menü başarıyla yüklendi"}
# 📥 Sipariş Ekle
@app.post("/siparis-ekle")
async def siparis_ekle(request: Request):
    veri = await request.json()
    masa = veri.get("masa")
    sepet = veri.get("sepet")
    mesaj = veri.get("mesaj", "")
    yanit = veri.get("yanit", "")

    if not sepet or not isinstance(sepet, list) or not masa:
        raise HTTPException(status_code=400, detail="Geçersiz sipariş")

    # Sepetten istek metni üret
    istek = ", ".join([f"{item['adet']} adet {item['urun']}" for item in sepet])

    conn = get_siparis_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO siparisler (masa, istek, mesaj, yanit, tarih, durum) VALUES (?, ?, ?, ?, ?, ?)",
        (masa, istek, mesaj, yanit, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "hazırlanıyor"),
    )
    conn.commit()
    conn.close()
    return {"mesaj": "Sipariş kaydedildi"}

# 🔄 Sipariş Durumu Güncelle
@app.post("/siparis-durum")
def siparis_durum_guncelle(veri: dict):
    siparis_id = veri.get("id")
    yeni_durum = veri.get("durum")
    if not siparis_id or not yeni_durum:
        raise HTTPException(status_code=400, detail="Eksik bilgi")

    conn = get_siparis_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE siparisler SET durum = ? WHERE id = ?", (yeni_durum, siparis_id))
    conn.commit()
    conn.close()
    return {"mesaj": "Sipariş durumu güncellendi"}

# ❌ Sipariş İptal
@app.post("/siparis-iptal")
def siparis_iptal(veri: dict):
    siparis_id = veri.get("id")
    if not siparis_id:
        raise HTTPException(status_code=400, detail="ID gerekli")

    conn = get_siparis_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE siparisler SET durum = 'iptal edildi' WHERE id = ?", (siparis_id,))
    conn.commit()
    conn.close()
    return {"mesaj": "Sipariş iptal edildi"}

# 👥 Online Kullanıcı Takibi
aktif_kullanicilar = {}

@app.middleware("http")
async def kullanici_takibi_middleware(request: Request, call_next):
    ip = request.client.host
    aktif_kullanicilar[ip] = datetime.now()
    yanit = await call_next(request)
    return yanit

@app.get("/online-kullanicilar")
def online_kullanicilar():
    simdi = datetime.now()
    aktif = [ip for ip, zaman in aktif_kullanicilar.items() if (simdi - zaman).seconds < 120]
    return {"online": len(aktif)}

# 📊 İstatistik: Günlük, Aylık, Yıllık
def istatistik_hesapla(zaman_format):
    conn = get_siparis_db()
    cursor = conn.cursor()
    cursor.execute("SELECT istek, tarih FROM siparisler")
    veriler = cursor.fetchall()
    conn.close()

    urun_sayac = {}
    for istek, tarih in veriler:
        zaman = datetime.strptime(tarih, "%Y-%m-%d %H:%M:%S").strftime(zaman_format)
        urunler = re.findall(r"(\d+)\s+adet\s+([\wçğıöşüÇĞİÖŞÜ\s\-]+)", istek, re.IGNORECASE)
        for adet, urun in urunler:
            anahtar = urun.strip().lower()
            if anahtar not in urun_sayac:
                urun_sayac[anahtar] = 0
            urun_sayac[anahtar] += int(adet)
    return urun_sayac

@app.get("/istatistik/gunluk")
def gunluk_istatistik():
    return istatistik_hesapla("%Y-%m-%d")

@app.get("/istatistik/aylik")
def aylik_istatistik():
    return istatistik_hesapla("%Y-%m")

@app.get("/istatistik/yillik")
def yillik_istatistik():
    return istatistik_hesapla("%Y")

@app.get("/istatistik/en-cok-satilan")
def en_cok_satilan():
    veriler = istatistik_hesapla("%Y")
    sirali = sorted(veriler.items(), key=lambda x: x[1], reverse=True)
    return dict(sirali[:10])

# 🔐 Admin Şifre Değiştir
@app.post("/parola-degistir")
def parola_degistir(credentials: HTTPBasicCredentials = Depends(security), yeni_sifre: dict = Body(...)):
    if credentials.username != "admin" or credentials.password != "admin123":
        raise HTTPException(status_code=401, detail="Kimlik doğrulama başarısız")
    if not yeni_sifre.get("sifre"):
        raise HTTPException(status_code=400, detail="Yeni şifre boş olamaz")
    # (Not: Gerçek sistemde şifre veritabanına yazılmalı, burada sabit!)
    return {"mesaj": "Şifre başarıyla değiştirildi (demo)"}

# 💬 Asistan Yanıtı ve Sesli Yanıt
SISTEM_MESAJI = {
    "role": "system",
    "content": "Sen Neso adında Fıstık Kafe için tasarlanmış sesli ve yazılı bir yapay zeka modelisin. "
               "Amacın gelen müşterilerin mutlu memnun şekilde ayrılmalarını sağlamak. Kendine has tarzın ve zekanla "
               "insanların verdiği alakasız tepki ve sorulara mümkün olduğunca saygılı, sınırı aşan durumlarda ise "
               "idareye bildirmeyi bilen bir yapıdasın. Yapay zeka modeli olduğun için insanlar seni sınayacak; "
               "buna mümkün olan en iyi şekilde, sana yaraşır bir şekilde karşılık ver."
}

@app.post("/yanitla")
async def yanitla(veri: dict):
    mesaj = veri.get("mesaj")
    if not mesaj:
        raise HTTPException(status_code=400, detail="Mesaj boş olamaz")

    tamamlayici = client.chat.completions.create(
        model="gpt-4",
        messages=[
            SISTEM_MESAJI,
            {"role": "user", "content": mesaj}
        ]
    )

    yanit = tamamlayici.choices[0].message.content
    return {"yanit": yanit}
