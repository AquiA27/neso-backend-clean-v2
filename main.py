import os
import base64
import tempfile
import sqlite3
import json
import re
import io
import csv
from fastapi import FastAPI, Request, Body, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, Response
from starlette.middleware.sessions import SessionMiddleware
from openai import OpenAI
from dotenv import load_dotenv
from google.cloud import texttospeech
from datetime import datetime, timedelta

# Ortam değişkenlerini yükle
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

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ ONLINE KULLANICI TAKİBİ
app.add_middleware(SessionMiddleware, secret_key="neso_super_secret")
aktif_kullanicilar = set()

@app.middleware("http")
async def aktif_kullanici_takibi(request: Request, call_next):
    ip = request.client.host
    aktif_kullanicilar.add(ip)
    response = await call_next(request)
    return response

@app.get("/istatistik/online")
def online_kullanici_sayisi():
    return {"count": len(aktif_kullanicilar)}

# ✅ Veritabanı oluştur
def init_db():
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
    conn.close()

def init_menu_db():
    if not os.path.exists("neso_menu.db"):
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
        conn.close()
        print("✅ Menü veritabanı oluşturuldu.")

init_db()
init_menu_db()

# Menü çekme
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

# CSV'den Menü Yükleme
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
            cursor.execute(
                "INSERT INTO menu (ad, fiyat, kategori_id) VALUES (?, ?, ?)",
                (urun, fiyat, kategori_id)
            )
        conn.commit()
        conn.close()
        return {"mesaj": "CSV'den menü başarıyla yüklendi."}
    except Exception as e:
        return {"hata": str(e)}

# 🔧 Menüye ürün ekle
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

# ❌ Menüden ürün sil
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

# OpenAI ile model tanımı
SISTEM_MESAJI = {
    "role": "system",
    "content": (
        "Sen Neso adında Fıstık Kafe için tasarlanmış sesli ve yazılı bir yapay zeka modelisin. "
        "Amacın gelen müşterilerin mutlu memnun şekilde ayrılmalarını sağlamak. "
        "Kendine has tarzın ve zekanla insanların verdiği alakasız tepki ve sorulara mümkün olduğunca saygılı "
        "ve sınırı aşan durumlarda ise idareye bildirmeyi bilen bir yapıdasın. "
        "Yapay zeka modeli olduğun için insanlar seni sınayacak; buna mümkün olan en iyi şekilde, sana yaraşır şekilde karşılık ver."
    )
}

# Zaman bazlı istatistik endpointleri
@app.get("/istatistik/filtreli")
def filtreli_istatistik(baslangic: str = Query(...), bitis: str = Query(...)):
    conn = sqlite3.connect("neso.db")
    cursor = conn.cursor()
    cursor.execute("SELECT sepet FROM siparisler WHERE zaman BETWEEN ? AND ?", (baslangic, bitis))
    veriler = cursor.fetchall()
    siparis_sayisi, gelir = istatistik_hesapla(veriler)
    return {"aralik": f"{baslangic} → {bitis}", "siparis_sayisi": siparis_sayisi, "gelir": gelir}

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
        ay = zaman[:7]
        urunler = json.loads(sepet_json)
        adet = sum([u.get("adet", 1) for u in urunler])
        aylik[ay] = aylik.get(ay, 0) + adet
    return dict(sorted(aylik.items()))

@app.get("/istatistik/en-cok-satilan")
def populer_urunler():
    conn = sqlite3.connect("neso.db")
    cursor = conn.cursor()
    cursor.execute("SELECT sepet FROM siparisler")
    veriler = cursor.fetchall()
    sayac = {}
    for (sepet_json,) in veriler:
        urunler = json.loads(sepet_json)
        for u in urunler:
            isim = u.get("urun")
            adet = u.get("adet", 1)
            sayac[isim] = sayac.get(isim, 0) + adet
    en_cok = sorted(sayac.items(), key=lambda x: x[1], reverse=True)[:5]
    return [{"urun": u, "adet": a} for u, a in en_cok]

# İstatistik hesaplama
def istatistik_hesapla(veriler):
    fiyatlar = {
        "çay": 20, "fincan çay": 30, "sahlep (tarçınlı fıstıklı)": 100,
        "bitki çayları (ıhlamur, nane-limon, vb.)": 80, "türk kahvesi": 75,
        "osmanlı kahvesi": 75, "menengiç kahvesi": 85, "süt": 40,
        "nescafe": 80, "nescafe sütlü": 85, "esspresso": 60, "filtre kahve": 75,
        "cappuccino": 90, "mocha (white/classic/caramel)": 100, "latte": 80,
        "sıcak çikolata": 100, "macchiato": 100
    }
    toplam_siparis = 0
    toplam_tutar = 0
    for (sepet_json,) in veriler:
        try:
            urunler = json.loads(sepet_json)
            for u in urunler:
                adet = u.get("adet", 1)
                urun_adi = u.get("urun", "").lower()
                fiyat = fiyatlar.get(urun_adi, 0)
                toplam_siparis += adet
                toplam_tutar += adet * fiyat
        except:
            continue
    return toplam_siparis, toplam_tutar
