import os
import base64
import tempfile
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
from dotenv import load_dotenv
import sqlite3
from datetime import datetime
import json
import re
import io
from google.cloud import texttospeech
from memory import get_memory, add_to_memory

# Ortam değişkenlerini yükle
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CREDS_BASE64 = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_BASE64")

# Google kimlik bilgilerini geçici dosyaya yaz
if GOOGLE_CREDS_BASE64:
    decoded = base64.b64decode(GOOGLE_CREDS_BASE64)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(decoded)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Veritabanı (siparişler)
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

# Menü veritabanı (otomatik oluşturma)
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

        veri = {
            "Sıcak İçecekler": [
                ("Çay", 20), ("Fincan Çay", 30), ("Sahlep (Tarçınlı Fıstıklı)", 100),
                ("Bitki Çayları (Ihlamur, Nane-Limon, vb.)", 80), ("Türk Kahvesi", 75),
                ("Osmanlı Kahvesi", 75), ("Menengiç Kahvesi", 85), ("Süt", 40),
                ("Nescafe", 80), ("Nescafe Sütlü", 85), ("Esspresso", 60), ("Filtre Kahve", 75),
                ("Cappuccino", 90), ("Mocha (White/Classic/Caramel)", 100), ("Latte", 80),
                ("Sıcak Çikolata", 100), ("Macchiato", 100)
            ],
            "Soğuk İçecekler": [
                ("Limonata", 75), ("Cola", 70), ("Fanta", 70), ("Sprite", 70),
                ("Cappy Vişne", 70), ("Cappy Şeftali", 70), ("Cappy Kayısı", 70),
                ("Cappy Karışık", 70), ("Cappy Portakal", 70), ("Fuse Tea Karpuz", 70),
                ("Fuse Tea Şeftali", 70), ("Fuse Tea Limon", 70), ("Cola Zero", 70),
                ("Churchill", 50), ("Taze Sıkılmış Portakal Suyu", 20),
                ("Milkshake (Çilek, Muz, vs.)", 85), ("Ice Mocha (Classic, Karamel, White)", 90),
                ("Frozen (Çeşitli Meyveler)", 75), ("Meyveli Soda", 35), ("Soda", 30),
                ("Cool Lime", 70), ("Caramel Frappuccino", 90)
            ],
            "Çerezler": [
                ("Kavrulmuş Antep Fıstığı", 130), ("Atom Çerez", 110), ("Taze Antep Fıstığı", 25)
            ],
            "Tatlılar": [
                ("Fıstık Rüyası", 125), ("Frambuazlı Cheesecake", 125),
                ("Limonlu Cheesecake", 125), ("Mozaik", 125), ("Profiterol", 125),
                ("Tiramisu", 125), ("Latte", 125), ("Devils", 125),
                ("Yer Fıstıklı Pasta", 125), ("Kara Ormanlı Pasta", 125)
            ],
            "Dondurmalar": [
                ("Kaymak", 20), ("Fıstık", 20), ("Çikolata", 20), ("Karamel", 20),
                ("Çilek", 20), ("Limon Sorbe", 20), ("Bal Badem", 20), ("Karadut", 20),
                ("Oreo", 20), ("Blue Sky", 20), ("Vişne", 20), ("Kavun", 20),
                ("Meyve Şöleni", 20), ("Muz", 20)
            ]
        }

        for kategori, urunler in veri.items():
            cursor.execute("INSERT INTO kategoriler (isim) VALUES (?)", (kategori,))
            cursor.execute("SELECT id FROM kategoriler WHERE isim = ?", (kategori,))
            kategori_id = cursor.fetchone()[0]
            for ad, fiyat in urunler:
                cursor.execute(
                    "INSERT INTO menu (ad, fiyat, kategori_id) VALUES (?, ?, ?)",
                    (ad, fiyat, kategori_id)
                )

        conn.commit()
        conn.close()
        print("✅ Menü veritabanı oluşturuldu.")

init_db()
init_menu_db()

# MENU_LISTESI aynı kalabilir ya da kaldırılabilir (isteğe bağlı)
