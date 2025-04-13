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

init_db()

# Tüm menü ürünleri
MENU_LISTESI = [
    # Sıcak İçecekler
    "Çay", "Fincan Çay", "Sahlep (Tarçınlı Fıstıklı)", "Bitki Çayları (Ihlamur, Nane-Limon, vb.)",
    "Türk Kahvesi", "Osmanlı Kahvesi", "Menengiç Kahvesi", "Süt", "Nescafe", "Nescafe Sütlü",
    "Esspresso", "Filtre Kahve", "Cappuccino", "Mocha (White/Classic/Caramel)", "Latte",
    "Sıcak Çikolata", "Macchiato",

    # Soğuk İçecekler
    "Limonata", "Cola", "Fanta", "Sprite", "Cappy Vişne", "Cappy Şeftali", "Cappy Kayısı",
    "Cappy Karışık", "Cappy Portakal", "Fuse Tea Karpuz", "Fuse Tea Şeftali", "Fuse Tea Limon",
    "Cola Zero", "Churchill", "Taze Sıkılmış Portakal Suyu", "Milkshake (Çilek, Muz, vs.)",
    "Ice Mocha (Classic, Karamel, White)", "Frozen (Çeşitli Meyveler)", "Meyveli Soda",
    "Soda", "Cool Lime", "Caramel Frappuccino",

    # Çerezler
    "Kavrulmuş Antep Fıstığı", "Atom Çerez", "Taze Antep Fıstığı",

    # Tatlılar
    "Fıstık Rüyası", "Frambuazlı Cheesecake", "Limonlu Cheesecake", "Mozaik", "Profiterol",
    "Tiramisu", "Latte", "Devils", "Yer Fıstıklı Pasta", "Kara Ormanlı Pasta",

    # Dondurmalar
    "Kaymak", "Fıstık", "Çikolata", "Karamel", "Çilek", "Limon Sorbe", "Bal Badem",
    "Karadut", "Oreo", "Blue Sky", "Vişne", "Kavun", "Meyve Şöleni", "Muz"
]
