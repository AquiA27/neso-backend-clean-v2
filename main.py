from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
from dotenv import load_dotenv
import os
import sqlite3
from datetime import datetime
import json
import re
import io
from google.cloud import texttospeech

# Hafıza yönetimi
from memory import get_memory, add_to_memory

# Ortam değişkenlerini yükle
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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

# Veritabanı bağlantısı ve tablo oluşturma
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

# Menü listesi
MENU_LISTESI = [
    "Çay", "Fincan Çay", "Sahlep", "Bitki Çayları", "Türk Kahvesi",
    "Osmanlı Kahvesi", "Menengiç Kahvesi", "Süt", "Nescafe",
    "Nescafe Sütlü", "Esspresso", "Filtre Kahve", "Cappuccino",
    "Mocha", "White Mocha", "Classic Mocha", "Caramel Mocha",
    "Latte", "Sıcak Çikolata", "Macchiato"
]

# Emojileri temizleyen yardımcı fonksiyon
def remove_emojis(text):
    emoji_pattern = re.compile("[" 
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # Transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # Flags
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text).strip()

@app.post("/neso")
async def neso_asistan(req: Request):
    try:
        data = await req.json()
        user_text = data.get("text")
        masa = data.get("masa", "bilinmiyor")

        menu_metni = ", ".join(MENU_LISTESI)

        system_prompt = {
            "role": "system",
            "content": (
                f"Sen Neso adında kibar, sevimli ve espirili bir restoran yapay zeka asistanısın. "
                f"Aşağıdaki ürünler kafenin menüsüdür. Sadece bu ürünler sipariş edilebilir:\n\n"
                f"{menu_metni}\n\n"
                "Kullanıcının mesajı sipariş içeriyorsa, kibar ve doğal konuşma diliyle yanıt ver. Yanıt kısa, gerçekçi ve profesyonel olsun. Dilersen samimi bir emoji ile süsle ama abartma. Format şu olmalı:\n"
                '{\n  "reply": "Siparişi kibar ve gerçekçi bir şekilde onaylayan kısa bir mesaj yaz. '
                'Örneğin: \'Latte siparişiniz alındı, 10 dakika içinde hazır olacak ☕️\' gibi. Emoji eklemeyi unutma.",\n'
                '  "sepet": [ { "urun": "ürün adı", "adet": sayı } ]\n}\n\n'
                "Eğer müşteri sohbet ediyorsa (örneğin 'ne içmeliyim?', 'bugün ne önerirsin?'), "
                "sadece öneri ver, samimi ol, emoji kullan. JSON kullanma.\n\n"
                "Eğer müşteri menüde olmayan bir ürün isterse (örneğin 'menemen' veya 'pizza'), "
                "kibarca menüde olmadığını belirt. Sakın uydurma ürün ekleme veya tahminde bulunma."
            )
        }

        # 🧠 Hafızayı al, sistemi ve kullanıcı mesajını ekle
        history = get_memory(masa)
        full_messages = history + [system_prompt, {"role": "user", "content": user_text}]

        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=full_messages,
            temperature=0.7
        )

        raw = chat_completion.choices[0].message.content
        print("🔍 OpenAI Yanıtı:", raw)

        # 🧠 Hafızayı güncelle
        add_to_memory(masa, "user", user_text)
        add_to_memory(masa, "assistant", raw)

        if raw.strip().startswith("{"):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {
                    "reply": "Siparişinizi tam anlayamadım efendim. Menüdeki ürünlerden tekrar deneyebilir misiniz? 🥲",
                    "sepet": []
                }

            conn = sqlite3.connect("neso.db")
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO siparisler (masa, istek, yanit, sepet, zaman)
                VALUES (?, ?, ?, ?, ?)
            """, (
                masa,
                user_text,
                remove_emojis(parsed.get("reply", "")),
                json.dumps(parsed.get("sepet", []), ensure_ascii=False),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            conn.commit()
            conn.close()

            return {
                "reply": parsed.get("reply", ""),
                "voice_reply": remove_emojis(parsed.get("reply", ""))
            }
        else:
            return {
                "reply": raw,
                "voice_reply": remove_emojis(raw)
            }

    except Exception as e:
        print("💥 HATA:", e)
        return {"reply": f"Hata oluştu: {str(e)}"}

@app.post("/sesli-siparis")
async def eski_neso(req: Request):
    return await neso_asistan(req)

@app.get("/siparisler")
def siparis_listele():
    try:
        conn = sqlite3.connect("neso.db")
        cursor = conn.cursor()
        cursor.execute("SELECT masa, istek, yanit, sepet, zaman FROM siparisler ORDER BY zaman DESC")
        rows = cursor.fetchall()
        conn.close()

        orders = [
            {
                "masa": row[0],
                "istek": row[1],
                "yanit": row[2],
                "sepet": json.loads(row[3]),
                "zaman": row[4]
            } for row in rows
        ]
        return {"orders": orders}
    except Exception as e:
        return {"orders": [], "error": str(e)}

# 🔊 Google Text-to-Speech ile sesli yanıt üret
def google_sesli_yanit(text):
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="tr-TR",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.0,
        pitch=1.2,
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    return response.audio_content  # MP3 olarak döner

# 🎧 MP3 formatında sesli yanıt dönen endpoint
@app.post("/sesli-yanit")
async def sesli_yanit_api(req: Request):
    data = await req.json()
    text = data.get("text", "")
    if not text:
        return {"error": "Metin verisi bulunamadı."}

    try:
        audio = google_sesli_yanit(text)
        return StreamingResponse(io.BytesIO(audio), media_type="audio/mpeg")
    except Exception as e:
        return {"error": f"Ses üretilemedi: {str(e)}"}
