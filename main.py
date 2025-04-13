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

# Veritabanı
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

MENU_LISTESI = [
    "Çay", "Fincan Çay", "Sahlep", "Bitki Çayları", "Türk Kahvesi",
    "Osmanlı Kahvesi", "Menengiç Kahvesi", "Süt", "Nescafe",
    "Nescafe Sütlü", "Esspresso", "Filtre Kahve", "Cappuccino",
    "Mocha", "White Mocha", "Classic Mocha", "Caramel Mocha",
    "Latte", "Sıcak Çikolata", "Macchiato"
]

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002700-\U000027BF"  # dingbats
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA70-\U0001FAFF"  # extended symbols
        "\U00002600-\U000026FF"  # miscellaneous
        "\U0001F700-\U0001F77F"  # alchemical
        "\u200d"                 # Zero-width joiner
        "\ufe0f"                 # Variation Selector-16
        "]+", flags=re.UNICODE
    )
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
                f"Aşağıdaki ürünler kafenin menüsüdür:\n\n"
                f"{menu_metni}\n\n"
                "Yanıtlar kısa, gerçekçi ve profesyonel olsun. Format şu olmalı:\n"
                '{ "reply": "Latte siparişiniz alındı ☕️", "sepet": [{"urun": "Latte", "adet": 1}] }'
            )
        }

        history = get_memory(masa)
        full_messages = history + [system_prompt, {"role": "user", "content": user_text}]
        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=full_messages,
            temperature=0.7
        )

        raw = chat_completion.choices[0].message.content
        print("🔍 Yanıt:", raw)

        add_to_memory(masa, "user", user_text)
        add_to_memory(masa, "assistant", raw)

        if raw.strip().startswith("{"):
            try:
                parsed = json.loads(raw)
            except:
                parsed = {"reply": "Siparişi anlayamadım.", "sepet": []}

            conn = sqlite3.connect("neso.db")
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO siparisler (masa, istek, yanit, sepet, zaman)
                VALUES (?, ?, ?, ?, ?)
            """, (
                masa, user_text, remove_emojis(parsed["reply"]),
                json.dumps(parsed["sepet"], ensure_ascii=False),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            conn.commit()
            conn.close()

            return {
                "reply": parsed["reply"],
                "voice_reply": remove_emojis(parsed["reply"])
            }

        return {"reply": raw, "voice_reply": remove_emojis(raw)}

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
        return {"orders": [
            {"masa": row[0], "istek": row[1], "yanit": row[2], "sepet": json.loads(row[3]), "zaman": row[4]}
            for row in rows
        ]}
    except Exception as e:
        return {"orders": [], "error": str(e)}

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
    return response.audio_content

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
