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
from openai import OpenAI
from google.cloud import texttospeech

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
security = HTTPBasic()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
            await websocket.receive_text()
    except WebSocketDisconnect:
        aktif_mutfak_websocketleri.remove(websocket)

async def mutfaga_gonder(siparis):
    for ws in aktif_mutfak_websocketleri:
        try:
            await ws.send_text(json.dumps(siparis))
        except:
            continue

@app.post("/siparis-ekle")
async def siparis_ekle(data: dict = Body(...)):
    print("📅 Yeni sipariş geldi:", data)
    masa = data.get("masa")
    yanit = data.get("yanit")
    sepet_verisi = data.get("sepet", [])
    zaman = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 👉 İstek metni sepetten oluşturulsun
    try:
        istek = ", ".join([
            f"{item.get('urun', '').strip()} ({item.get('adet', 1)} adet)"
            for item in sepet_verisi
        ])
    except Exception as e:
        istek = "Tanımsız"

    try:
        sepet_json = json.dumps(sepet_verisi)
        conn = sqlite3.connect("neso.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO siparisler (masa, istek, yanit, sepet, zaman)
            VALUES (?, ?, ?, ?, ?)
        """, (masa, istek, yanit, sepet_json, zaman))
        conn.commit()
        conn.close()

        if isinstance(sepet_verisi, list) and len(sepet_verisi) > 0:
            await mutfaga_gonder({
                "masa": masa,
                "istek": istek,
                "yanit": yanit,
                "sepet": sepet_verisi,
                "zaman": zaman
            })

        return {"mesaj": "Sipariş kaydedildi. Mutfak bilgilendirmesi yapıldı (eğer gerekiyorsa)."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sipariş eklenemedi: {e}")

def init_db():
    conn = sqlite3.connect("neso.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS siparisler (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            masa TEXT,
            istek TEXT,
            yanit TEXT,
            zaman TEXT
        )
    """)
    cursor.execute("PRAGMA table_info(siparisler)")
    kolonlar = [row[1] for row in cursor.fetchall()]
    if "sepet" not in kolonlar:
        cursor.execute("ALTER TABLE siparisler ADD COLUMN sepet TEXT")
    conn.commit()
    conn.close()

def init_menu_db():
    yeni_olustu = not os.path.exists("neso_menu.db")
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
            print("❌ CSV otomatik yukleme hatasi:", e)
    conn.close()

init_db()
init_menu_db()


# ✨ OpenAI modele menü aktarım fonksiyonu
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
        "Amacın gelen müşterilerin mutlu memnun şekilde ayrılmalarını sağlamak. "
        "Kendine has tarzın ve zekanla insanların verdiği alakasız tepki ve sorulara mümkün olduğunca saygılı, "
        "ve sınırı aşan durumlarda ise idareye bildirmeyi bilen bir yapıdasın. "
        "Yapay zeka modeli olduğun için insanlar seni sınayacak; buna mümkün olan en iyi şekilde, sana yaraşır şekilde karşılık ver.\n\n"
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
            raise ValueError("Metin boş geldi. Sesli yanıt oluşturulamaz.")

        print("🟡 Sesli yanıt istendi. Metin:", metin)

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

        print("✅ Sesli yanıt başarıyla oluşturuldu.")
        return Response(content=response.audio_content, media_type="audio/mpeg")

    except Exception as e:
        print("❌ SESLİ YANIT HATASI:", str(e))
        raise HTTPException(status_code=500, detail=f"Sesli yanıt hatası: {e}")
