from fastapi import (
    FastAPI, Request, Body, Query, UploadFile, File, HTTPException,
    status, Depends, WebSocket, WebSocketDisconnect
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel, Field, ValidationError
import os
import base64
import regex # Standart 're' yerine emoji için bunu kullanıyoruz
import tempfile
import sqlite3
import json
import csv
import logging
import logging.config # Daha gelişmiş loglama için
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from google.cloud import texttospeech
from google.api_core import exceptions as google_exceptions
import re # Bazı basit regexler için hala kullanılabilir
import asyncio # Broadcast için

# --------------------------------------------------------------------------
# Loglama Yapılandırması
# --------------------------------------------------------------------------
# Temel yapılandırma yerine daha detaylı bir yapılandırma kullanılabilir.
# Örneğin, dosyaya loglama, farklı seviyeler vb.
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout", # veya sys.stderr
        },
        # İsteğe bağlı olarak dosyaya loglama eklenebilir:
        # "file": {
        #     "class": "logging.FileHandler",
        #     "formatter": "default",
        #     "filename": "neso_backend.log",
        #     "encoding": "utf-8",
        # },
    },
    "loggers": {
        "root": { # Kök logger
            "level": "INFO",
            "handlers": ["console"], # veya ["console", "file"]
        },
        "uvicorn.error": { # Uvicorn hataları için
             "level": "INFO",
             "handlers": ["console"],
             "propagate": False,
        },
         "uvicorn.access": { # Erişim logları (isteğe bağlı)
             "level": "WARNING", # Sadece uyarı ve üstünü göster
             "handlers": ["console"],
             "propagate": False,
         },
         "app_logger": { # Kendi uygulama loglarımız için özel logger
             "level": "INFO",
             "handlers": ["console"],
             "propagate": False, # Kök logger'a tekrar gitmesin
         },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("app_logger") # Özel logger'ımızı kullanalım

# --------------------------------------------------------------------------
# Ortam Değişkenleri ve Başlangıç Kontrolleri
# --------------------------------------------------------------------------
load_dotenv()
logger.info("Ortam değişkenleri yükleniyor...")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CREDS_BASE64 = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_BASE64")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
SECRET_KEY = os.getenv("SECRET_KEY", "cok-gizli-bir-anahtar-olmalı")
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "*") # '*' yerine 'http://localhost:3000,https://neso-guncel.vercel.app' gibi

if not OPENAI_API_KEY:
    logger.critical("KRİTİK: OpenAI API anahtarı (OPENAI_API_KEY) bulunamadı! Yanıtlama özelliği çalışmayacak.")
if not GOOGLE_CREDS_BASE64:
    logger.warning("UYARI: Google Cloud kimlik bilgileri (GOOGLE_APPLICATION_CREDENTIALS_BASE64) bulunamadı. Sesli yanıt özelliği çalışmayabilir.")
if SECRET_KEY == "cok-gizli-bir-anahtar-olmalı":
     logger.warning("UYARI: Güvenli bir SECRET_KEY ortam değişkeni ayarlanmamış!")
if CORS_ALLOWED_ORIGINS == "*":
    logger.warning("UYARI: CORS tüm kaynaklara izin veriyor (*). Üretimde spesifik domainlere izin verin!")

# --------------------------------------------------------------------------
# Yardımcı Fonksiyonlar
# --------------------------------------------------------------------------
def temizle_emoji(text: str | None) -> str:
    """Verilen metinden emojileri temizler (regex kütüphanesi kullanarak)."""
    if not isinstance(text, str):
        return "" # String değilse boş string döndür
    try:
        # \p{Emoji_Presentation} sadece görsel emojileri hedefler, daha güvenli olabilir.
        # \p{Extended_Pictographic} diğer sembolleri de kapsayabilir.
        # İkisini birleştirelim:
        emoji_pattern = regex.compile(r"[\p{Emoji_Presentation}\p{Extended_Pictographic}]+")
        cleaned_text = emoji_pattern.sub(r'', text)
        return cleaned_text
    except regex.error as e:
        logger.error(f"Emoji regex (regex lib) derleme hatası: {e}")
        return text # Hata durumunda orijinal metni döndür
    except Exception as e:
        logger.error(f"Emoji temizleme (regex lib) sırasında beklenmedik hata: {e}")
        return text

# --------------------------------------------------------------------------
# API İstemcileri Başlatma
# --------------------------------------------------------------------------
openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("✅ OpenAI istemcisi başarıyla başlatıldı.")
    except Exception as e:
         logger.error(f"❌ OpenAI istemcisi başlatılamadı: {e}")
else:
    logger.error("❌ OpenAI istemcisi API anahtarı olmadığı için başlatılamadı.")


google_creds_path = None
tts_client = None
if GOOGLE_CREDS_BASE64:
    try:
        decoded_creds = base64.b64decode(GOOGLE_CREDS_BASE64)
        # Güvenli geçici dosya oluşturma (uygulama kapanınca silinir)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w+b') as tmp_file:
            tmp_file.write(decoded_creds)
            google_creds_path = tmp_file.name
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_creds_path
        logger.info("✅ Google Cloud kimlik bilgileri geçici dosyaya yazıldı.")
        try:
            tts_client = texttospeech.TextToSpeechClient()
            logger.info("✅ Google Text-to-Speech istemcisi başarıyla başlatıldı.")
        except Exception as e:
            logger.error(f"❌ Google Text-to-Speech istemcisi başlatılamadı: {e}")
    except base64.binascii.Error as e:
         logger.error(f"❌ Google Cloud kimlik bilgileri base64 formatında değil: {e}")
    except Exception as e:
        logger.error(f"❌ Google Cloud kimlik bilgileri işlenirken hata: {e}")

# --------------------------------------------------------------------------
# FastAPI Uygulaması ve Güvenlik
# --------------------------------------------------------------------------
app = FastAPI(
    title="Neso Sipariş Asistanı API",
    version="1.2.2",
    description="Fıstık Kafe için sesli ve yazılı sipariş alma backend servisi."
)
security = HTTPBasic()

# --------------------------------------------------------------------------
# Middleware Ayarları
# --------------------------------------------------------------------------
# CORS ayarları ortam değişkeninden alınır
allowed_origins_list = [origin.strip() for origin in CORS_ALLOWED_ORIGINS.split(',')] if CORS_ALLOWED_ORIGINS != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"], # Veya spesifik metodlar: ["GET", "POST", "DELETE"]
    allow_headers=["*"], # Veya spesifik başlıklar
)
logger.info(f"CORS Middleware etkinleştirildi. İzin verilen kaynaklar: {allowed_origins_list}")

app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    session_cookie="neso_session" # Cookie adı
)
logger.info("Session Middleware etkinleştirildi.")

# --------------------------------------------------------------------------
# WebSocket Bağlantı Yönetimi
# --------------------------------------------------------------------------
aktif_mutfak_websocketleri: set[WebSocket] = set() # Liste yerine set daha verimli olabilir
aktif_admin_websocketleri: set[WebSocket] = set()

async def broadcast_message(connections: set[WebSocket], message: dict):
    """Belirtilen WebSocket bağlantılarına JSON mesajı gönderir."""
    if not connections:
        return # Gönderilecek bağlantı yoksa çık

    message_json = json.dumps(message)
    # Asenkron görevleri topla
    tasks = [ws.send_text(message_json) for ws in connections]
    # Görevleri concurrently çalıştır ve sonuçları bekle (hataları yakala)
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Hata alan veya kapanan bağlantıları tespit et
    disconnected_sockets = set()
    for ws, result in zip(list(connections), results): # Set'i listeye çevirerek zip yap
        if isinstance(result, Exception):
            client_info = f"{ws.client.host}:{ws.client.port}" if ws.client else "Bilinmeyen"
            logger.warning(f"🔌 WebSocket gönderme hatası ({client_info}): {result}")
            disconnected_sockets.add(ws)
            # Bağlantıyı kapatmayı deneyebiliriz (opsiyonel)
            # try:
            #     await ws.close(code=status.WS_1011_INTERNAL_ERROR)
            # except RuntimeError: # Zaten kapalıysa
            #     pass

    # Kapananları set'ten çıkar
    for ws in disconnected_sockets:
        if ws in connections: # Hala setteyse çıkar (nadiren de olsa race condition olabilir)
            connections.remove(ws)
            client_info = f"{ws.client.host}:{ws.client.port}" if ws.client else "Bilinmeyen"
            logger.info(f"📉 WebSocket bağlantısı (hata sonrası) kaldırıldı: {client_info}")

# --------------------------------------------------------------------------
# WebSocket Endpoint'leri
# --------------------------------------------------------------------------
async def websocket_lifecycle(websocket: WebSocket, connections: set[WebSocket], endpoint_name: str):
    """WebSocket bağlantı yaşam döngüsünü yöneten genel fonksiyon."""
    await websocket.accept()
    connections.add(websocket)
    client_host = websocket.client.host if websocket.client else "Bilinmeyen"
    logger.info(f"🔗 {endpoint_name} WS bağlandı: {client_host} (Toplam: {len(connections)})")
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                # Endpoint'e özel mesaj işleme burada yapılabilir (şimdilik sadece ping)
                # elif endpoint_name == "Admin" and message.get("type") == "some_admin_action":
                #     pass
            except json.JSONDecodeError:
                logger.warning(f"⚠️ {endpoint_name} WS ({client_host}): Geçersiz JSON: {data}")
            except Exception as e:
                 logger.error(f"❌ {endpoint_name} WS ({client_host}) Mesaj işleme hatası: {e}")
                 # break # Hata durumunda döngüden çıkıp bağlantıyı kapatabiliriz
    except WebSocketDisconnect as e:
        # Beklenen veya beklenmeyen kapanma durumları
        if e.code == status.WS_1000_NORMAL_CLOSURE or e.code == status.WS_1001_GOING_AWAY:
             logger.info(f"🔌 {endpoint_name} WS normal kapatıldı: {client_host} (Kod: {e.code})")
        else:
             logger.warning(f"🔌 {endpoint_name} WS beklenmedik şekilde kapandı: {client_host} (Kod: {e.code})")
    except Exception as e:
        logger.error(f"❌ {endpoint_name} WS hatası ({client_host}): {e}")
    finally:
        # Bağlantı set'ten kaldırılır
        if websocket in connections:
            connections.remove(websocket)
        logger.info(f"📉 {endpoint_name} WS kaldırıldı: {client_host} (Kalan: {len(connections)})")

@app.websocket("/ws/admin")
async def websocket_admin_endpoint(websocket: WebSocket):
    await websocket_lifecycle(websocket, aktif_admin_websocketleri, "Admin")

@app.websocket("/ws/mutfak")
async def websocket_mutfak_endpoint(websocket: WebSocket):
    await websocket_lifecycle(websocket, aktif_mutfak_websocketleri, "Mutfak/Masa")

# --------------------------------------------------------------------------
# Veritabanı İşlemleri ve Yardımcıları
# --------------------------------------------------------------------------
DB_NAME = "neso.db"
MENU_DB_NAME = "neso_menu.db"
DB_DATA_DIR = os.getenv("DB_DATA_DIR", ".") # Veritabanı dosyalarının konumu (Render için önemli olabilir)
DB_PATH = os.path.join(DB_DATA_DIR, DB_NAME)
MENU_DB_PATH = os.path.join(DB_DATA_DIR, MENU_DB_NAME)

# Veritabanı dizininin var olduğundan emin ol
os.makedirs(DB_DATA_DIR, exist_ok=True)

def get_db_connection(db_path: str) -> sqlite3.Connection:
    """Veritabanı bağlantısı oluşturur."""
    try:
        conn = sqlite3.connect(db_path, timeout=10) # Timeout eklendi
        conn.row_factory = sqlite3.Row # Sütun adlarıyla erişim için
        return conn
    except sqlite3.Error as e:
        logger.critical(f"❌ KRİTİK: Veritabanı bağlantısı kurulamadı ({db_path}): {e}")
        raise HTTPException(status_code=503, detail=f"Veritabanı bağlantı hatası: {e}")

async def update_table_status(masa_id: str, islem: str = "Erişim"):
    """Veritabanındaki masa durumunu günceller ve admin paneline bildirir."""
    now = datetime.now()
    try:
        with get_db_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO masa_durumlar (masa_id, son_erisim, aktif, son_islem)
                VALUES (?, ?, TRUE, ?)
                ON CONFLICT(masa_id) DO UPDATE SET
                    son_erisim = excluded.son_erisim,
                    aktif = excluded.aktif,
                    son_islem = CASE WHEN excluded.son_islem IS NOT NULL THEN excluded.son_islem ELSE son_islem END
            """, (masa_id, now.strftime("%Y-%m-%d %H:%M:%S"), islem))
            conn.commit()
            # logger.info(f"⏱️ Masa durumu güncellendi: Masa {masa_id}, İşlem: {islem}") # Çok sık log

        if aktif_admin_websocketleri:
             await broadcast_message(aktif_admin_websocketleri, {
                 "type": "masa_durum",
                 "data": {"masaId": masa_id, "sonErisim": now.isoformat(), "aktif": True, "sonIslem": islem}
             })
             # logger.info(f"📢 Masa durumu admin paneline bildirildi: Masa {masa_id}") # Çok sık log

    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (masa durumu güncellenemedi): {e}")
    except Exception as e:
        logger.error(f"❌ Masa durumu güncelleme hatası: {e}")

# --------------------------------------------------------------------------
# Middleware
# --------------------------------------------------------------------------
@app.middleware("http")
async def track_active_users(request: Request, call_next):
    """Gelen isteklerde masa ID'si varsa durumu günceller."""
    # Path parametresinden masaId'yi al (örn: /masa/{masaId})
    masa_id = request.path_params.get("masaId")
    # Alternatif: Query parametresinden al (örn: /endpoint?masa_id=1)
    # if not masa_id: masa_id = request.query_params.get("masa_id")
    # Alternatif: Request body'den al (POST/PUT istekleri için)
    # if not masa_id and request.method in ["POST", "PUT"]:
    #     try:
    #         body = await request.json()
    #         masa_id = body.get("masa")
    #     except: pass # Body JSON değilse veya 'masa' yoksa

    if masa_id:
        # İşlem tipini daha anlamlı hale getirebiliriz
        endpoint_name = request.scope.get("endpoint").__name__ if request.scope.get("endpoint") else request.url.path
        islem = f"{request.method} {endpoint_name}"
        await update_table_status(str(masa_id), islem) # ID'yi string yapalım

    try:
        response = await call_next(request)
        return response
    except Exception as e:
        # Middleware seviyesinde genel hata yakalama
        logger.exception(f"💥 Beklenmedik Middleware Hatası: {e}") # Tam traceback loglanır
        return JSONResponse(
            status_code=500,
            content={"detail": "Sunucuda beklenmedik bir hata oluştu."}
        )

# --------------------------------------------------------------------------
# Aktif Masalar Endpoint
# --------------------------------------------------------------------------
@app.get("/aktif-masalar")
async def get_active_tables_endpoint():
    """Son 5 dakika içinde aktif olan masaları döndürür."""
    try:
        active_time_limit = datetime.now() - timedelta(minutes=5)
        with get_db_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT masa_id, son_erisim, aktif, son_islem FROM masa_durumlar
                WHERE son_erisim >= ? AND aktif = TRUE ORDER BY son_erisim DESC
            """, (active_time_limit.strftime("%Y-%m-%d %H:%M:%S"),))
            results = cursor.fetchall()
            active_tables_data = [dict(row) for row in results]
        logger.info(f"📊 Aktif masalar sorgulandı, {len(active_tables_data)} adet bulundu.")
        return {"tables": active_tables_data}
    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (aktif masalar alınamadı): {e}")
        raise HTTPException(status_code=503, detail="Veritabanına erişilemiyor.")
    except Exception as e:
        logger.error(f"❌ Aktif masalar alınırken hata: {e}")
        raise HTTPException(status_code=500, detail=f"Aktif masalar alınırken bir hata oluştu.")

# --------------------------------------------------------------------------
# Admin Kimlik Doğrulama
# --------------------------------------------------------------------------
def check_admin(credentials: HTTPBasicCredentials = Depends(security)):
    """Admin kimlik bilgilerini ortam değişkenleriyle doğrular."""
    # Ortam değişkenlerinden kullanıcı adı ve şifreyi al
    correct_username = ADMIN_USERNAME
    correct_password = ADMIN_PASSWORD

    # Zamanlama saldırılarına karşı küçük bir önlem (her zaman aynı sürede kontrol)
    # import secrets
    # is_user_ok = secrets.compare_digest(credentials.username.encode('utf-8'), correct_username.encode('utf-8'))
    # is_pass_ok = secrets.compare_digest(credentials.password.encode('utf-8'), correct_password.encode('utf-8'))
    # Basit karşılaştırma (şimdilik yeterli)
    is_user_ok = credentials.username == correct_username
    is_pass_ok = credentials.password == correct_password

    if not (is_user_ok and is_pass_ok):
        logger.warning(f"🔒 Başarısız admin girişi denemesi: Kullanıcı adı '{credentials.username}'")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Geçersiz kimlik bilgileri",
            headers={"WWW-Authenticate": "Basic"},
        )
    # logger.debug(f"🔑 Admin girişi başarılı: {credentials.username}") # Başarılı girişleri loglamaktan kaçın
    return True # Başarılı ise True döner

# --------------------------------------------------------------------------
# Pydantic Modelleri (Veri Doğrulama için)
# --------------------------------------------------------------------------
class SepetItem(BaseModel):
    urun: str = Field(..., min_length=1)
    adet: int = Field(..., gt=0)
    fiyat: float | None = None # Fiyat opsiyonel olabilir, backend'den alınacak
    kategori: str | None = None # Kategori opsiyonel olabilir

class SiparisEkleData(BaseModel):
    masa: str = Field(..., min_length=1)
    sepet: list[SepetItem] = Field(..., min_items=1) # Sepet boş olamaz
    istek: str | None = None
    yanit: str | None = None

class SiparisGuncelleData(BaseModel):
    masa: str = Field(..., min_length=1)
    durum: str # Geçerli durumlar endpoint içinde kontrol edilecek
    id: int | None = None # Opsiyonel sipariş ID

class MenuEkleData(BaseModel):
    ad: str = Field(..., min_length=1)
    fiyat: float = Field(..., gt=0) # Fiyat 0'dan büyük olmalı
    kategori: str = Field(..., min_length=1)

class AdminCredentialsUpdate(BaseModel):
    yeniKullaniciAdi: str = Field(..., min_length=1)
    yeniSifre: str = Field(..., min_length=4) # Minimum şifre uzunluğu eklendi

class SesliYanitData(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = "tr-TR"

# --------------------------------------------------------------------------
# Sipariş Yönetimi Endpoint'leri
# --------------------------------------------------------------------------
@app.post("/siparis-ekle", status_code=status.HTTP_201_CREATED)
async def add_order_endpoint(data: SiparisEkleData):
    masa = data.masa
    yanit = data.yanit
    sepet_verisi = data.sepet # Pydantic modeli sayesinde liste ve item'lar doğrulanmış oldu
    istek_orijinal = data.istek
    zaman_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"📥 Yeni sipariş isteği alındı: Masa {masa}, Sepet: {len(sepet_verisi)} ürün")

    # Fiyatları backend'den alıp sepete ekleyelim (güvenlik ve tutarlılık için)
    price_dict = get_menu_price_dict()
    processed_sepet = []
    for item in sepet_verisi:
        item_dict = item.model_dump() # Pydantic modelini dict'e çevir
        urun_adi_lower = item_dict['urun'].lower().strip()
        item_dict['fiyat'] = price_dict.get(urun_adi_lower, 0.0) # Güncel fiyatı ekle
        # Kategori bilgisi varsa koru, yoksa belki DB'den bulunur? (Şimdilik opsiyonel)
        processed_sepet.append(item_dict)

    try:
        istek_ozet = ", ".join([f"{item.get('adet', 1)}x {item.get('urun', '').strip()}" for item in processed_sepet])
    except Exception as e:
        logger.error(f"❌ Sipariş özeti oluşturma hatası (Masa {masa}): {e}")
        istek_ozet = "Detay alınamadı"

    try:
        sepet_json = json.dumps(processed_sepet) # İşlenmiş sepeti JSON'a çevir
        siparis_id = None
        with get_db_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO siparisler (masa, istek, yanit, sepet, zaman, durum)
                VALUES (?, ?, ?, ?, ?, 'bekliyor')
            """, (masa, istek_orijinal or istek_ozet, yanit, sepet_json, zaman_str))
            siparis_id = cursor.lastrowid
            conn.commit()
        logger.info(f"💾 Sipariş veritabanına kaydedildi: Masa {masa}, Sipariş ID: {siparis_id}")

        siparis_bilgisi = {
            "type": "siparis",
            "data": {"id": siparis_id, "masa": masa, "istek": istek_orijinal or istek_ozet, "sepet": processed_sepet, "zaman": zaman_str, "durum": "bekliyor"}
        }
        await broadcast_message(aktif_mutfak_websocketleri, siparis_bilgisi)
        await broadcast_message(aktif_admin_websocketleri, siparis_bilgisi)
        logger.info(f"📢 Yeni sipariş bildirimi gönderildi: Mutfak ({len(aktif_mutfak_websocketleri)}), Admin ({len(aktif_admin_websocketleri)})")
        await update_table_status(masa, f"Sipariş verdi ({len(processed_sepet)} ürün)")
        return {"mesaj": "Sipariş başarıyla kaydedildi ve ilgili birimlere iletildi.", "siparisId": siparis_id}

    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (sipariş eklenemedi - Masa {masa}): {e}")
        raise HTTPException(status_code=503, detail=f"Sipariş veritabanına kaydedilirken hata oluştu.")
    except Exception as e:
        logger.error(f"❌ Sipariş ekleme sırasında genel hata (Masa {masa}): {e}")
        raise HTTPException(status_code=500, detail=f"Sipariş eklenirken beklenmedik bir hata oluştu.")

@app.post("/siparis-guncelle")
async def update_order_status_endpoint(data: SiparisGuncelleData, auth: bool = Depends(check_admin)):
    masa = data.masa
    durum = data.durum
    siparis_id = data.id
    logger.info(f"🔄 Sipariş durumu güncelleme isteği: Masa {masa}, Yeni Durum: {durum}, ID: {siparis_id}")

    valid_statuses = ["hazirlaniyor", "hazir", "iptal", "bekliyor"]
    if durum not in valid_statuses:
         logger.error(f"❌ Sipariş güncelleme hatası (Masa {masa}): Geçersiz durum '{durum}'.")
         raise HTTPException(status_code=400, detail=f"Geçersiz durum: {durum}. Geçerli durumlar: {valid_statuses}")

    rows_affected = 0
    try:
        with get_db_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            if siparis_id:
                 # Belirli bir siparişi ID ile güncelle
                 cursor.execute("UPDATE siparisler SET durum = ? WHERE id = ?", (durum, siparis_id))
            else:
                 # ID belirtilmemişse, masanın son aktif siparişini güncelle
                 cursor.execute("""
                     UPDATE siparisler SET durum = ? WHERE id = (
                         SELECT id FROM siparisler WHERE masa = ? AND durum NOT IN ('hazir', 'iptal')
                         ORDER BY id DESC LIMIT 1)
                 """, (durum, masa))
            rows_affected = cursor.rowcount
            conn.commit()

        if rows_affected > 0:
             logger.info(f"💾 Sipariş durumu veritabanında güncellendi: Masa {masa}, Durum: {durum}, Etkilenen: {rows_affected}")
             notification = {
                 "type": "durum",
                 "data": {"id": siparis_id, "masa": masa, "durum": durum, "zaman": datetime.now().isoformat()}
             }
             await broadcast_message(aktif_mutfak_websocketleri, notification)
             await broadcast_message(aktif_admin_websocketleri, notification)
             logger.info(f"📢 Sipariş durum güncellemesi bildirildi: Masa {masa}, Durum: {durum}")
             await update_table_status(masa, f"Sipariş durumu -> {durum}")
             return {"success": True, "message": f"Sipariş durumu '{durum}' olarak güncellendi."}
        else:
             logger.warning(f"⚠️ Sipariş durumu güncellenemedi (Masa {masa}, Durum: {durum}): Uygun sipariş bulunamadı veya zaten güncel.")
             raise HTTPException(status_code=404, detail="Güncellenecek uygun sipariş bulunamadı veya durum zaten aynı.")

    except sqlite3.Error as e:
         logger.error(f"❌ Veritabanı hatası (sipariş durumu güncellenemedi - Masa {masa}): {e}")
         raise HTTPException(status_code=503, detail=f"Sipariş durumu güncellenirken veritabanı hatası oluştu.")
    except Exception as e:
        logger.error(f"❌ Sipariş durumu güncelleme sırasında genel hata: {e}")
        raise HTTPException(status_code=500, detail=f"Sipariş durumu güncellenirken beklenmedik bir hata oluştu.")


@app.get("/siparisler")
def get_orders_endpoint(auth: bool = Depends(check_admin)):
    try:
        with get_db_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, masa, istek, yanit, sepet, zaman, durum FROM siparisler ORDER BY id DESC")
            rows = cursor.fetchall()
            # Sepet JSON string'ini parse etmeye çalışalım (opsiyonel, frontend de yapabilir)
            orders_data = []
            for row in rows:
                order_dict = dict(row)
                try:
                    order_dict['sepet'] = json.loads(order_dict['sepet'] or '[]')
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ Sipariş listesi: Geçersiz sepet JSON (ID: {order_dict['id']})")
                    order_dict['sepet'] = [] # Hata durumunda boş liste ata
                orders_data.append(order_dict)

        logger.info(f" Görüntülenen sipariş sayısı: {len(orders_data)}")
        return {"orders": orders_data}
    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (siparişler alınamadı): {e}")
        raise HTTPException(status_code=503, detail="Veritabanına erişilemiyor.")
    except Exception as e:
        logger.error(f"❌ Siparişler alınırken hata: {e}")
        raise HTTPException(status_code=500, detail=f"Siparişler alınırken bir hata oluştu.")

# --------------------------------------------------------------------------
# Veritabanı Başlatma
# --------------------------------------------------------------------------
def init_db(db_path: str):
    """Ana veritabanı tablolarını oluşturur veya doğrular."""
    logger.info(f"Ana veritabanı kontrol ediliyor: {db_path}")
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS siparisler (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, masa TEXT NOT NULL, istek TEXT,
                    yanit TEXT, sepet TEXT, zaman TEXT NOT NULL,
                    durum TEXT DEFAULT 'bekliyor' CHECK(durum IN ('bekliyor', 'hazirlaniyor', 'hazir', 'iptal'))
                )""")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS masa_durumlar (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, masa_id TEXT UNIQUE NOT NULL,
                    son_erisim TIMESTAMP NOT NULL, aktif BOOLEAN DEFAULT TRUE, son_islem TEXT
                )""")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_siparisler_masa_zaman ON siparisler(masa, zaman DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_masa_durumlar_erisim ON masa_durumlar(son_erisim DESC)")
            conn.commit()
            logger.info(f"✅ Ana veritabanı ({db_path}) başarıyla doğrulandı/oluşturuldu.")
    except Exception as e:
        logger.critical(f"❌ KRİTİK HATA: Ana veritabanı ({db_path}) başlatılamadı! Hata: {e}")
        raise # Uygulamanın başlamasını engelle

def init_menu_db(db_path: str):
    """Menü veritabanı tablolarını oluşturur veya doğrular."""
    logger.info(f"Menü veritabanı kontrol ediliyor: {db_path}")
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS kategoriler (id INTEGER PRIMARY KEY AUTOINCREMENT, isim TEXT UNIQUE NOT NULL COLLATE NOCASE)")
            # kategori_id için NOT NULL eklendi ve stok_durumu'ndan sonra virgül kontrol edildi
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS menu (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    ad TEXT NOT NULL COLLATE NOCASE,
                    fiyat REAL NOT NULL CHECK(fiyat >= 0), 
                    kategori_id INTEGER NOT NULL,  -- Burası düzeltildi
                    stok_durumu INTEGER DEFAULT 1, /* 1: Var, 0: Yok */
                    FOREIGN KEY (kategori_id) REFERENCES kategoriler(id) ON DELETE CASCADE, 
                    UNIQUE(ad, kategori_id)
                )""")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_menu_kategori ON menu(kategori_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_menu_ad ON menu(ad)")
            conn.commit()
            logger.info(f"✅ Menü veritabanı ({db_path}) başarıyla doğrulandı/oluşturuldu.")
    except Exception as e:
        logger.critical(f"❌ KRİTİK HATA: Menü veritabanı ({db_path}) başlatılamadı! Hata: {e}")
        raise

# Uygulama başlangıcında veritabanlarını başlat
try:
    init_db(DB_PATH)
    init_menu_db(MENU_DB_PATH)
except Exception as e:
     # Başlatma sırasında kritik hata olursa uygulamayı durdur
     logger.critical(f"💥 Uygulama başlatılamadı: Veritabanı hatası. Hata: {e}")
     # Uygulamayı güvenli bir şekilde sonlandırmak için ek kod gerekebilir
     # sys.exit(1) # Eğer import sys yapıldıysa
     raise SystemExit(f"Uygulama başlatılamadı: Veritabanı hatası - {e}")


# --------------------------------------------------------------------------
# Menü Yönetimi Yardımcıları ve Endpoint'leri
# --------------------------------------------------------------------------
def get_menu_for_prompt():
    """AI prompt'u için stoktaki menü öğelerini formatlar."""
    try:
        with get_db_connection(MENU_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""SELECT k.isim, m.ad FROM menu m JOIN kategoriler k ON m.kategori_id = k.id WHERE m.stok_durumu = 1 ORDER BY k.isim, m.ad""")
            urunler = cursor.fetchall()
        if not urunler: return "Üzgünüm, menü bilgisi şu anda mevcut değil."
        kategorili_menu = {}
        for kategori, urun in urunler: kategorili_menu.setdefault(kategori, []).append(urun)
        menu_aciklama = "\n".join([f"- {k}: {', '.join(u)}" for k, u in kategorili_menu.items()])
        return "Mevcut menümüz şöyledir:\n" + menu_aciklama
    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (menü prompt için alınamadı): {e}")
        return "Menü bilgisi alınırken bir sorun oluştu."
    except Exception as e:
        logger.error(f"❌ Menü prompt'u oluşturulurken hata: {e}")
        return "Menü bilgisi şu anda yüklenemedi."

def get_menu_price_dict():
    """Ürün adı (küçük harf) -> fiyat eşleşmesini içeren sözlük döndürür."""
    fiyatlar = {}
    try:
        with get_db_connection(MENU_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT LOWER(TRIM(ad)), fiyat FROM menu")
            fiyatlar = {ad: fiyat for ad, fiyat in cursor.fetchall()}
    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (fiyat sözlüğü alınamadı): {e}")
    except Exception as e:
        logger.error(f"❌ Fiyat sözlüğü oluşturulurken hata: {e}")
    return fiyatlar

# Sistem mesajını global olarak tanımla ve başlangıçta oluştur
SISTEM_MESAJI_ICERIK = (
    "Sen, Gaziantep'teki Fıstık Kafe için özel olarak tasarlanmış, Neso adında bir sipariş asistanısın. "
    "Görevin, masadaki müşterilerin sesli veya yazılı taleplerini anlayıp menüdeki ürünlerle eşleştirerek siparişlerini almak ve bu siparişleri mutfağa doğru bir şekilde iletmektir. "
    "Siparişleri sen hazırlamıyorsun, sadece alıyorsun. "
    "Her zaman nazik, yardımsever, samimi ve çözüm odaklı olmalısın. Gaziantep ağzıyla veya şivesiyle konuşmamalısın, standart ve kibar bir Türkçe kullanmalısın. "
    "Müşterinin ne istediğini tam anlayamazsan, soruyu tekrar sormaktan veya seçenekleri netleştirmesini istemekten çekinme. "
    "Sipariş tamamlandığında veya müşteri teşekkür ettiğinde 'Afiyet olsun!' demeyi unutma.\n\n"
    f"{get_menu_for_prompt()}" # Başlangıçta menüyü ekle
)
SYSTEM_PROMPT = {"role": "system", "content": SISTEM_MESAJI_ICERIK}

# Menü değiştiğinde prompt'u güncellemek için fonksiyon (opsiyonel)
def update_system_prompt():
    global SISTEM_MESAJI_ICERIK, SYSTEM_PROMPT
    logger.info("🔄 Sistem mesajı (menü bilgisi) güncelleniyor...")
    SISTEM_MESAJI_ICERIK = SISTEM_MESAJI_ICERIK.split("\n\nMevcut menümüz şöyledir:\n")[0] + "\n\nMevcut menümüz şöyledir:\n" + get_menu_for_prompt()
    SYSTEM_PROMPT = {"role": "system", "content": SISTEM_MESAJI_ICERIK}
    logger.info("✅ Sistem mesajı güncellendi.")


@app.get("/menu")
def get_full_menu_endpoint():
    """Tüm menüyü kategorilere göre gruplanmış olarak döndürür."""
    try:
        full_menu_data = []
        with get_db_connection(MENU_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, isim FROM kategoriler ORDER BY isim")
            kategoriler = cursor.fetchall()
            for kat_row in kategoriler:
                cursor.execute("SELECT ad, fiyat, stok_durumu FROM menu WHERE kategori_id = ? ORDER BY ad", (kat_row['id'],))
                urunler_rows = cursor.fetchall()
                full_menu_data.append({"kategori": kat_row['isim'], "urunler": [dict(urun) for urun in urunler_rows]})
        return {"menu": full_menu_data}
    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (menü alınamadı): {e}")
        raise HTTPException(status_code=503, detail="Veritabanına erişilemiyor.")
    except Exception as e:
        logger.error(f"❌ Menü alınırken hata: {e}")
        raise HTTPException(status_code=500, detail=f"Menü bilgileri alınırken bir hata oluştu.")

@app.post("/menu/ekle", status_code=status.HTTP_201_CREATED)
async def add_menu_item_endpoint(item_data: MenuEkleData, auth: bool = Depends(check_admin)):
    item_name = item_data.ad.strip()
    item_price = item_data.fiyat
    item_category = item_data.kategori.strip()
    logger.info(f"➕ Menüye ekleme isteği: Ad: {item_name}, Fiyat: {item_price}, Kategori: {item_category}")
    try:
        with get_db_connection(MENU_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR IGNORE INTO kategoriler (isim) VALUES (?)", (item_category,))
            cursor.execute("SELECT id FROM kategoriler WHERE isim = ?", (item_category,))
            category_result = cursor.fetchone()
            if not category_result: raise HTTPException(status_code=500, detail="Kategori işlenirken hata oluştu.")
            category_id = category_result[0]
            cursor.execute("INSERT INTO menu (ad, fiyat, kategori_id, stok_durumu) VALUES (?, ?, ?, 1)", (item_name, item_price, category_id))
            conn.commit()
            item_id = cursor.lastrowid
        logger.info(f"💾 Menü öğesi başarıyla eklendi: ID {item_id}, Ad: {item_name}")
        update_system_prompt() # Sistem mesajını güncelle
        return {"mesaj": f"'{item_name}' menüye başarıyla eklendi.", "itemId": item_id}
    except sqlite3.IntegrityError:
         logger.warning(f"⚠️ Menü ekleme hatası: '{item_name}' zaten '{item_category}' kategorisinde mevcut olabilir.")
         raise HTTPException(status_code=409, detail=f"'{item_name}' ürünü '{item_category}' kategorisinde zaten mevcut.")
    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (menü öğesi eklenemedi): {e}")
        raise HTTPException(status_code=503, detail="Veritabanı hatası oluştu.")
    except Exception as e:
        logger.error(f"❌ Menü öğesi eklenirken genel hata: {e}")
        raise HTTPException(status_code=500, detail="Menü öğesi eklenirken beklenmedik bir hata oluştu.")

@app.delete("/menu/sil")
async def delete_menu_item_endpoint(urun_adi: str = Query(..., min_length=1), auth: bool = Depends(check_admin)):
    item_name_to_delete = urun_adi.strip()
    logger.info(f"➖ Menüden silme isteği: Ad: {item_name_to_delete}")
    try:
        with get_db_connection(MENU_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM menu WHERE ad = ?", (item_name_to_delete,))
            rows_affected = cursor.rowcount
            conn.commit()
        if rows_affected > 0:
            logger.info(f"💾 Menü öğesi silindi: Ad: {item_name_to_delete}, Etkilenen: {rows_affected}")
            update_system_prompt() # Sistem mesajını güncelle
            return {"mesaj": f"'{item_name_to_delete}' isimli ürün(ler) menüden başarıyla silindi."}
        else:
            logger.warning(f"⚠️ Menü silme: '{item_name_to_delete}' adında ürün bulunamadı.")
            raise HTTPException(status_code=404, detail=f"'{item_name_to_delete}' adında ürün menüde bulunamadı.")
    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (menü öğesi silinemedi): {e}")
        raise HTTPException(status_code=503, detail="Veritabanı hatası oluştu.")
    except Exception as e:
        logger.error(f"❌ Menü öğesi silinirken genel hata: {e}")
        raise HTTPException(status_code=500, detail="Menü öğesi silinirken beklenmedik bir hata oluştu.")

# --------------------------------------------------------------------------
# AI Yanıt Üretme Endpoint'i
# --------------------------------------------------------------------------
@app.post("/yanitla")
async def handle_message_endpoint(data: dict = Body(...)): # Daha spesifik Pydantic modeli kullanılabilir
    user_message = data.get("text", "")
    table_id = data.get("masa", "bilinmiyor")
    if not user_message: raise HTTPException(status_code=400, detail="Mesaj içeriği boş olamaz.")
    logger.info(f"💬 Mesaj alındı: Masa {table_id}, Mesaj: '{user_message[:50]}...'")
    try:
        if not openai_client: raise HTTPException(status_code=503, detail="Yapay zeka hizmeti şu anda kullanılamıyor.")
        # Güncel sistem mesajını kullan
        messages = [SYSTEM_PROMPT, {"role": "user", "content": user_message}]
        response = openai_client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, temperature=0.6, max_tokens=150)
        ai_reply = response.choices[0].message.content.strip()
        logger.info(f"🤖 AI yanıtı üretildi: Masa {table_id}, Yanıt: '{ai_reply[:50]}...'")
        return {"reply": ai_reply}
    except OpenAIError as e:
        logger.error(f"❌ OpenAI API hatası (Masa {table_id}): {e}")
        raise HTTPException(status_code=503, detail=f"Yapay zeka servisinden yanıt alınamadı: {e}")
    except Exception as e:
        logger.error(f"❌ AI yanıtı üretme hatası (Masa {table_id}): {e}")
        raise HTTPException(status_code=500, detail=f"Yapay zeka yanıtı alınırken bir sorun oluştu.")

# --------------------------------------------------------------------------
# İstatistik Endpoint'leri
# --------------------------------------------------------------------------
@app.get("/istatistik/en-cok-satilan")
def get_popular_items_endpoint():
    try:
        item_counts = {}
        with get_db_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT sepet FROM siparisler WHERE durum != 'iptal'")
            all_carts_json = cursor.fetchall()
        for (sepet_json_str,) in all_carts_json:
            if not sepet_json_str: continue
            try:
                items_in_cart = json.loads(sepet_json_str)
                if not isinstance(items_in_cart, list): continue
                for item in items_in_cart:
                     if not isinstance(item, dict): continue
                     item_name = item.get("urun")
                     quantity = item.get("adet", 1)
                     if item_name and isinstance(quantity, (int, float)) and quantity > 0:
                         item_counts[item_name] = item_counts.get(item_name, 0) + quantity
            except Exception as e: logger.warning(f"⚠️ Popüler ürünler: Sepet işleme hatası ({e}): {sepet_json_str[:50]}...")
        sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        popular_items_data = [{"urun": item, "adet": count} for item, count in sorted_items]
        return popular_items_data
    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (popüler ürünler): {e}")
        raise HTTPException(status_code=503, detail="Veritabanına erişilemiyor.")
    except Exception as e:
        logger.error(f"❌ Popüler ürünler hesaplanırken genel hata: {e}")
        raise HTTPException(status_code=500, detail=f"Popüler ürünler hesaplanırken bir hata oluştu.")

@app.get("/istatistik/gunluk")
def get_daily_stats_endpoint():
    today_str = datetime.now().strftime("%Y-%m-%d")
    try:
        with get_db_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT sepet FROM siparisler WHERE zaman LIKE ? AND durum != 'iptal'", (f"{today_str}%",))
            daily_data = cursor.fetchall()
        total_items, total_revenue = calculate_statistics(daily_data)
        return {"tarih": today_str, "siparis_sayisi": total_items, "gelir": total_revenue}
    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (günlük istatistik): {e}")
        raise HTTPException(status_code=503, detail="Veritabanına erişilemiyor.")
    except Exception as e:
        logger.error(f"❌ Günlük istatistik hesaplanırken genel hata: {e}")
        raise HTTPException(status_code=500, detail=f"Günlük istatistikler hesaplanırken bir hata oluştu.")

@app.get("/istatistik/aylik")
def get_monthly_stats_endpoint():
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    try:
        with get_db_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT sepet FROM siparisler WHERE zaman >= ? AND durum != 'iptal'", (start_date,))
            monthly_data = cursor.fetchall()
        total_items, total_revenue = calculate_statistics(monthly_data)
        return {"baslangic": start_date, "siparis_sayisi": total_items, "gelir": total_revenue}
    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (aylık istatistik): {e}")
        raise HTTPException(status_code=503, detail="Veritabanına erişilemiyor.")
    except Exception as e:
        logger.error(f"❌ Aylık istatistik hesaplanırken genel hata: {e}")
        raise HTTPException(status_code=500, detail=f"Aylık istatistikler hesaplanırken bir hata oluştu.")

@app.get("/istatistik/yillik")
def get_yearly_stats_endpoint():
    try:
        monthly_item_counts = {}
        with get_db_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT zaman, sepet FROM siparisler WHERE durum != 'iptal'")
            all_data = cursor.fetchall()
        for time_str, cart_json_str in all_data:
            if not cart_json_str or not time_str: continue
            try:
                month_key = time_str[:7]
                items_in_cart = json.loads(cart_json_str)
                if not isinstance(items_in_cart, list): continue
                month_total = sum(item.get("adet", 1) for item in items_in_cart if isinstance(item, dict) and isinstance(item.get("adet", 1), (int, float)) and item.get("adet", 1) > 0)
                monthly_item_counts[month_key] = monthly_item_counts.get(month_key, 0) + month_total
            except Exception as e: logger.warning(f"⚠️ Yıllık ist.: Sepet işleme hatası ({e}): {cart_json_str[:50]}...")
        sorted_monthly_data = dict(sorted(monthly_item_counts.items()))
        return sorted_monthly_data
    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (yıllık istatistik): {e}")
        raise HTTPException(status_code=503, detail="Veritabanına erişilemiyor.")
    except Exception as e:
        logger.error(f"❌ Yıllık istatistik hesaplanırken genel hata: {e}")
        raise HTTPException(status_code=500, detail=f"Yıllık istatistikler hesaplanırken bir hata oluştu.")

@app.get("/istatistik/filtreli")
def get_filtered_stats_endpoint(baslangic: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"), bitis: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$")):
    try:
        end_date_inclusive = (datetime.strptime(bitis, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        with get_db_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT sepet FROM siparisler WHERE zaman >= ? AND zaman < ? AND durum != 'iptal'", (baslangic, end_date_inclusive))
            filtered_data = cursor.fetchall()
        total_items, total_revenue = calculate_statistics(filtered_data)
        return {"aralik": f"{baslangic} → {bitis}", "siparis_sayisi": total_items, "gelir": total_revenue}
    except ValueError:
        logger.error(f"❌ Filtreli istatistik: Geçersiz tarih değeri.")
        raise HTTPException(status_code=400, detail="Geçersiz tarih değeri.")
    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (filtreli istatistik): {e}")
        raise HTTPException(status_code=503, detail="Veritabanına erişilemiyor.")
    except Exception as e:
        logger.error(f"❌ Filtreli istatistik hesaplanırken genel hata: {e}")
        raise HTTPException(status_code=500, detail=f"Filtreli istatistikler hesaplanırken bir hata oluştu.")

# --------------------------------------------------------------------------
# Sesli Yanıt Endpoint'i
# --------------------------------------------------------------------------
@app.post("/sesli-yanit")
async def generate_speech_endpoint(data: SesliYanitData):
    text_to_speak = data.text
    language_code = data.language
    if not tts_client: raise HTTPException(status_code=503, detail="Sesli yanıt hizmeti şu anda kullanılamıyor.")
    try:
        cleaned_text = temizle_emoji(text_to_speak)
        if not cleaned_text.strip(): raise HTTPException(status_code=400, detail="Seslendirilecek geçerli metin bulunamadı.")
        logger.info(f"🗣️ Sesli yanıt isteği: Dil: {language_code}, Metin: '{cleaned_text[:50]}...'")
        synthesis_input = texttospeech.SynthesisInput(text=cleaned_text)
        voice = texttospeech.VoiceSelectionParams(language_code=language_code, ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=1.0)
        response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        logger.info("✅ Sesli yanıt başarıyla oluşturuldu.")
        return Response(content=response.audio_content, media_type="audio/mpeg")
    except google_exceptions.GoogleAPIError as e:
        logger.error(f"❌ Google TTS API hatası: {e}")
        raise HTTPException(status_code=503, detail=f"Google sesli yanıt hizmetinde hata: {e}")
    except HTTPException as http_err: raise http_err
    except Exception as e:
        logger.error(f"❌ Sesli yanıt üretme hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Sesli yanıt oluşturulurken beklenmedik bir hata oluştu.")


# --------------------------------------------------------------------------
# Admin Şifre Değiştirme Endpoint'i
# --------------------------------------------------------------------------
@app.post("/admin/sifre-degistir")
async def change_admin_password_endpoint(
    creds: AdminCredentialsUpdate,
    auth: bool = Depends(check_admin)
):
    """Admin kullanıcı adı/şifresini değiştirmek için endpoint (Sadece bilgilendirme)."""
    new_username = creds.yeniKullaniciAdi.strip()
    new_password = creds.yeniSifre

    logger.warning(f"ℹ️ Admin şifre değiştirme isteği alındı (Kullanıcı: {new_username}). "
                   f"Gerçek değişiklik için .env dosyasını güncelleyip sunucuyu yeniden başlatın.")

    # Gerçek şifre değiştirme mekanizması burada olmalıydı (YAPILMIYOR)
    # Güvenlik ve basitlik için manuel .env güncellemesi önerilir.

    return {
        "mesaj": "Şifre değiştirme isteği alındı. Güvenlik nedeniyle, değişikliğin etkili olması için lütfen .env dosyasını manuel olarak güncelleyin ve uygulamayı yeniden başlatın."
    }

# --------------------------------------------------------------------------
# Uygulama Kapatma Olayı
# --------------------------------------------------------------------------
@app.on_event("shutdown")
def shutdown_event():
    """Uygulama kapatılırken kaynakları temizler."""
    logger.info("🚪 Uygulama kapatılıyor...")
    if google_creds_path and os.path.exists(google_creds_path):
        try:
            os.remove(google_creds_path)
            logger.info("✅ Geçici Google kimlik bilgisi dosyası silindi.")
        except OSError as e:
            logger.error(f"❌ Geçici Google kimlik bilgisi dosyası silinemedi: {e}")
    logger.info("👋 Uygulama kapatıldı.")

# --------------------------------------------------------------------------
# Ana Çalıştırma Bloğu (Geliştirme için)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 FastAPI uygulaması geliştirme modunda başlatılıyor...")
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")
    # Geliştirme sırasında otomatik yeniden yükleme için reload=True
    # Render gibi ortamlarda bu genellikle dışarıdan yönetilir.
    uvicorn.run("main:app", host=host, port=port, reload=True, log_level="info")