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
SECRET_KEY = os.getenv("SECRET_KEY", "cok-gizli-bir-anahtar-olmali")
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "*") # '*' yerine 'http://localhost:3000,https://neso-guncel.vercel.app' gibi
DB_DATA_DIR = os.getenv("DB_DATA_DIR", ".") # Veritabanı dosyalarının konumu (Render için önemli olabilir)

if not OPENAI_API_KEY:
    logger.critical("KRİTİK: OpenAI API anahtarı (OPENAI_API_KEY) bulunamadı! Yanıtlama özelliği çalışmayacak.")
if not GOOGLE_CREDS_BASE64:
    logger.warning("UYARI: Google Cloud kimlik bilgileri (GOOGLE_APPLICATION_CREDENTIALS_BASE64) bulunamadı. Sesli yanıt özelliği çalışmayabilir.")
if SECRET_KEY == "cok-gizli-bir-anahtar-olmali":
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
            if google_creds_path and os.path.exists(google_creds_path):
                os.remove(google_creds_path)
                google_creds_path = None # Yolu temizle
                logger.info("Temizlik: Başarısız TTS istemcisi sonrası geçici kimlik dosyası silindi.")
    except base64.binascii.Error as e:
         logger.error(f"❌ Google Cloud kimlik bilgileri base64 formatında değil: {e}")
    except Exception as e:
        logger.error(f"❌ Google Cloud kimlik bilgileri işlenirken hata: {e}")

# --------------------------------------------------------------------------
# FastAPI Uygulaması ve Güvenlik
# --------------------------------------------------------------------------
app = FastAPI(
    title="Neso Sipariş Asistanı API",
    version="1.2.7", # Versiyon güncellendi
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
    allow_methods=["*"], # Veya spesifik metodlar: ["GET", "POST", "DELETE", "OPTIONS"]
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
aktif_mutfak_websocketleri: set[WebSocket] = set()
aktif_admin_websocketleri: set[WebSocket] = set()

async def broadcast_message(connections: set[WebSocket], message: dict):
    """Belirtilen WebSocket bağlantılarına JSON mesajı gönderir."""
    if not connections: return

    message_json = json.dumps(message)
    # Kopya bir set üzerinde iterasyon yapalım ki döngü sırasında silme işlemi sorun çıkarmasın
    current_connections = connections.copy()
    tasks = []
    disconnected_sockets = set()

    for ws in current_connections:
        try:
            if ws.client_state == ws.client_state.CONNECTED: # Sadece bağlı olanlara gönder
                tasks.append(ws.send_text(message_json))
            else: # Bağlı değilse ayıkla
                disconnected_sockets.add(ws)
        except Exception as e: # Runtime Error vs. yakalamak için
             client_info = f"{ws.client.host}:{ws.client.port}" if ws.client else "Bilinmeyen"
             logger.warning(f"🔌 WebSocket gönderme sırasında istisna ({client_info}): {e}")
             disconnected_sockets.add(ws)

    if tasks: # Eğer gönderilecek görev varsa
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Görevlerin sonuçlarını işle (hataları yakala)
        for ws, result in zip(current_connections - disconnected_sockets, results): # disconnected olanları çıkar
            if isinstance(result, Exception):
                client_info = f"{ws.client.host}:{ws.client.port}" if ws.client else "Bilinmeyen"
                logger.warning(f"🔌 WebSocket gönderme hatası (gather) ({client_info}): {result}")
                disconnected_sockets.add(ws)

    # Kapananları ana set'ten çıkar
    if disconnected_sockets:
         for ws in disconnected_sockets:
            if ws in connections: # Ana sette hala varsa çıkar
                connections.remove(ws)
                client_info = f"{ws.client.host}:{ws.client.port}" if ws.client else "Bilinmeyen"
                logger.info(f"📉 WebSocket bağlantısı (hata/kapalı sonrası) kaldırıldı: {client_info}")

# --------------------------------------------------------------------------
# WebSocket Endpoint'leri
# --------------------------------------------------------------------------
async def websocket_lifecycle(websocket: WebSocket, connections: set[WebSocket], endpoint_name: str):
    """WebSocket bağlantı yaşam döngüsünü yöneten genel fonksiyon."""
    await websocket.accept()
    connections.add(websocket)
    client_host = websocket.client.host if websocket.client else "Bilinmeyen"
    client_port = websocket.client.port if websocket.client else "0" # Port bilgisi de eklendi
    client_id = f"{client_host}:{client_port}"
    logger.info(f"🔗 {endpoint_name} WS bağlandı: {client_id} (Toplam: {len(connections)})")
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                # else:
                #     logger.debug(f" Alınan WS mesajı ({endpoint_name} - {client_id}): {data[:200]}...") # Diğer mesajları logla (debug)
            except json.JSONDecodeError:
                logger.warning(f"⚠️ {endpoint_name} WS ({client_id}): Geçersiz JSON alındı: {data[:100]}...") # Kısaltılmış log
            except Exception as e:
                 logger.error(f"❌ {endpoint_name} WS ({client_id}) Mesaj işleme hatası: {e}")
    except WebSocketDisconnect as e:
        # 1000 (Normal), 1001 (Gidiyor), 1005 (Durum Yok), 1006 (Anormal Kapanma - tarayıcı kapatma vb.)
        if e.code in [status.WS_1000_NORMAL_CLOSURE, status.WS_1001_GOING_AWAY, 1005, 1006]:
             logger.info(f"🔌 {endpoint_name} WS kapatıldı (Kod {e.code}): {client_id}")
        else:
             logger.warning(f"🔌 {endpoint_name} WS beklenmedik şekilde kapandı (Kod {e.code}): {client_id}")
    except Exception as e: # Diğer olası hatalar (örn: Runtime Error)
        logger.error(f"❌ {endpoint_name} WS kritik hatası ({client_id}): {e}")
    finally:
        # Bağlantı set'ten güvenli bir şekilde kaldırılır
        if websocket in connections:
            connections.remove(websocket)
        logger.info(f"📉 {endpoint_name} WS kaldırıldı: {client_id} (Kalan: {len(connections)})")

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
DB_PATH = os.path.join(DB_DATA_DIR, DB_NAME)
MENU_DB_PATH = os.path.join(DB_DATA_DIR, MENU_DB_NAME)

# Veritabanı dizininin var olduğundan emin ol
os.makedirs(DB_DATA_DIR, exist_ok=True)

def get_db_connection(db_path: str) -> sqlite3.Connection:
    """Veritabanı bağlantısı oluşturur."""
    try:
        # timeout eklendi, WAL modu açılabilir (daha iyi eşzamanlılık için ama dikkatli kullanılmalı)
        conn = sqlite3.connect(db_path, timeout=10) #, isolation_level=None)
        # conn.execute("PRAGMA journal_mode=WAL;") # WAL modu (isteğe bağlı)
        conn.row_factory = sqlite3.Row # Sütun adlarıyla erişim için
        return conn
    except sqlite3.Error as e:
        logger.critical(f"❌ KRİTİK: Veritabanı bağlantısı kurulamadı ({db_path}): {e}")
        # Burada uygulama belki de başlamamalı? init_db içinde kontrol ediliyor.
        raise HTTPException(status_code=503, detail=f"Veritabanı bağlantı hatası: {e}")

async def update_table_status(masa_id: str, islem: str = "Erişim"):
    """Veritabanındaki masa durumunu günceller ve admin paneline bildirir."""
    now = datetime.now()
    try:
        with get_db_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            # Son işlem daha açıklayıcı olabilir
            son_islem_str = f"{islem} @ {now.strftime('%H:%M:%S')}"
            cursor.execute("""
                INSERT INTO masa_durumlar (masa_id, son_erisim, aktif, son_islem)
                VALUES (?, ?, TRUE, ?)
                ON CONFLICT(masa_id) DO UPDATE SET
                    son_erisim = excluded.son_erisim,
                    aktif = excluded.aktif,
                    son_islem = excluded.son_islem
            """, (masa_id, now.strftime("%Y-%m-%d %H:%M:%S.%f"), son_islem_str)) # Milisaniye eklendi
            conn.commit()

        # Sadece admin'e bildirim gönderelim (mutfak ve masa asistanı için gereksiz olabilir)
        if aktif_admin_websocketleri:
             await broadcast_message(aktif_admin_websocketleri, {
                 "type": "masa_durum",
                 "data": {"masaId": masa_id, "sonErisim": now.isoformat(), "aktif": True, "sonIslem": son_islem_str}
             })

    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (masa durumu güncellenemedi - Masa {masa_id}): {e}")
    except Exception as e:
        logger.error(f"❌ Masa durumu güncelleme hatası (Masa {masa_id}): {e}")

# --------------------------------------------------------------------------
# Middleware
# --------------------------------------------------------------------------
@app.middleware("http")
async def track_active_users(request: Request, call_next):
    """Gelen isteklerde masa ID'si varsa durumu günceller."""
    # Path parametresinden masaId'yi al (örn: /masa/{masaId}/...)
    masa_id = request.path_params.get("masaId")

    # Endpoint'e göre işlem belirle
    endpoint_func = request.scope.get("endpoint")
    endpoint_name = endpoint_func.__name__ if endpoint_func else request.url.path
    islem = f"{request.method} {endpoint_name}"

    if masa_id:
        # Arka planda çalıştırarak isteği bloklamamasını sağlayabiliriz (opsiyonel)
        asyncio.create_task(update_table_status(str(masa_id), islem))
    else:
        # Masa ID'si olmayan istekler (örn: admin paneli) için işlem yapma
        pass

    try:
        response = await call_next(request)
        return response
    except Exception as e:
        # Genel hata yakalama
        logger.exception(f"💥 Beklenmedik HTTP Middleware Hatası: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Sunucuda beklenmedik bir hata oluştu."}
        )

# --------------------------------------------------------------------------
# Aktif Masalar Endpoint
# --------------------------------------------------------------------------
@app.get("/aktif-masalar")
async def get_active_tables_endpoint():
    """Son X dakika içinde aktif olan masaları döndürür."""
    ACTIVE_MINUTES = 5 # Aktiflik süresi (dakika)
    try:
        active_time_limit = datetime.now() - timedelta(minutes=ACTIVE_MINUTES)
        with get_db_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT masa_id, son_erisim, aktif, son_islem FROM masa_durumlar
                WHERE son_erisim >= ? AND aktif = TRUE ORDER BY son_erisim DESC
            """, (active_time_limit.strftime("%Y-%m-%d %H:%M:%S.%f"),)) # Milisaniye eklendi
            results = cursor.fetchall()
            # Satırları dict'e çevir
            active_tables_data = [dict(row) for row in results]
        # logger.info(f"📊 Aktif masalar sorgulandı, {len(active_tables_data)} adet bulundu.") # Çok sık log
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
    correct_username = ADMIN_USERNAME
    correct_password = ADMIN_PASSWORD
    # Güvenli karşılaştırma (secrets modülü ile)
    import secrets
    is_user_ok = secrets.compare_digest(credentials.username.encode('utf8'), correct_username.encode('utf8'))
    is_pass_ok = secrets.compare_digest(credentials.password.encode('utf8'), correct_password.encode('utf8'))

    if not (is_user_ok and is_pass_ok):
        logger.warning(f"🔒 Başarısız admin girişi denemesi: Kullanıcı adı '{credentials.username}'")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Geçersiz kimlik bilgileri",
            headers={"WWW-Authenticate": 'Basic realm="Admin Alanı"'}, # Realm eklendi
        )
    # logger.debug(f"🔑 Admin girişi başarılı: {credentials.username}") # Başarıyı loglama
    return True # Başarılı ise True döner

# --------------------------------------------------------------------------
# Pydantic Modelleri (Veri Doğrulama için)
# --------------------------------------------------------------------------
class SepetItem(BaseModel):
    urun: str = Field(..., min_length=1, description="Ürün adı")
    adet: int = Field(..., gt=0, description="Ürün adedi (0'dan büyük)")
    fiyat: float | None = Field(None, description="Ürün fiyatı (backend'den alınır)")
    kategori: str | None = Field(None, description="Ürün kategorisi (bilgi amaçlı)")

class SiparisEkleData(BaseModel):
    masa: str = Field(..., min_length=1, description="Masa numarası")
    sepet: list[SepetItem] = Field(..., min_items=1, description="Sipariş sepeti (en az 1 ürün)")
    istek: str | None = Field(None, description="Müşterinin orijinal isteği/notu")
    yanit: str | None = Field(None, description="AI tarafından verilen yanıt")

class SiparisGuncelleData(BaseModel):
    masa: str = Field(..., min_length=1, description="Masa numarası")
    durum: str = Field(..., description="Yeni sipariş durumu ('bekliyor', 'hazirlaniyor', 'hazir', 'iptal')")
    id: int = Field(..., description="Güncellenecek siparişin ID'si") # ID artık zorunlu

class MenuEkleData(BaseModel):
    ad: str = Field(..., min_length=1)
    fiyat: float = Field(..., gt=0) # Fiyat 0'dan büyük olmalı
    kategori: str = Field(..., min_length=1)

# class AdminCredentialsUpdate(BaseModel): # Bu model kullanılmıyor, NameError'a neden oluyordu
#     yeniKullaniciAdi: str = Field(..., min_length=1)
#     yeniSifre: str = Field(..., min_length=4)

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
    sepet_verisi = data.sepet
    istek_orijinal = data.istek
    zaman_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f") # Milisaniye eklendi
    logger.info(f"📥 Yeni sipariş isteği: Masa {masa}, Sepet: {len(sepet_verisi)} ürün, İstek: '{istek_orijinal[:50]}...'")

    price_dict = get_menu_price_dict()
    processed_sepet = []
    for item in sepet_verisi:
        item_dict = item.model_dump()
        urun_adi_lower = item_dict['urun'].lower().strip()
        item_dict['fiyat'] = price_dict.get(urun_adi_lower, 0.0)
        processed_sepet.append(item_dict)

    try:
        sepet_json = json.dumps(processed_sepet, ensure_ascii=False) # Türkçe karakterler için ensure_ascii=False
        siparis_id = None
        with get_db_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO siparisler (masa, istek, yanit, sepet, zaman, durum)
                VALUES (?, ?, ?, ?, ?, 'bekliyor')
            """, (masa, istek_orijinal, yanit, sepet_json, zaman_str))
            siparis_id = cursor.lastrowid # Yeni eklenen siparişin ID'sini al
            conn.commit()

        if siparis_id is None:
             raise sqlite3.Error("Sipariş ID alınamadı!")

        logger.info(f"💾 Sipariş veritabanına kaydedildi: Masa {masa}, Sipariş ID: {siparis_id}")

        siparis_bilgisi = {
            "type": "siparis",
            "data": {
                "id": siparis_id, "masa": masa, "istek": istek_orijinal,
                "sepet": processed_sepet, "zaman": zaman_str, "durum": "bekliyor"
            }
        }
        await broadcast_message(aktif_mutfak_websocketleri, siparis_bilgisi)
        await broadcast_message(aktif_admin_websocketleri, siparis_bilgisi)
        logger.info(f"📢 Yeni sipariş bildirimi gönderildi (ID: {siparis_id}): Mutfak ({len(aktif_mutfak_websocketleri)}), Admin ({len(aktif_admin_websocketleri)})")
        asyncio.create_task(update_table_status(masa, f"Sipariş verdi ({len(processed_sepet)} ürün)"))
        return {"mesaj": "Sipariş başarıyla kaydedildi ve ilgili birimlere iletildi.", "siparisId": siparis_id}

    except sqlite3.Error as e:
        logger.exception(f"❌ Veritabanı hatası (sipariş eklenemedi - Masa {masa}): {e}")
        raise HTTPException(status_code=503, detail=f"Sipariş veritabanına kaydedilirken hata oluştu.")
    except json.JSONDecodeError as e:
         logger.exception(f"❌ Sepet JSON'a çevirme hatası (Masa {masa}): {e}")
         raise HTTPException(status_code=400, detail="Sipariş sepeti verisi geçersiz.")
    except Exception as e:
        logger.exception(f"❌ Sipariş ekleme sırasında genel hata (Masa {masa}): {e}")
        raise HTTPException(status_code=500, detail=f"Sipariş eklenirken beklenmedik bir hata oluştu.")

@app.post("/siparis-guncelle", status_code=status.HTTP_200_OK)
async def update_order_status_endpoint(data: SiparisGuncelleData, auth: bool = Depends(check_admin)):
    siparis_id = data.id
    masa = data.masa
    durum = data.durum
    logger.info(f"🔄 Sipariş durumu güncelleme isteği: ID: {siparis_id}, Masa: {masa}, Yeni Durum: {durum}")

    valid_statuses = ["hazirlaniyor", "hazir", "iptal", "bekliyor"]
    if durum not in valid_statuses:
         logger.error(f"❌ Sipariş güncelleme hatası (ID: {siparis_id}): Geçersiz durum '{durum}'.")
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Geçersiz durum: {durum}.")

    rows_affected = 0
    try:
        with get_db_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE siparisler SET durum = ? WHERE id = ?", (durum, siparis_id))
            rows_affected = cursor.rowcount
            conn.commit()

        if rows_affected > 0:
             logger.info(f"💾 Sipariş durumu güncellendi (ID: {siparis_id}): Yeni Durum: {durum}")
             notification = {
                 "type": "durum",
                 "data": {"id": siparis_id, "masa": masa, "durum": durum, "zaman": datetime.now().isoformat()}
             }
             await broadcast_message(aktif_mutfak_websocketleri, notification)
             await broadcast_message(aktif_admin_websocketleri, notification)
             logger.info(f"📢 Sipariş durum güncellemesi bildirildi (ID: {siparis_id}): Durum: {durum}")
             asyncio.create_task(update_table_status(masa, f"Sipariş (ID:{siparis_id}) durumu -> {durum}"))
             return {"success": True, "message": f"Sipariş (ID: {siparis_id}) durumu '{durum}' olarak güncellendi."}
        else:
             logger.warning(f"⚠️ Sipariş durumu güncellenemedi (ID: {siparis_id}): Sipariş bulunamadı veya durum zaten aynı.")
             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Sipariş (ID: {siparis_id}) bulunamadı veya güncellenmesi gerekmiyor.")

    except sqlite3.Error as e:
         logger.exception(f"❌ Veritabanı hatası (sipariş durumu güncellenemedi - ID: {siparis_id}): {e}")
         raise HTTPException(status_code=503, detail=f"Sipariş durumu güncellenirken veritabanı hatası oluştu.")
    except Exception as e:
        logger.exception(f"❌ Sipariş durumu güncelleme sırasında genel hata (ID: {siparis_id}): {e}")
        raise HTTPException(status_code=500, detail=f"Sipariş durumu güncellenirken beklenmedik bir hata oluştu.")

# --- DÜZELTME UYGULANDI: Backend'de JSON parse etme kaldırıldı ---
@app.get("/siparisler")
def get_orders_endpoint(auth: bool = Depends(check_admin)):
    """Tüm siparişleri ID'ye göre tersten sıralı ve sepeti HAM string olarak döndürür."""
    logger.info("Sipariş listesi isteniyor (/siparisler)...")
    try:
        with get_db_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            # Sepet verisini ham string olarak seçiyoruz
            cursor.execute("SELECT id, masa, istek, yanit, sepet, zaman, durum FROM siparisler ORDER BY id DESC")
            rows = cursor.fetchall()
            # Satırları doğrudan dict listesine çeviriyoruz (sepet parse edilmeden)
            orders_data = [dict(row) for row in rows]

            # Backend JSON parse bloğu yorum satırı yapıldı/kaldırıldı:
            # for row_index, order_dict_item in enumerate(orders_data):
            #     try:
            #         if order_dict_item['sepet'] and isinstance(order_dict_item['sepet'], str):
            #             orders_data[row_index]['sepet'] = json.loads(order_dict_item['sepet'])
            #         elif not order_dict_item['sepet']: # Eğer sepet null veya boş string ise
            #             orders_data[row_index]['sepet'] = []
            #         # Eğer zaten bir list ise (beklenmedik durum ama olabilir) dokunma
            #         elif not isinstance(order_dict_item['sepet'], list):
            #             logger.warning(f"⚠️ Sipariş listesi: Beklenmedik sepet tipi (ID: {order_dict_item['id']}), boş liste olarak ayarlandı.")
            #             orders_data[row_index]['sepet'] = []
            #     except json.JSONDecodeError:
            #         logger.warning(f"⚠️ Sipariş listesi: Geçersiz sepet JSON (ID: {order_dict_item['id']})")
            #         orders_data[row_index]['sepet'] = [] # Hata durumunda boş liste ata

        logger.info(f"✅ Sipariş listesi başarıyla alındı ({len(orders_data)} adet).")
        return {"orders": orders_data}
    except sqlite3.Error as e:
        logger.exception(f"❌ Veritabanı hatası (siparişler alınamadı): {e}")
        raise HTTPException(status_code=503, detail="Veritabanı hatası nedeniyle siparişler alınamadı.")
    except Exception as e:
        logger.exception(f"❌ Siparişler alınırken genel hata: {e}")
        raise HTTPException(status_code=500, detail="Siparişler alınırken sunucu hatası oluştu.")

# --------------------------------------------------------------------------
# Veritabanı Başlatma
# --------------------------------------------------------------------------
def init_db(db_path: str):
    """Ana veritabanı tablolarını oluşturur veya doğrular."""
    logger.info(f"Ana veritabanı kontrol ediliyor: {db_path}")
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            # Siparisler tablosu (durum sütunu için NOT NULL eklendi)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS siparisler (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    masa TEXT NOT NULL,
                    istek TEXT,
                    yanit TEXT,
                    sepet TEXT,                 -- JSON string olarak saklanacak
                    zaman TEXT NOT NULL,        -- ISO formatında veya YYYY-MM-DD HH:MM:SS.ffffff
                    durum TEXT DEFAULT 'bekliyor' CHECK(durum IN ('bekliyor', 'hazirlaniyor', 'hazir', 'iptal')) NOT NULL
                )""")
            # Masa Durumları tablosu (son_erisim TEXT olarak düzeltildi)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS masa_durumlar (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    masa_id TEXT UNIQUE NOT NULL,
                    son_erisim TEXT NOT NULL,  -- TIMESTAMP yerine TEXT (ISO formatı)
                    aktif BOOLEAN DEFAULT TRUE,
                    son_islem TEXT
                )""")
            # İndeksler
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_siparisler_masa_zaman ON siparisler(masa, zaman DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_masa_durumlar_erisim ON masa_durumlar(son_erisim DESC)")
            conn.commit()
            logger.info(f"✅ Ana veritabanı ({db_path}) başarıyla doğrulandı/oluşturuldu.")
    except Exception as e:
        logger.exception(f"❌ KRİTİK HATA: Ana veritabanı ({db_path}) başlatılamadı! Hata: {e}") # exception logla
        raise # Uygulamanın başlamasını engelle

def init_menu_db(db_path: str):
    """Menü veritabanı tablolarını oluşturur veya doğrular."""
    logger.info(f"Menü veritabanı kontrol ediliyor: {db_path}")
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            # Kategoriler tablosu
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS kategoriler (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    isim TEXT UNIQUE NOT NULL COLLATE NOCASE
                )""")
            # Menu tablosu (stok_durumu için CHECK constraint eklendi, kategori_id NOT NULL yapıldı)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS menu (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ad TEXT NOT NULL COLLATE NOCASE,
                    fiyat REAL NOT NULL CHECK(fiyat >= 0),
                    kategori_id INTEGER NOT NULL,
                    stok_durumu INTEGER DEFAULT 1 CHECK(stok_durumu IN (0, 1)), /* 1: Var, 0: Yok */
                    FOREIGN KEY (kategori_id) REFERENCES kategoriler(id) ON DELETE CASCADE,
                    UNIQUE(ad, kategori_id)
                )""")
            # İndeksler
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_menu_kategori ON menu(kategori_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_menu_ad ON menu(ad COLLATE NOCASE)") # COLLATE NOCASE eklendi
            conn.commit()
            logger.info(f"✅ Menü veritabanı ({db_path}) başarıyla doğrulandı/oluşturuldu.")
    except Exception as e:
        logger.exception(f"❌ KRİTİK HATA: Menü veritabanı ({db_path}) başlatılamadı! Hata: {e}") # exception logla
        raise

# Uygulama başlangıcında veritabanlarını başlat
try:
    init_db(DB_PATH)
    init_menu_db(MENU_DB_PATH)
except Exception as e:
     logger.critical(f"💥 Uygulama başlatılamadı: Veritabanı başlatma hatası. Detaylar yukarıdaki loglarda. Hata: {e}")
     raise SystemExit(f"Uygulama başlatılamadı: Veritabanı başlatma hatası - {e}")


# --------------------------------------------------------------------------
# Menü Yönetimi Yardımcıları ve Endpoint'leri
# --------------------------------------------------------------------------
def get_menu_for_prompt():
    """AI prompt'u için STOKTAKİ menü öğelerini formatlar."""
    menu_items = []
    try:
        with get_db_connection(MENU_DB_PATH) as conn:
            cursor = conn.cursor()
            # Sadece stokta olanları (stok_durumu=1) ve kategorileri çek
            cursor.execute("""
                SELECT k.isim, m.ad FROM menu m
                JOIN kategoriler k ON m.kategori_id = k.id
                WHERE m.stok_durumu = 1
                ORDER BY k.isim, m.ad COLLATE NOCASE
            """)
            menu_items = cursor.fetchall()

        if not menu_items:
             return "Üzgünüm, şu anda menüde servis edebileceğimiz bir ürün bulunmuyor."

        # Kategorilere göre grupla
        kategorili_menu = {}
        for kat_row in menu_items: # Düzeltilmiş unpacking
             kategori_isim = kat_row['isim'] # Sütun adıyla erişim
             urun_ad = kat_row['ad']         # Sütun adıyla erişim
             kategorili_menu.setdefault(kategori_isim, []).append(urun_ad)

        # Prompt metnini oluştur
        menu_aciklama_lines = ["Mevcut ve stokta olan menümüz şöyledir:"]
        for kategori, urunler_list in kategorili_menu.items(): # Düzeltilmiş değişken adı
            menu_aciklama_lines.append(f"- {kategori}: {', '.join(urunler_list)}")

        return "\n".join(menu_aciklama_lines)

    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (menü prompt için alınamadı): {e}")
        return "Üzgünüm, menü bilgisine şu an ulaşılamıyor." # AI'a daha net bilgi
    except Exception as e:
        logger.error(f"❌ Menü prompt'u oluşturulurken genel hata: {e}")
        return "Üzgünüm, menü bilgisi yüklenirken bir sorun oluştu."


def get_menu_price_dict():
    """Ürün adı (küçük harf, trim edilmiş) -> fiyat eşleşmesini içeren sözlük döndürür."""
    fiyatlar = {}
    logger.debug("Fiyat sözlüğü oluşturuluyor...") # Log eklendi
    try:
        with get_db_connection(MENU_DB_PATH) as conn:
            cursor = conn.cursor()
            # Sadece ad ve fiyatı çek, küçük harfe çevir ve boşlukları temizle
            cursor.execute("SELECT LOWER(TRIM(ad)), fiyat FROM menu")
            # fetchall yerine dict comprehension ile direkt oluştur
            fiyatlar = {ad: fiyat for ad, fiyat in cursor.fetchall()}
        logger.debug(f"Fiyat sözlüğü başarıyla oluşturuldu: {len(fiyatlar)} ürün.")
    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (fiyat sözlüğü alınamadı): {e}")
    except Exception as e:
        logger.error(f"❌ Fiyat sözlüğü oluşturulurken hata: {e}")
    return fiyatlar # Hata olsa bile boş sözlük döner

# Sistem mesajını global olarak tanımla ve başlangıçta oluştur
SISTEM_MESAJI_ICERIK = "" # Başlangıçta boş
SYSTEM_PROMPT = {} # Başlangıçta boş

def update_system_prompt():
    """Sistem prompt'unu güncel menü ile yeniler."""
    global SISTEM_MESAJI_ICERIK, SYSTEM_PROMPT
    logger.info("🔄 Sistem mesajı (menü bilgisi) güncelleniyor...")
    # Sabit metinleri tanımla
    giris_metni = (
        "Sen, Gaziantep'teki Fıstık Kafe için özel olarak tasarlanmış, Neso adında bir sipariş asistanısın. "
        "Görevin, masadaki müşterilerin sesli veya yazılı taleplerini anlayıp menüdeki ürünlerle eşleştirerek siparişlerini almak ve bu siparişleri mutfağa doğru bir şekilde iletmektir. "
        "Siparişleri sen hazırlamıyorsun, sadece alıyorsun. "
        "Her zaman nazik, yardımsever, samimi ve çözüm odaklı olmalısın. Gaziantep ağzıyla veya şivesiyle konuşmamalısın, standart ve kibar bir Türkçe kullanmalısın. "
        "Müşterinin ne istediğini tam anlayamazsan, soruyu tekrar sormaktan veya seçenekleri netleştirmesini istemekten çekinme. "
        "Sipariş tamamlandığında veya müşteri teşekkür ettiğinde 'Afiyet olsun!' demeyi unutma.\n\n"
    )
    menu_bilgisi = get_menu_for_prompt() # Güncel menüyü al
    SISTEM_MESAJI_ICERIK = giris_metni + menu_bilgisi
    SYSTEM_PROMPT = {"role": "system", "content": SISTEM_MESAJI_ICERIK}
    logger.info("✅ Sistem mesajı güncellendi.")

# Uygulama başlangıcında sistem prompt'unu oluştur/güncelle
update_system_prompt()


@app.get("/menu")
def get_full_menu_endpoint():
    """Tüm menüyü kategorilere göre gruplanmış ve stok bilgisiyle döndürür."""
    logger.info("Tam menü isteniyor (/menu)...")
    try:
        full_menu_data = []
        with get_db_connection(MENU_DB_PATH) as conn:
            cursor = conn.cursor()
            # Önce kategorileri çek
            cursor.execute("SELECT id, isim FROM kategoriler ORDER BY isim COLLATE NOCASE")
            kategoriler = cursor.fetchall()
            # Her kategori için ürünleri çek
            for kat_row in kategoriler:
                cursor.execute("""
                    SELECT ad, fiyat, stok_durumu FROM menu
                    WHERE kategori_id = ? ORDER BY ad COLLATE NOCASE
                """, (kat_row['id'],))
                urunler_rows = cursor.fetchall()
                # Ürünleri dict listesine çevir
                urunler_list = [dict(urun) for urun in urunler_rows]
                full_menu_data.append({"kategori": kat_row['isim'], "urunler": urunler_list})
        logger.info(f"✅ Tam menü başarıyla alındı ({len(full_menu_data)} kategori).")
        return {"menu": full_menu_data}
    except sqlite3.Error as e:
        logger.exception(f"❌ Veritabanı hatası (tam menü alınamadı): {e}")
        raise HTTPException(status_code=503, detail="Veritabanı hatası nedeniyle menü alınamadı.")
    except Exception as e:
        logger.exception(f"❌ Tam menü alınırken genel hata: {e}")
        raise HTTPException(status_code=500, detail=f"Menü bilgileri alınırken sunucu hatası oluştu.")


@app.post("/menu/ekle", status_code=status.HTTP_201_CREATED)
async def add_menu_item_endpoint(item_data: MenuEkleData, auth: bool = Depends(check_admin)):
    # Gelen veriyi temizle
    item_name = item_data.ad.strip()
    item_price = item_data.fiyat
    item_category = item_data.kategori.strip()
    logger.info(f"➕ Menüye ekleme isteği: Ad='{item_name}', Fiyat={item_price}, Kategori='{item_category}'")

    if item_price <= 0: # Fiyat kontrolü (Pydantic'te gt=0 var ama ek kontrol)
         raise HTTPException(status_code=400, detail="Fiyat 0'dan büyük olmalıdır.")

    try:
        with get_db_connection(MENU_DB_PATH) as conn:
            cursor = conn.cursor()
            # Kategoriyi ekle veya ID'sini al (Büyük/küçük harf duyarsız)
            cursor.execute("INSERT OR IGNORE INTO kategoriler (isim) VALUES (?)", (item_category,))
            # Ekledikten sonra ID'yi al
            cursor.execute("SELECT id FROM kategoriler WHERE isim = ? COLLATE NOCASE", (item_category,))
            category_result = cursor.fetchone()
            if not category_result:
                 logger.error(f" Kategori bulunamadı veya eklenemedi: {item_category}")
                 raise HTTPException(status_code=500, detail="Kategori işlenirken hata oluştu.")
            category_id = category_result['id'] # dict'ten ID'yi al

            # Ürünü ekle (UNIQUE constraint hatasını yakalamak için try-except yerine INSERT OR FAIL kullanılabilir)
            cursor.execute("""
                INSERT INTO menu (ad, fiyat, kategori_id, stok_durumu)
                VALUES (?, ?, ?, 1)
                """, (item_name, item_price, category_id))
            item_id = cursor.lastrowid # Yeni eklenen ürünün ID'si
            conn.commit()

        logger.info(f"💾 Menü öğesi başarıyla eklendi: ID {item_id}, Ad: {item_name}")
        update_system_prompt() # Sistem mesajını (AI prompt) güncelle
        return {"mesaj": f"'{item_name}' menüye başarıyla eklendi.", "itemId": item_id}

    except sqlite3.IntegrityError as e: # UNIQUE constraint hatası
         # Hata mesajını daha detaylı logla
         logger.warning(f"⚠️ Menü ekleme hatası (IntegrityError): '{item_name}', '{item_category}' kategorisinde zaten mevcut olabilir. Hata: {e}")
         raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"'{item_name}' ürünü '{item_category}' kategorisinde zaten mevcut.")
    except sqlite3.Error as e:
        logger.exception(f"❌ Veritabanı hatası (menü öğesi eklenemedi): {e}")
        raise HTTPException(status_code=503, detail="Veritabanı hatası nedeniyle ürün eklenemedi.")
    except Exception as e:
        logger.exception(f"❌ Menü öğesi eklenirken genel hata: {e}")
        raise HTTPException(status_code=500, detail="Menü öğesi eklenirken beklenmedik bir hata oluştu.")

@app.delete("/menu/sil", status_code=status.HTTP_200_OK) # Başarı kodu 200 veya 204 olabilir
async def delete_menu_item_endpoint(urun_adi: str = Query(..., min_length=1, description="Silinecek ürünün tam adı"), auth: bool = Depends(check_admin)):
    item_name_to_delete = urun_adi.strip()
    logger.info(f"➖ Menüden silme isteği: Ad='{item_name_to_delete}'")

    rows_affected = 0
    try:
        with get_db_connection(MENU_DB_PATH) as conn:
            cursor = conn.cursor()
            # Silme işlemini yap ve etkilenen satır sayısını al (Büyük/küçük harf duyarsız)
            cursor.execute("DELETE FROM menu WHERE ad = ? COLLATE NOCASE", (item_name_to_delete,))
            rows_affected = cursor.rowcount
            conn.commit()

        if rows_affected > 0:
            logger.info(f"🗑️ Menü öğesi silindi: Ad='{item_name_to_delete}', Etkilenen: {rows_affected}")
            update_system_prompt() # Sistem mesajını (AI prompt) güncelle
            return {"mesaj": f"'{item_name_to_delete}' isimli ürün menüden başarıyla silindi."}
        else:
            # Ürün bulunamadıysa 404 hatası döndür
            logger.warning(f"⚠️ Menü silme: '{item_name_to_delete}' adında ürün bulunamadı.")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{item_name_to_delete}' adında ürün menüde bulunamadı.")
    except sqlite3.Error as e:
        logger.exception(f"❌ Veritabanı hatası (menü öğesi silinemedi): {e}")
        raise HTTPException(status_code=503, detail="Veritabanı hatası nedeniyle ürün silinemedi.")
    except Exception as e:
        logger.exception(f"❌ Menü öğesi silinirken genel hata: {e}")
        raise HTTPException(status_code=500, detail="Menü öğesi silinirken beklenmedik bir hata oluştu.")

# --------------------------------------------------------------------------
# AI Yanıt Üretme Endpoint'i
# --------------------------------------------------------------------------
@app.post("/yanitla")
async def handle_message_endpoint(data: dict = Body(...)): # Pydantic modeli daha iyi olur
    user_message = data.get("text", "").strip()
    table_id = data.get("masa", "bilinmiyor")
    if not user_message: raise HTTPException(status_code=400, detail="Mesaj içeriği boş olamaz.")
    logger.info(f"💬 Mesaj alındı: Masa {table_id}, Mesaj: '{user_message[:100]}...'") # Loglanan mesaj uzunluğu arttı

    if not openai_client:
         logger.error(f" OpenAI istemcisi mevcut değil, yanıt verilemiyor (Masa {table_id}).")
         raise HTTPException(status_code=503, detail="Yapay zeka hizmeti şu anda kullanılamıyor.")

    try:
        # Güncel sistem mesajını (menü bilgisi dahil) kullan
        messages = [SYSTEM_PROMPT, {"role": "user", "content": user_message}]
        # OpenAI API çağrısı
        response = openai_client.chat.completions.create(
             model="gpt-3.5-turbo", # Model adı doğru varsayılıyor
             messages=messages,
             temperature=0.6, # Yaratıcılık seviyesi (ayarlanabilir)
             max_tokens=150 # Yanıt uzunluğu limiti
        )
        # Yanıtı al ve temizle
        ai_reply = response.choices[0].message.content.strip() if response.choices else "Üzgünüm, anlayamadım."
        logger.info(f"🤖 AI yanıtı üretildi: Masa {table_id}, Yanıt: '{ai_reply[:100]}...'") # Loglanan yanıt uzunluğu arttı
        return {"reply": ai_reply}
    except OpenAIError as e: # OpenAI'ye özgü hatalar
        logger.error(f"❌ OpenAI API hatası (Masa {table_id}): {e.status_code} - {e.response.text if e.response else e}")
        raise HTTPException(status_code=e.status_code or 503, detail=f"Yapay zeka servisinden yanıt alınamadı: {e.code}")
    except Exception as e: # Diğer genel hatalar
        logger.exception(f"❌ AI yanıtı üretme hatası (Masa {table_id}): {e}")
        raise HTTPException(status_code=500, detail=f"Yapay zeka yanıtı alınırken bir sunucu hatası oluştu.")

# --------------------------------------------------------------------------
# İstatistik Hesaplama Yardımcı Fonksiyonu
# --------------------------------------------------------------------------
def calculate_statistics(cart_data_tuples: list[tuple]):
    """Verilen sepet verilerinden toplam ürün adedini ve geliri hesaplar."""
    total_items = 0
    total_revenue = 0.0
    menu_prices = get_menu_price_dict() # Güncel fiyatları al

    for (cart_json_str,) in cart_data_tuples:
        if not cart_json_str: continue
        try:
            items_in_cart = json.loads(cart_json_str)
            if not isinstance(items_in_cart, list): continue
            for item in items_in_cart:
                 if not isinstance(item, dict): continue
                 item_name = item.get("urun")
                 quantity = item.get("adet", 1)
                 if item_name and isinstance(quantity, (int, float)) and quantity > 0:
                     total_items += quantity
                     # Fiyatı menüden al, item içindeki fiyata güvenme
                     price = menu_prices.get(item_name.lower().strip(), 0.0)
                     total_revenue += quantity * price
        except Exception as e:
             logger.warning(f"⚠️ İstatistik hesaplama: Sepet işleme hatası ({e}): {cart_json_str[:50]}...")
    return total_items, round(total_revenue, 2) # Geliri 2 ondalık basamağa yuvarla

# --------------------------------------------------------------------------
# İstatistik Endpoint'leri
# --------------------------------------------------------------------------
@app.get("/istatistik/en-cok-satilan")
def get_popular_items_endpoint():
    logger.info("Popüler ürünler isteniyor...")
    try:
        item_counts = {}
        with get_db_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            # Sadece 'iptal' olmayan siparişlerin sepetlerini al
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
                     quantity = item.get("adet", 1) # Adet yoksa 1 varsay
                     # Ürün adı geçerliyse ve adet sayıysa ve 0'dan büyükse say
                     if item_name and isinstance(quantity, (int, float)) and quantity > 0:
                         item_counts[item_name] = item_counts.get(item_name, 0) + quantity
            except Exception as e:
                 logger.warning(f"⚠️ Popüler ürünler: Sepet işleme hatası ({e}): {sepet_json_str[:50]}...")

        # En çok satılanları adetlerine göre sırala ve ilk 5'i al
        sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        popular_items_data = [{"urun": item, "adet": count} for item, count in sorted_items]
        logger.info(f"✅ Popüler ürünler hesaplandı: {len(popular_items_data)} ürün.")
        return popular_items_data
    except sqlite3.Error as e:
        logger.exception(f"❌ Veritabanı hatası (popüler ürünler): {e}")
        raise HTTPException(status_code=503, detail="Veritabanı hatası nedeniyle popüler ürünler alınamadı.")
    except Exception as e:
        logger.exception(f"❌ Popüler ürünler hesaplanırken genel hata: {e}")
        raise HTTPException(status_code=500, detail="Popüler ürünler hesaplanırken sunucu hatası oluştu.")

@app.get("/istatistik/gunluk")
def get_daily_stats_endpoint():
    logger.info("Günlük istatistikler isteniyor...")
    today_str = datetime.now().strftime("%Y-%m-%d")
    try:
        with get_db_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            # Günün başlangıcını ve sonunu kullanarak sorgula ve durumu kontrol et
            cursor.execute("""
                SELECT sepet FROM siparisler
                WHERE zaman >= ? AND zaman < ? AND durum != 'iptal'
            """, (f"{today_str} 00:00:00", f"{today_str} 23:59:59.999999"))
            daily_data = cursor.fetchall()
        total_items, total_revenue = calculate_statistics(daily_data) # Yardımcı fonksiyonu kullan
        logger.info(f"✅ Günlük istatistikler hesaplandı: {total_items} ürün, {total_revenue} TL.")
        return {"tarih": today_str, "siparis_sayisi": total_items, "gelir": total_revenue}
    except sqlite3.Error as e:
        logger.exception(f"❌ Veritabanı hatası (günlük istatistik): {e}")
        raise HTTPException(status_code=503, detail="Veritabanı hatası nedeniyle günlük istatistikler alınamadı.")
    except Exception as e:
        logger.exception(f"❌ Günlük istatistik hesaplanırken genel hata: {e}")
        raise HTTPException(status_code=500, detail="Günlük istatistikler hesaplanırken sunucu hatası oluştu.")

@app.get("/istatistik/aylik")
def get_monthly_stats_endpoint():
    logger.info("Aylık istatistikler isteniyor (son 30 gün)...")
    # Son 30 günü kapsayan başlangıç tarihini hesapla
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S.%f")
    try:
        with get_db_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            # Başlangıç tarihinden itibaren ve durumu 'iptal' olmayanları al
            cursor.execute("SELECT sepet FROM siparisler WHERE zaman >= ? AND durum != 'iptal'", (start_date,))
            monthly_data = cursor.fetchall()
        total_items, total_revenue = calculate_statistics(monthly_data) # Yardımcı fonksiyonu kullan
        logger.info(f"✅ Aylık istatistikler hesaplandı: {total_items} ürün, {total_revenue} TL.")
        return {"baslangic": start_date[:10], "siparis_sayisi": total_items, "gelir": total_revenue}
    except sqlite3.Error as e:
        logger.exception(f"❌ Veritabanı hatası (aylık istatistik): {e}")
        raise HTTPException(status_code=503, detail="Veritabanı hatası nedeniyle aylık istatistikler alınamadı.")
    except Exception as e:
        logger.exception(f"❌ Aylık istatistik hesaplanırken genel hata: {e}")
        raise HTTPException(status_code=500, detail="Aylık istatistikler hesaplanırken sunucu hatası oluştu.")

@app.get("/istatistik/yillik")
def get_yearly_stats_endpoint():
    logger.info("Yıllık (aylık kırılımda) istatistikler isteniyor...")
    try:
        monthly_item_counts = {}
        with get_db_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            # Durumu 'iptal' olmayan tüm siparişlerin zaman ve sepetini al
            cursor.execute("SELECT zaman, sepet FROM siparisler WHERE durum != 'iptal'")
            all_data = cursor.fetchall()

        for time_str, cart_json_str in all_data:
            if not cart_json_str or not time_str: continue
            try:
                # Zaman bilgisinden YYYY-MM anahtarını çıkar
                month_key = time_str[:7] # İlk 7 karakter (YYYY-MM)
                items_in_cart = json.loads(cart_json_str)
                if not isinstance(items_in_cart, list): continue
                # Ay toplamını hesapla
                month_total = sum(item.get("adet", 1) for item in items_in_cart if isinstance(item, dict) and isinstance(item.get("adet", 1), (int, float)) and item.get("adet", 1) > 0)
                monthly_item_counts[month_key] = monthly_item_counts.get(month_key, 0) + month_total
            except Exception as e:
                 logger.warning(f"⚠️ Yıllık ist.: Sepet işleme hatası ({e}): {cart_json_str[:50]}...")

        # Aylara göre sıralanmış dict döndür
        sorted_monthly_data = dict(sorted(monthly_item_counts.items()))
        logger.info(f"✅ Yıllık istatistikler hesaplandı ({len(sorted_monthly_data)} ay).")
        return sorted_monthly_data
    except sqlite3.Error as e:
        logger.exception(f"❌ Veritabanı hatası (yıllık istatistik): {e}")
        raise HTTPException(status_code=503, detail="Veritabanı hatası nedeniyle yıllık istatistikler alınamadı.")
    except Exception as e:
        logger.exception(f"❌ Yıllık istatistik hesaplanırken genel hata: {e}")
        raise HTTPException(status_code=500, detail="Yıllık istatistikler hesaplanırken sunucu hatası oluştu.")

# --- DÜZELTİLMİŞ FONKSİYON (Syntax Hatası Giderildi) ---
@app.get("/istatistik/filtreli")
def get_filtered_stats_endpoint(baslangic: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"), bitis: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$")):
    logger.info(f"Filtreli istatistik: {baslangic} - {bitis}")
    try:
        # Bitiş tarihini de kapsamak için sonraki günün başlangıcını al
        end_date_exclusive = (datetime.strptime(bitis, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = f"{baslangic} 00:00:00" # Başlangıç saatini ekle

        with get_db_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            # Belirtilen tarih aralığında ve durumu 'iptal' olmayanları al
            cursor.execute("""
                SELECT sepet FROM siparisler
                WHERE zaman >= ? AND zaman < ? AND durum != 'iptal'
            """, (start_date, end_date_exclusive))
            filtered_data = cursor.fetchall()

        total_items, total_revenue = calculate_statistics(filtered_data) # Yardımcı fonksiyonu kullan
        logger.info(f"✅ Filtreli istatistik: {total_items} ürün, {total_revenue} TL.")
        return {"aralik": f"{baslangic} → {bitis}", "siparis_sayisi": total_items, "gelir": total_revenue}
    except ValueError: # Tarih formatı hatası
        logger.error(f"❌ Filtreli: Geçersiz tarih formatı.")
        raise HTTPException(status_code=400, detail="Geçersiz tarih formatı.")
    except Exception as e: # Diğer tüm hatalar
        logger.exception(f"❌ Filtreli istatistik hatası: {e}")
        raise HTTPException(status_code=500,detail="Filtreli istatistik alınamadı.")

# --------------------------------------------------------------------------
# Sesli Yanıt Endpoint'i
# --------------------------------------------------------------------------
@app.post("/sesli-yanit")
async def generate_speech_endpoint(data: SesliYanitData):
    text=data.text; lang=data.language
    if not tts_client: logger.error(" TTS istemcisi yok."); raise HTTPException(503, "Ses hizmeti yok.")
    try:
        clean_text = temizle_emoji(text).strip();
        if not clean_text: raise HTTPException(400, "Seslendirilecek metin yok.")
        logger.info(f"🗣️ Ses isteği: Dil: {lang}, Metin: '{clean_text[:70]}...'")
        input_text = texttospeech.SynthesisInput(text=clean_text)
        voice = texttospeech.VoiceSelectionParams(language_code=lang, ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=1.0)
        response = tts_client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
        logger.info("✅ Ses başarıyla oluşturuldu.")
        return Response(content=response.audio_content, media_type="audio/mpeg")
    except google_exceptions.GoogleAPIError as e: logger.exception(f"❌ Google TTS API hatası: {e}"); raise HTTPException(503, f"Google ses hizmeti hatası: {e.message}")
    except HTTPException as he: raise he
    except Exception as e: logger.exception(f"❌ Ses üretme hatası: {e}"); raise HTTPException(500, "Ses üretilemedi.")

# --------------------------------------------------------------------------
# Admin Şifre Değiştirme Endpoint'i (Yorum Satırı Haline Getirildi)
# --------------------------------------------------------------------------
# @app.post("/admin/sifre-degistir")
# async def change_admin_password_endpoint(
#     # creds: AdminCredentialsUpdate, # Bu model tanımlı değildi, NameError'a neden oluyordu
#     auth: bool = Depends(check_admin)
# ):
#     """Admin kullanıcı adı/şifresini değiştirmek için endpoint (Sadece bilgilendirme)."""
#     logger.warning(f"ℹ️ Admin şifre değiştirme isteği alındı. "
#                    f"Gerçek değişiklik için .env dosyasını güncelleyip sunucuyu yeniden başlatın.")
#     return {
#         "mesaj": "Şifre değiştirme isteği alındı. Güvenlik nedeniyle, değişikliğin etkili olması için lütfen .env dosyasını manuel olarak güncelleyin ve uygulamayı yeniden başlatın."
#     }

# --------------------------------------------------------------------------
# Uygulama Kapatma Olayı
# --------------------------------------------------------------------------
@app.on_event("shutdown")
def shutdown_event():
    """Uygulama kapatılırken kaynakları temizler."""
    logger.info("🚪 Uygulama kapatılıyor...")
    global google_creds_path # Global olarak erişmek için
    if google_creds_path and os.path.exists(google_creds_path):
        try:
            os.remove(google_creds_path)
            logger.info("✅ Geçici Google kimlik bilgisi dosyası silindi.")
            google_creds_path = None # Yolu temizle
        except OSError as e:
            logger.error(f"❌ Geçici Google kimlik bilgisi dosyası silinemedi: {e}")
    logger.info("👋 Uygulama kapatıldı.")

# --------------------------------------------------------------------------
# Ana Çalıştırma Bloğu (Geliştirme için)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 FastAPI uygulaması geliştirme modunda başlatılıyor...")
    # Ortam değişkenlerinden host ve port al, yoksa varsayılan kullan
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")
    # Geliştirme için reload=True, üretimde False olmalı veya dışarıdan yönetilmeli
    uvicorn.run("main:app", host=host, port=port, reload=True, log_level="info")