from fastapi import (
    FastAPI, Request, Body, Query, HTTPException, status, Depends, WebSocket, WebSocketDisconnect, Response
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Set
from functools import lru_cache
from databases import Database
import os
import base64
import regex # type: ignore
import tempfile
import sqlite3
import json
import logging
import logging.config
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from google.cloud import texttospeech # type: ignore
from google.api_core import exceptions as google_exceptions # type: ignore
import asyncio
import secrets
from enum import Enum

# .env dosyasını yükle (özellikle yerel geliştirme için)
load_dotenv()

# Loglama Yapılandırması
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
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "default",
            "filename": "neso_backend.log",
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "root": {
            "level": "INFO",
            "handlers": ["console", "file"],
        },
        "uvicorn.error": {
            "level": "INFO",
            "handlers": ["console", "file"],
            "propagate": False,
        },
        "uvicorn.access": {
            "level": "INFO",
            "handlers": ["console", "file"],
            "propagate": False,
        },
        "app_logger": {
            "level": "INFO",
            "handlers": ["console", "file"],
            "propagate": False,
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("app_logger")

# Ortam Değişkenleri Doğrulama
class Settings(BaseSettings):
    OPENAI_API_KEY: str
    GOOGLE_APPLICATION_CREDENTIALS_BASE64: str
    ADMIN_USERNAME: str
    ADMIN_PASSWORD: str
    SECRET_KEY: str
    CORS_ALLOWED_ORIGINS: str = "http://localhost:3000,https://neso-guncel.vercel.app"
    DB_DATA_DIR: str = "."
    OPENAI_MODEL: str = "gpt-3.5-turbo"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

try:
    settings = Settings()
except ValueError as e:
    logger.critical(f"❌ Ortam değişkenleri eksik: {e}")
    raise SystemExit(f"Ortam değişkenleri eksik: {e}")

# Yardımcı Fonksiyonlar
def temizle_emoji(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    try:
        emoji_pattern = regex.compile(r"[\p{Emoji_Presentation}\p{Extended_Pictographic}]+", regex.UNICODE)
        return emoji_pattern.sub('', text)
    except Exception as e:
        logger.error(f"Emoji temizleme hatası: {e}")
        return text

# API İstemcileri Başlatma
openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
logger.info("✅ OpenAI istemcisi başlatıldı.")

google_creds_path: Optional[str] = None
tts_client: Optional[texttospeech.TextToSpeechClient] = None
try:
    decoded_creds = base64.b64decode(settings.GOOGLE_APPLICATION_CREDENTIALS_BASE64)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w+b') as tmp_file:
        tmp_file.write(decoded_creds)
        google_creds_path = tmp_file.name
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_creds_path
    tts_client = texttospeech.TextToSpeechClient()
    logger.info("✅ Google TTS istemcisi başlatıldı.")
except Exception as e:
    logger.warning(f"❌ Google TTS istemcisi başlatılamadı: {e}. Sesli yanıt özelliği devre dışı kalabilir.")

# FastAPI Uygulaması
app = FastAPI(
    title="Neso Sipariş Asistanı API",
    version="1.2.3",
    description="Fıstık Kafe için sipariş backend servisi."
)
security = HTTPBasic()

# Middleware Ayarları
allowed_origins_list = [origin.strip() for origin in settings.CORS_ALLOWED_ORIGINS.split(',')]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.SECRET_KEY,
    session_cookie="neso_session"
)
logger.info(f"CORS Middleware etkin: {allowed_origins_list}")
logger.info(f"Session Middleware etkinleştirildi.")


# Veritabanı Bağlantı Havuzu
DB_NAME = "neso.db"
MENU_DB_NAME = "neso_menu.db"
DB_PATH = os.path.join(settings.DB_DATA_DIR, DB_NAME)
MENU_DB_PATH = os.path.join(settings.DB_DATA_DIR, MENU_DB_NAME)
os.makedirs(settings.DB_DATA_DIR, exist_ok=True)

db = Database(f"sqlite:///{DB_PATH}")
menu_db = Database(f"sqlite:///{MENU_DB_PATH}")

@app.on_event("startup")
async def startup_event():
    await db.connect()
    await menu_db.connect()
    logger.info("✅ Veritabanı bağlantıları kuruldu.")
    await init_databases()
    await update_system_prompt()
    logger.info(f"🚀 FastAPI uygulaması başlatıldı. Sistem mesajı güncellendi.")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🚪 Uygulama kapatılıyor...")
    await db.disconnect()
    await menu_db.disconnect()
    if google_creds_path and os.path.exists(google_creds_path):
        try:
            os.remove(google_creds_path)
            logger.info("✅ Geçici Google kimlik bilgisi dosyası silindi.")
        except OSError as e:
            logger.error(f"❌ Google kimlik bilgisi dosyası silinemedi: {e}")
    logger.info("👋 Uygulama kapatıldı.")

# WebSocket Yönetimi
aktif_mutfak_websocketleri: Set[WebSocket] = set()
aktif_admin_websocketleri: Set[WebSocket] = set()

async def broadcast_message(connections: Set[WebSocket], message: Dict, ws_type_name: str):
    if not connections:
        logger.warning(f"⚠️ Broadcast: Bağlı {ws_type_name} istemcisi yok. Mesaj: {message.get('type')}")
        return

    message_json = json.dumps(message)
    tasks = []
    disconnected_ws = set()

    for ws in connections:
        try:
            tasks.append(ws.send_text(message_json))
        except RuntimeError:
            disconnected_ws.add(ws)
            logger.warning(f"⚠️ {ws_type_name} WS bağlantısı zaten kopuk, listeden kaldırılıyor: {ws.client}")

    for ws in disconnected_ws:
        connections.discard(ws)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    current_connections = list(connections)
    for i, result in enumerate(results):
        ws_to_check = current_connections[i]
        if isinstance(result, Exception):
            if ws_to_check in connections:
                 connections.discard(ws_to_check)
            logger.warning(f"⚠️ {ws_type_name} WS gönderme hatası, bağlantı kaldırılıyor ({ws_to_check.client}): {result}")


async def websocket_lifecycle(websocket: WebSocket, connections: Set[WebSocket], endpoint_name: str):
    await websocket.accept()
    connections.add(websocket)
    client_info = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "Bilinmeyen İstemci"
    logger.info(f"🔗 {endpoint_name} WS bağlandı: {client_info} (Toplam: {len(connections)})")
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                    logger.debug(f"🏓 {endpoint_name} WS: Ping alındı, Pong gönderildi: {client_info}")
                elif message.get("type") == "status_update" and endpoint_name == "Admin":
                     logger.info(f"Admin WS: Durum güncelleme mesajı alındı: {message.get('data')} from {client_info}")
            except json.JSONDecodeError:
                logger.warning(f"⚠️ {endpoint_name} WS: Geçersiz JSON formatında mesaj alındı: {data} from {client_info}")
            except Exception as e_inner:
                logger.error(f"❌ {endpoint_name} WS mesaj işleme hatası ({client_info}): {e_inner} - Mesaj: {data}")
    except WebSocketDisconnect as e:
        if e.code == 1012:
             logger.warning(f"🔌 {endpoint_name} WS beklenmedik şekilde kapandı (Kod {e.code} - Sunucu Yeniden Başlıyor Olabilir): {client_info}")
        else:
             logger.info(f"🔌 {endpoint_name} WS normal şekilde kapandı (Kod {e.code}): {client_info}")
    except Exception as e_outer:
        logger.error(f"❌ {endpoint_name} WS beklenmedik hata ({client_info}): {e_outer}")
    finally:
        if websocket in connections:
            connections.discard(websocket)
        logger.info(f"📉 {endpoint_name} WS kaldırıldı: {client_info} (Kalan: {len(connections)})")

@app.websocket("/ws/admin")
async def websocket_admin_endpoint(websocket: WebSocket):
    await websocket_lifecycle(websocket, aktif_admin_websocketleri, "Admin")

@app.websocket("/ws/mutfak")
async def websocket_mutfak_endpoint(websocket: WebSocket):
    await websocket_lifecycle(websocket, aktif_mutfak_websocketleri, "Mutfak/Masa")

# Veritabanı İşlemleri
async def update_table_status(masa_id: str, islem: str = "Erişim"):
    now = datetime.now()
    try:
        await db.execute("""
            INSERT INTO masa_durumlar (masa_id, son_erisim, aktif, son_islem)
            VALUES (:masa_id, :son_erisim, TRUE, :islem)
            ON CONFLICT(masa_id) DO UPDATE SET
                son_erisim = excluded.son_erisim,
                aktif = excluded.aktif,
                son_islem = excluded.son_islem
        """, {"masa_id": masa_id, "son_erisim": now.strftime("%Y-%m-%d %H:%M:%S"), "islem": islem})

        await broadcast_message(aktif_admin_websocketleri, {
            "type": "masa_durum",
            "data": {"masaId": masa_id, "sonErisim": now.isoformat(), "aktif": True, "sonIslem": islem}
        }, "Admin")
    except Exception as e:
        logger.error(f"❌ Masa durumu ({masa_id}) güncelleme hatası: {e}")

# Middleware
@app.middleware("http")
async def track_active_users(request: Request, call_next):
    masa_id_param = request.path_params.get("masaId")
    masa_id_query = request.query_params.get("masa_id")
    masa_id = masa_id_param or masa_id_query

    if masa_id:
        endpoint_name = "Bilinmeyen Endpoint"
        if request.scope.get("endpoint") and hasattr(request.scope["endpoint"], "__name__"):
            endpoint_name = request.scope["endpoint"].__name__
        else:
            endpoint_name = request.url.path
        await update_table_status(str(masa_id), f"{request.method} {endpoint_name}")
    try:
        response = await call_next(request)
        return response
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception(f"❌ HTTP Middleware genel hata ({request.url.path}): {e}")
        return Response("Sunucuda bir hata oluştu.", status_code=500, media_type="text/plain")


# Endpoint'ler
@app.get("/ping")
async def ping_endpoint():
    logger.info("📢 /ping endpoint'ine istek geldi!")
    return {"message": "Neso backend pong! Service is running."}

@app.get("/aktif-masalar")
async def get_active_tables_endpoint(auth: bool = Depends(lambda: True)):
    active_time_limit = datetime.now() - timedelta(minutes=15)
    try:
        tables = await db.fetch_all("""
            SELECT masa_id, son_erisim, aktif, son_islem FROM masa_durumlar
            WHERE son_erisim >= :limit AND aktif = TRUE ORDER BY son_erisim DESC
        """, {"limit": active_time_limit.strftime("%Y-%m-%d %H:%M:%S")})
        return {"tables": [dict(row) for row in tables]}
    except Exception as e:
        logger.error(f"❌ Aktif masalar alınamadı: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Veritabanı hatası nedeniyle aktif masalar alınamadı.")

# Admin Doğrulama
def check_admin(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username.encode('utf-8'), settings.ADMIN_USERNAME.encode('utf-8'))
    correct_password = secrets.compare_digest(credentials.password.encode('utf-8'), settings.ADMIN_PASSWORD.encode('utf-8'))
    if not (correct_username and correct_password):
        logger.warning(f"🔒 Başarısız admin giriş denemesi: {credentials.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Geçersiz kimlik bilgileri",
            headers={"WWW-Authenticate": "Basic"},
        )
    logger.info(f"🔑 Admin girişi başarılı: {credentials.username}")
    return True

# Pydantic Modelleri
class Durum(str, Enum):
    BEKLIYOR = "bekliyor"
    HAZIRLANIYOR = "hazirlaniyor"
    HAZIR = "hazir"
    IPTAL = "iptal"

class SepetItem(BaseModel):
    urun: str = Field(..., min_length=1, description="Sipariş edilen ürünün adı.")
    adet: int = Field(..., gt=0, description="Sipariş edilen ürünün adedi.")
    fiyat: float = Field(..., ge=0, description="Ürünün birim fiyatı.")
    kategori: Optional[str] = Field(None, description="Ürünün kategorisi (isteğe bağlı).")

class SiparisEkleData(BaseModel):
    masa: str = Field(..., min_length=1, description="Siparişin verildiği masa numarası/adı.")
    sepet: List[SepetItem] = Field(..., min_items=1, description="Sipariş edilen ürünlerin listesi.")
    istek: Optional[str] = Field(None, description="Müşterinin özel isteği.")
    yanit: Optional[str] = Field(None, description="AI tarafından üretilen yanıt (müşteri isteğine karşılık).")

class SiparisGuncelleData(BaseModel):
    masa: str = Field(..., min_length=1, description="Durumu güncellenecek siparişin masa numarası.")
    durum: Durum = Field(..., description="Siparişin yeni durumu.")
    id: Optional[int] = Field(None, description="Durumu güncellenecek siparişin ID'si (belirli bir sipariş için).")

class MenuEkleData(BaseModel):
    ad: str = Field(..., min_length=1, description="Menüye eklenecek ürünün adı.")
    fiyat: float = Field(..., gt=0, description="Ürünün fiyatı.")
    kategori: str = Field(..., min_length=1, description="Ürünün kategorisi.")

class AdminCredentialsUpdate(BaseModel):
    yeniKullaniciAdi: str = Field(..., min_length=1)
    yeniSifre: str = Field(..., min_length=8)

class SesliYanitData(BaseModel):
    text: str = Field(..., min_length=1, description="Sese dönüştürülecek metin.")
    language: str = Field(default="tr-TR", pattern=r"^[a-z]{2}-[A-Z]{2}$", description="Metnin dili (örn: tr-TR, en-US).")

# Sipariş Yönetimi
@app.post("/siparis-ekle", status_code=status.HTTP_201_CREATED)
async def add_order_endpoint(data: SiparisEkleData):
    masa = data.masa
    sepet = data.sepet
    istek = data.istek
    yanit = data.yanit
    zaman_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"📥 Yeni sipariş isteği alındı: Masa {masa}, {len(sepet)} çeşit ürün.")

    cached_price_dict = await get_menu_price_dict()
    cached_stock_dict = await get_menu_stock_dict()
    processed_sepet = []

    for item in sepet:
        urun_adi_lower = item.urun.lower().strip()
        if urun_adi_lower not in cached_stock_dict or cached_stock_dict[urun_adi_lower] == 0:
            logger.warning(f"⚠️ Stokta olmayan ürün sipariş edilmeye çalışıldı: '{item.urun}' (Masa: {masa})")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"'{item.urun}' adlı ürün şu anda stokta bulunmamaktadır.")
        item_dict = item.model_dump()
        item_dict['fiyat'] = cached_price_dict.get(urun_adi_lower, item.fiyat)
        if item_dict['fiyat'] == 0 and item.fiyat == 0 :
             logger.warning(f"⚠️ '{item.urun}' için fiyat bilgisi 0 olarak ayarlandı. Lütfen menüyü kontrol edin.")
        processed_sepet.append(item_dict)

    if not processed_sepet:
        logger.warning(f"⚠️ Sipariş verilemedi, sepetteki tüm ürünler stok dışı. (Masa: {masa})")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Sepetinizdeki ürünlerin hiçbiri şu anda mevcut değil.")

    istek_ozet = ", ".join([f"{p_item['adet']}x {p_item['urun']}" for p_item in processed_sepet])
    try:
        async with db.transaction():
            siparis_id = await db.fetch_val("""
                INSERT INTO siparisler (masa, istek, yanit, sepet, zaman, durum)
                VALUES (:masa, :istek, :yanit, :sepet, :zaman, 'bekliyor')
                RETURNING id
            """, {
                "masa": masa,
                "istek": istek or istek_ozet,
                "yanit": yanit,
                "sepet": json.dumps(processed_sepet, ensure_ascii=False),
                "zaman": zaman_str
            })
        if siparis_id is None:
            logger.error(f"❌ Sipariş ID'si alınamadı, veritabanı ekleme başarısız oldu. Masa: {masa}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Sipariş veritabanına kaydedilirken bir sorun oluştu.")

        siparis_bilgisi = {
            "type": "siparis",
            "data": {"id": siparis_id, "masa": masa, "istek": istek or istek_ozet, "sepet": processed_sepet, "zaman": zaman_str, "durum": "bekliyor"}
        }
        logger.info(f"📢 Broadcast: Yeni sipariş (ID: {siparis_id}, Masa: {masa}) tüm istemcilere gönderiliyor...")
        await broadcast_message(aktif_mutfak_websocketleri, siparis_bilgisi, "Mutfak/Masa")
        await broadcast_message(aktif_admin_websocketleri, siparis_bilgisi, "Admin")
        await update_table_status(masa, f"Sipariş verdi ({len(processed_sepet)} çeşit ürün)")
        logger.info(f"✅ Sipariş (ID: {siparis_id}) Masa: {masa} için başarıyla kaydedildi ve yayınlandı.")
        return {"mesaj": "Siparişiniz başarıyla alındı ve mutfağa iletildi.", "siparisId": siparis_id}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"❌ Sipariş ekleme sırasında beklenmedik hata (Masa: {masa}): {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Siparişiniz işlenirken bir sunucu hatası oluştu.")

@app.post("/siparis-guncelle")
async def update_order_status_endpoint(data: SiparisGuncelleData, auth: bool = Depends(check_admin)):
    logger.info(f"🔄 Sipariş durum güncelleme isteği: ID {data.id or 'Son'}, Masa {data.masa}, Durum {data.durum}")
    try:
        async with db.transaction():
            if data.id:
                query = "UPDATE siparisler SET durum = :durum WHERE id = :id RETURNING id, masa, durum, sepet, istek, zaman"
                values = {"durum": data.durum.value, "id": data.id}
            else:
                query = """
                    UPDATE siparisler SET durum = :durum
                    WHERE id = (
                        SELECT id FROM siparisler
                        WHERE masa = :masa AND durum NOT IN ('hazir', 'iptal')
                        ORDER BY id DESC LIMIT 1
                    )
                    RETURNING id, masa, durum, sepet, istek, zaman
                """
                values = {"durum": data.durum.value, "masa": data.masa}

            updated_order = await db.fetch_one(query, values)

        if updated_order:
            updated_order_dict = dict(updated_order)
            try:
                updated_order_dict['sepet'] = json.loads(updated_order_dict.get('sepet', '[]'))
            except json.JSONDecodeError:
                updated_order_dict['sepet'] = []
                logger.warning(f"⚠️ Sipariş güncelleme sonrası sepet JSON parse hatası: ID {updated_order_dict.get('id')}")

            notification = {
                "type": "durum",
                "data": {
                    "id": updated_order_dict.get("id"),
                    "masa": updated_order_dict.get("masa"),
                    "durum": updated_order_dict.get("durum"),
                    "sepet": updated_order_dict.get("sepet"),
                    "istek": updated_order_dict.get("istek"),
                    "zaman": datetime.now().isoformat()
                }
            }
            await broadcast_message(aktif_mutfak_websocketleri, notification, "Mutfak/Masa")
            await broadcast_message(aktif_admin_websocketleri, notification, "Admin")
            await update_table_status(updated_order_dict.get("masa", data.masa), f"Sipariş durumu güncellendi -> {data.durum.value}")
            logger.info(f"✅ Sipariş (ID: {updated_order_dict.get('id')}, Masa: {updated_order_dict.get('masa')}) durumu '{data.durum.value}' olarak güncellendi.")
            return {"message": f"Sipariş (ID: {updated_order_dict.get('id')}) durumu '{data.durum.value}' olarak güncellendi.", "data": updated_order_dict}
        else:
            logger.warning(f"⚠️ Güncellenecek sipariş bulunamadı: ID {data.id or 'Son'}, Masa {data.masa}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Güncellenecek uygun bir sipariş bulunamadı.")
    except Exception as e:
        logger.error(f"❌ Sipariş durumu güncelleme hatası (Masa: {data.masa}): {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Sipariş durumu güncellenirken bir hata oluştu.")

@app.get("/siparisler")
async def get_orders_endpoint(auth: bool = Depends(check_admin)):
    try:
        orders_raw = await db.fetch_all("SELECT id, masa, istek, yanit, sepet, zaman, durum FROM siparisler ORDER BY id DESC")
        orders_data = []
        for row in orders_raw:
            order_dict = dict(row)
            try:
                order_dict['sepet'] = json.loads(order_dict.get('sepet') or '[]')
            except json.JSONDecodeError:
                order_dict['sepet'] = []
                logger.warning(f"⚠️ Sipariş listelemede geçersiz sepet JSON: ID {order_dict.get('id')}")
            orders_data.append(order_dict)
        logger.info(f"📋 {len(orders_data)} adet sipariş listelendi.")
        return {"orders": orders_data}
    except Exception as e:
        logger.error(f"❌ Tüm siparişler alınırken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Siparişler veritabanından alınırken bir sorun oluştu.")

# Veritabanı Başlatma
async def init_db():
    logger.info(f"Ana veritabanı kontrol ediliyor: {DB_PATH}")
    try:
        async with db.transaction():
            await db.execute("""
                CREATE TABLE IF NOT EXISTS siparisler (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    masa TEXT NOT NULL,
                    istek TEXT,
                    yanit TEXT,
                    sepet TEXT,
                    zaman TEXT NOT NULL,
                    durum TEXT DEFAULT 'bekliyor' CHECK(durum IN ('bekliyor', 'hazirlaniyor', 'hazir', 'iptal'))
                )""")
            await db.execute("""
                CREATE TABLE IF NOT EXISTS masa_durumlar (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    masa_id TEXT UNIQUE NOT NULL,
                    son_erisim TIMESTAMP NOT NULL,
                    aktif BOOLEAN DEFAULT TRUE,
                    son_islem TEXT
                )""")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_siparisler_masa_zaman ON siparisler(masa, zaman DESC)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_masa_durumlar_erisim ON masa_durumlar(son_erisim DESC)")
        logger.info(f"✅ Ana veritabanı ({DB_PATH}) başarıyla doğrulandı/oluşturuldu.")
    except Exception as e:
        logger.critical(f"❌ Ana veritabanı başlatılırken kritik hata: {e}", exc_info=True)
        raise

async def init_menu_db():
    logger.info(f"Menü veritabanı kontrol ediliyor: {MENU_DB_PATH}")
    try:
        async with menu_db.transaction():
            await menu_db.execute("""
                CREATE TABLE IF NOT EXISTS kategoriler (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    isim TEXT UNIQUE NOT NULL COLLATE NOCASE
                )""")
            await menu_db.execute("""
                CREATE TABLE IF NOT EXISTS menu (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ad TEXT NOT NULL COLLATE NOCASE,
                    fiyat REAL NOT NULL CHECK(fiyat >= 0),
                    kategori_id INTEGER NOT NULL,
                    stok_durumu INTEGER DEFAULT 1,
                    FOREIGN KEY (kategori_id) REFERENCES kategoriler(id) ON DELETE CASCADE,
                    UNIQUE(ad, kategori_id)
                )""")
            await menu_db.execute("CREATE INDEX IF NOT EXISTS idx_menu_kategori_id ON menu(kategori_id)")
            await menu_db.execute("CREATE INDEX IF NOT EXISTS idx_menu_ad ON menu(ad)")
        logger.info(f"✅ Menü veritabanı ({MENU_DB_PATH}) başarıyla doğrulandı/oluşturuldu.")
    except Exception as e:
        logger.critical(f"❌ Menü veritabanı başlatılırken kritik hata: {e}", exc_info=True)
        raise

async def init_databases():
    await init_db()
    await init_menu_db()

# Menü Yönetimi
@lru_cache(maxsize=1)
async def get_menu_for_prompt_cached() -> str:
    logger.debug("get_menu_for_prompt_cached çağrıldı (cache'den veya yeniden)")
    try:
        if not menu_db.is_connected: await menu_db.connect()
        urunler_raw = await menu_db.fetch_all("""
            SELECT k.isim as kategori_isim, m.ad as urun_ad FROM menu m
            JOIN kategoriler k ON m.kategori_id = k.id
            WHERE m.stok_durumu = 1 ORDER BY k.isim, m.ad
        """)
        if not urunler_raw:
            logger.warning("Menü prompt için ürün bulunamadı.")
            return "Menüde şu anda görüntülenecek aktif ürün bulunmamaktadır."

        kategorili_menu: Dict[str, List[str]] = {}
        for row in urunler_raw:
            kategorili_menu.setdefault(row['kategori_isim'], []).append(row['urun_ad'])

        if not kategorili_menu:
            return "Menü bilgisi mevcut değil veya tüm ürünler stok dışı."

        menu_aciklama_list = []
        for kategori, urun_listesi in kategorili_menu.items():
            menu_aciklama_list.append(f"- {kategori}: {', '.join(urun_listesi)}")
        menu_aciklama = "\n".join(menu_aciklama_list)
        logger.info(f"Menü prompt için başarıyla oluşturuldu ({len(kategorili_menu)} kategori).")
        return "Mevcut menümüz aşağıdadır. Müşteriye sadece stokta olan ürünleri öner:\n" + menu_aciklama
    except Exception as e:
        logger.error(f"❌ Menü prompt oluşturma hatası: {e}", exc_info=True)
        return "Menü bilgisi şu anda alınamıyor. Lütfen daha sonra tekrar deneyin."

@lru_cache(maxsize=1)
async def get_menu_price_dict() -> Dict[str, float]:
    logger.debug("get_menu_price_dict çağrıldı (cache'den veya yeniden)")
    try:
        if not menu_db.is_connected: await menu_db.connect()
        prices_raw = await menu_db.fetch_all("SELECT ad, fiyat FROM menu")
        price_dict = {row['ad'].lower().strip(): float(row['fiyat']) for row in prices_raw}
        logger.info(f"Fiyat sözlüğü {len(price_dict)} ürün için oluşturuldu/alındı.")
        return price_dict
    except Exception as e:
        logger.error(f"❌ Fiyat sözlüğü oluşturma/alma hatası: {e}", exc_info=True)
        return {}

@lru_cache(maxsize=1)
async def get_menu_stock_dict() -> Dict[str, int]:
    logger.debug("get_menu_stock_dict çağrıldı (cache'den veya yeniden)")
    try:
        if not menu_db.is_connected: await menu_db.connect()
        stocks_raw = await menu_db.fetch_all("SELECT ad, stok_durumu FROM menu")
        stock_dict = {row['ad'].lower().strip(): int(row['stok_durumu']) for row in stocks_raw}
        logger.info(f"Stok sözlüğü {len(stock_dict)} ürün için oluşturuldu/alındı.")
        return stock_dict
    except Exception as e:
        logger.error(f"❌ Stok sözlüğü oluşturma/alma hatası: {e}", exc_info=True)
        return {}

SISTEM_MESAJI_ICERIK_TEMPLATE = (
    "Sen, Gaziantep'teki Fıstık Kafe için Neso adında bir sipariş asistanısın. "
    "Görevin, müşterilerin taleplerini nazikçe ve doğru bir şekilde anlayıp, yalnızca aşağıda listelenen ve stokta bulunan menüdeki ürünlerle eşleştirerek siparişlerini JSON formatında hazırlamaktır. "
    "Müşteriye her zaman kibar, yardımsever ve profesyonel bir Türkçe ile hitap et. "
    "Eğer bir isteği tam olarak anlayamazsan, netleştirmek için ek sorular sor. "
    "Siparişi onaylamadan önce müşteriye sipariş özetini ve toplam tutarı bildir. "
    "Sipariş tamamlandığında ve müşteri onayladığında 'Afiyet olsun!' gibi olumlu bir ifade kullan. "
    "Menü dışı veya stokta olmayan bir ürün istenirse, nazikçe olmadığını belirt ve alternatifler sunmaya çalış. "
    "Müşteriye fiyat bilgisi verirken, ürünlerin güncel fiyatlarını kullan. İşte şu anki menümüz ve stok durumları:\n\n{menu_prompt_data}"
    "\n\nSiparişleri şu formatta çıkar: {{\"sepet\": [{{\"urun\": \"Ürün Adı\", \"adet\": Miktar, \"fiyat\": BirimFiyat, \"kategori\": \"KategoriAdı\"}}], \"toplam_tutar\": ToplamTutar, \"musteri_notu\": \"Müşterinin özel isteği\"}}"
)
SYSTEM_PROMPT: Optional[Dict[str, str]] = None

async def update_system_prompt():
    global SYSTEM_PROMPT
    logger.info("🔄 Sistem mesajı (menü bilgisi) güncelleniyor...")
    try:
        get_menu_for_prompt_cached.cache_clear()
        get_menu_price_dict.cache_clear()
        get_menu_stock_dict.cache_clear()

        menu_data_for_prompt = await get_menu_for_prompt_cached()
        current_system_content = SISTEM_MESAJI_ICERIK_TEMPLATE.format(menu_prompt_data=menu_data_for_prompt)
        SYSTEM_PROMPT = {"role": "system", "content": current_system_content}
        logger.info("✅ Sistem mesajı başarıyla güncellendi.")
    except Exception as e:
        logger.error(f"❌ Sistem mesajı güncellenirken hata oluştu: {e}", exc_info=True)
        if SYSTEM_PROMPT is None:
            default_menu_info = "Menü bilgisi şu anda yüklenemedi."
            SYSTEM_PROMPT = {"role": "system", "content": SISTEM_MESAJI_ICERIK_TEMPLATE.format(menu_prompt_data=default_menu_info)}
            logger.warning("Fallback sistem mesajı kullanılıyor.")

@app.get("/menu")
async def get_full_menu_endpoint():
    logger.info("Tam menü isteniyor (/menu)...")
    try:
        full_menu_data = []
        kategoriler_raw = await menu_db.fetch_all("SELECT id, isim FROM kategoriler ORDER BY isim")
        for kat_row in kategoriler_raw:
            urunler_raw = await menu_db.fetch_all(
                "SELECT ad, fiyat, stok_durumu FROM menu WHERE kategori_id = :id ORDER BY ad",
                {"id": kat_row['id']}
            )
            full_menu_data.append({
                "kategori": kat_row['isim'],
                "urunler": [dict(urun) for urun in urunler_raw]
            })
        logger.info(f"✅ Tam menü başarıyla alındı ({len(full_menu_data)} kategori).")
        return {"menu": full_menu_data}
    except Exception as e:
        logger.error(f"❌ Tam menü alınırken veritabanı hatası: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Menü bilgileri alınırken bir sorun oluştu.")

@app.post("/menu/ekle", status_code=status.HTTP_201_CREATED)
async def add_menu_item_endpoint(item_data: MenuEkleData, auth: bool = Depends(check_admin)):
    logger.info(f"📝 Menüye yeni ürün ekleme isteği: {item_data.ad} ({item_data.kategori})")
    try:
        async with menu_db.transaction():
            await menu_db.execute("INSERT OR IGNORE INTO kategoriler (isim) VALUES (:isim)", {"isim": item_data.kategori})
            category_id_row = await menu_db.fetch_one("SELECT id FROM kategoriler WHERE isim = :isim", {"isim": item_data.kategori})
            if not category_id_row:
                logger.error(f"Menü ekleme: Kategori '{item_data.kategori}' oluşturulamadı veya bulunamadı.")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Kategori oluşturulurken bir sorun oluştu.")
            category_id = category_id_row['id']
            try:
                item_id = await menu_db.fetch_val("""
                    INSERT INTO menu (ad, fiyat, kategori_id, stok_durumu)
                    VALUES (:ad, :fiyat, :kategori_id, 1)
                    RETURNING id
                """, {"ad": item_data.ad, "fiyat": item_data.fiyat, "kategori_id": category_id})
            except sqlite3.IntegrityError as ie:
                 if "UNIQUE constraint failed" in str(ie):
                     logger.warning(f"Menü ekleme başarısız: '{item_data.ad}' adlı ürün '{item_data.kategori}' kategorisinde zaten mevcut.")
                     raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"'{item_data.ad}' adlı ürün bu kategoride zaten mevcut.")
                 raise
        await update_system_prompt()
        logger.info(f"✅ '{item_data.ad}' menüye başarıyla eklendi (ID: {item_id}). Sistem mesajı güncellendi.")
        return {"mesaj": f"'{item_data.ad}' ürünü menüye başarıyla eklendi.", "itemId": item_id}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"❌ Menüye ürün eklenirken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Menüye ürün eklenirken bir sunucu hatası oluştu.")

@app.delete("/menu/sil")
async def delete_menu_item_endpoint(urun_adi: str = Query(..., min_length=1, description="Silinecek ürünün tam adı."), auth: bool = Depends(check_admin)):
    logger.info(f"🗑️ Menüden ürün silme isteği: {urun_adi}")
    try:
        async with menu_db.transaction():
            item_to_delete = await menu_db.fetch_one("SELECT id FROM menu WHERE ad = :ad", {"ad": urun_adi})
            if not item_to_delete:
                logger.warning(f"Silinecek ürün bulunamadı: '{urun_adi}'")
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{urun_adi}' adlı ürün menüde bulunamadı.")
            rows_affected = await menu_db.execute("DELETE FROM menu WHERE ad = :ad", {"ad": urun_adi})
        if rows_affected and rows_affected > 0 :
            await update_system_prompt()
            logger.info(f"✅ '{urun_adi}' menüden başarıyla silindi. Sistem mesajı güncellendi.")
            return {"mesaj": f"'{urun_adi}' ürünü menüden başarıyla silindi."}
        else:
            logger.warning(f"Silme işlemi başarısız oldu veya ürün bulunamadı (rows_affected=0): '{urun_adi}'")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{urun_adi}' adlı ürün silinemedi veya bulunamadı.")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"❌ Menüden ürün silinirken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Menüden ürün silinirken bir sunucu hatası oluştu.")

# AI Yanıt
@app.post("/yanitla")
async def handle_message_endpoint(request: Request, data: dict = Body(...)):
    user_message = data.get("text", "").strip()
    table_id = data.get("masa", "bilinmiyor")
    session_id = request.session.get("session_id")

    if not session_id:
        session_id = secrets.token_hex(16)
        request.session["session_id"] = session_id
        request.session["chat_history"] = []
        logger.info(f"Yeni session başlatıldı: {session_id} Masa: {table_id}")

    chat_history = request.session.get("chat_history", [])
    logger.info(f"💬 AI Yanıt isteği: Masa '{table_id}', Kullanıcı Mesajı: '{user_message}'")

    if not user_message:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Mesaj boş olamaz.")
    if SYSTEM_PROMPT is None:
        logger.error("❌ AI Yanıt: Sistem promptu yüklenmemiş!")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="AI asistanı şu anda hazır değil, sistem ayarları eksik.")

    try:
        messages_for_openai = [SYSTEM_PROMPT] + chat_history + [{"role": "user", "content": user_message}]
        response = openai_client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages_for_openai, # type: ignore
            temperature=0.5,
            max_tokens=350,
        )
        ai_reply = response.choices[0].message.content
        if ai_reply is None:
            ai_reply = "Üzgünüm, şu anda bir yanıt üretemiyorum."
            logger.warning("OpenAI'den boş yanıt (None) alındı.")
        else:
            ai_reply = ai_reply.strip()

        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": ai_reply})
        request.session["chat_history"] = chat_history[-10:]

        logger.info(f"🤖 AI Yanıtı (Masa: {table_id}): '{ai_reply}'")
        return {"reply": ai_reply, "sessionId": session_id}
    except OpenAIError as e:
        logger.error(f"❌ OpenAI API ile iletişim hatası (Masa: {table_id}): {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"AI servisinden yanıt alınırken bir sorun oluştu: {e}")
    except Exception as e:
        logger.error(f"❌ AI yanıt endpoint'inde beklenmedik hata (Masa: {table_id}): {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Mesajınız işlenirken beklenmedik bir sunucu hatası oluştu.")

# İstatistikler
def calculate_statistics(orders_data: List[dict]) -> tuple[int, float, int]:
    total_orders_count = len(orders_data)
    total_items_sold = 0
    total_revenue = 0.0

    for order_row in orders_data:
        try:
            sepet_items_str = order_row['sepet'] # Düzeltildi: .get() yerine []
            if isinstance(sepet_items_str, str):
                items = json.loads(sepet_items_str or '[]')
            elif isinstance(sepet_items_str, list):
                items = sepet_items_str
            else:
                items = []
                logger.warning(f"⚠️ İstatistik: Beklenmeyen sepet formatı: {type(sepet_items_str)} - Sipariş ID: {order_row.get('id')}") # .get() burada kullanılabilir çünkü order_row bir dict

            for item in items:
                if isinstance(item, dict):
                    adet = item.get("adet", 0)
                    fiyat = item.get("fiyat", 0.0)
                    if isinstance(adet, (int, float)) and isinstance(fiyat, (int, float)):
                        total_items_sold += adet
                        total_revenue += adet * fiyat
                    else:
                        logger.warning(f"⚠️ İstatistik: Sepet öğesinde geçersiz adet/fiyat: {item} - Sipariş ID: {order_row.get('id')}")
                else:
                     logger.warning(f"⚠️ İstatistik: Sepet öğesi dict değil: {item} - Sipariş ID: {order_row.get('id')}")
        except json.JSONDecodeError:
            logger.warning(f"⚠️ İstatistik: Sepet JSON parse hatası. Sipariş ID: {order_row.get('id')}, Sepet Verisi (ilk 50 krkt): {str(order_row.get('sepet'))[:50]}")
        except KeyError: # Eğer 'sepet' anahtarı order_row'da yoksa
             logger.warning(f"⚠️ İstatistik: 'sepet' anahtarı bulunamadı. Sipariş ID: {order_row.get('id')}")
        except Exception as e:
            logger.error(f"⚠️ İstatistik hesaplama sırasında beklenmedik hata: {e} - Sipariş ID: {order_row.get('id')}", exc_info=True)

    return total_orders_count, total_items_sold, round(total_revenue, 2)


@app.get("/istatistik/en-cok-satilan")
async def get_popular_items_endpoint(limit: int = Query(5, ge=1, le=20), auth: bool = Depends(check_admin)):
    logger.info(f"📊 En çok satılan {limit} ürün istatistiği isteniyor.")
    item_counts: Dict[str, int] = {}
    try:
        orders_raw = await db.fetch_all("SELECT sepet FROM siparisler WHERE durum != 'iptal'")
        for row_record in orders_raw: # row_record bir databases.Record objesi
            row = dict(row_record) # Record'u dictionary'ye çevir
            try:
                sepet_items_str = row.get('sepet') # .get() şimdi kullanılabilir
                if isinstance(sepet_items_str, str):
                    items = json.loads(sepet_items_str or '[]')
                elif isinstance(sepet_items_str, list):
                    items = sepet_items_str
                else:
                    items = []
                    logger.warning(f"Popüler ürünler: Beklenmeyen sepet formatı: {type(sepet_items_str)} - Satır: {row}")

                for item in items:
                    if isinstance(item, dict):
                        item_name = item.get("urun")
                        quantity = item.get("adet", 0)
                        if item_name and isinstance(quantity, (int, float)) and quantity > 0:
                            item_counts[item_name] = item_counts.get(item_name, 0) + quantity
            except json.JSONDecodeError:
                logger.warning(f"⚠️ Popüler ürünler: Sepet JSON parse hatası. Veri (ilk 50): {str(sepet_items_str)[:50]}")
            except Exception as e_inner:
                logger.error(f"⚠️ Popüler ürünler: Sepet işleme sırasında beklenmedik iç hata: {e_inner} - Satır: {row}", exc_info=True)

        sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        logger.info(f"✅ En çok satılan {len(sorted_items)} ürün bulundu.")
        return [{"urun": item, "adet": count} for item, count in sorted_items]
    except Exception as e_outer:
        logger.error(f"❌ Popüler ürünler istatistiği alınırken genel hata: {e_outer}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Popüler ürün istatistikleri alınamadı.")

async def get_stats_for_period(start_date_str: str, end_date_str: Optional[str] = None) -> dict:
    query = "SELECT id, sepet, zaman FROM siparisler WHERE durum != 'iptal' AND zaman >= :start"
    values: Dict[str, any] = {"start": start_date_str}
    if end_date_str:
        end_date_dt = datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1)
        query += " AND zaman < :end_dt_str"
        values["end_dt_str"] = end_date_dt.strftime("%Y-%m-%d %H:%M:%S")

    orders_for_stats_records = await db.fetch_all(query, values)
    # calculate_statistics fonksiyonu dict listesi beklediği için burada dönüştürüyoruz
    orders_list = [dict(record) for record in orders_for_stats_records]
    total_orders_count, total_items_sold, total_revenue = calculate_statistics(orders_list)
    return {
        "siparis_sayisi": total_orders_count,
        "satilan_urun_adedi": total_items_sold,
        "toplam_gelir": total_revenue,
        "veri_adedi": len(orders_list)
    }

@app.get("/istatistik/gunluk")
async def get_daily_stats_endpoint(tarih: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="Belirli bir günün istatistiği (YYYY-AA-GG). Boş bırakılırsa bugünün istatistiği."), auth: bool = Depends(check_admin)):
    target_date_str = tarih if tarih else datetime.now().strftime("%Y-%m-%d")
    logger.info(f"📊 Günlük istatistik isteniyor: {target_date_str}")
    try:
        stats = await get_stats_for_period(target_date_str, target_date_str)
        logger.info(f"✅ Günlük istatistik ({target_date_str}) hesaplandı.")
        return {"tarih": target_date_str, **stats}
    except ValueError:
        logger.error(f"❌ Günlük istatistik: Geçersiz tarih formatı: {tarih}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Geçersiz tarih formatı. Lütfen YYYY-AA-GG formatını kullanın.")
    except Exception as e:
        logger.error(f"❌ Günlük istatistik ({target_date_str}) alınırken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Günlük istatistikler alınamadı.")

@app.get("/istatistik/aylik")
async def get_monthly_stats_endpoint(yil: Optional[int] = Query(None, ge=2000, le=datetime.now().year + 1), ay: Optional[int] = Query(None, ge=1, le=12), auth: bool = Depends(check_admin)):
    now = datetime.now()
    target_year = yil if yil else now.year
    target_month = ay if ay else now.month

    try:
        start_date = datetime(target_year, target_month, 1)
        if target_month == 12:
            end_date = datetime(target_year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(target_year, target_month + 1, 1) - timedelta(days=1)

        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        logger.info(f"📊 Aylık istatistik isteniyor: {target_year}-{target_month:02d} ({start_date_str} - {end_date_str})")
        stats = await get_stats_for_period(start_date_str, end_date_str)
        logger.info(f"✅ Aylık istatistik ({target_year}-{target_month:02d}) hesaplandı.")
        return {"yil": target_year, "ay": target_month, **stats}
    except ValueError as ve:
        logger.error(f"❌ Aylık istatistik: Geçersiz yıl/ay değeri: Yıl={yil}, Ay={ay}. Hata: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Geçersiz yıl veya ay değeri. {ve}")
    except Exception as e:
        logger.error(f"❌ Aylık istatistik ({target_year}-{target_month:02d}) alınırken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Aylık istatistikler alınamadı.")

@app.get("/istatistik/yillik-aylik-kirilim")
async def get_yearly_stats_by_month_endpoint(yil: Optional[int] = Query(None, ge=2000, le=datetime.now().year + 1), auth: bool = Depends(check_admin)):
    target_year = yil if yil else datetime.now().year
    logger.info(f"📊 Yıllık ({target_year}) aylık kırılımlı istatistik isteniyor (/istatistik/yillik-aylik-kirilim).") # Loga yol eklendi
    try:
        start_of_year = f"{target_year}-01-01 00:00:00"
        end_of_year = f"{target_year+1}-01-01 00:00:00"

        query = """
            SELECT id, sepet, zaman FROM siparisler
            WHERE durum != 'iptal' AND zaman >= :start AND zaman < :end
            ORDER BY zaman ASC
        """
        orders_raw_records = await db.fetch_all(query, {"start": start_of_year, "end": end_of_year})
        
        monthly_stats: Dict[str, Dict[str, any]] = {}
        orders_as_dicts = [dict(record) for record in orders_raw_records] # Hepsini başta dict'e çevir

        for row_dict in orders_as_dicts: # Artık row_dict bir dictionary
            try:
                order_time_str = row_dict.get('zaman', '')
                order_datetime = None
                possible_formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]
                for fmt in possible_formats:
                    try:
                        order_datetime = datetime.strptime(order_time_str.split('.')[0], fmt)
                        break
                    except ValueError:
                        continue

                if not order_datetime:
                    logger.warning(f"Yıllık istatistik: Geçersiz zaman formatı: {order_time_str} Sipariş ID: {row_dict.get('id')}")
                    continue

                month_key = order_datetime.strftime("%Y-%m")

                if month_key not in monthly_stats:
                    monthly_stats[month_key] = {"siparis_sayisi": 0, "satilan_urun_adedi": 0, "toplam_gelir": 0.0}

                sepet_items_str = row_dict.get('sepet') # .get() burada kullanılabilir
                if isinstance(sepet_items_str, str): items = json.loads(sepet_items_str or '[]')
                elif isinstance(sepet_items_str, list): items = sepet_items_str
                else: items = []

                current_order_item_count = 0
                current_order_revenue = 0.0

                for item in items:
                    if isinstance(item, dict):
                        adet = item.get("adet", 0)
                        fiyat = item.get("fiyat", 0.0)
                        if isinstance(adet, (int,float)) and isinstance(fiyat, (int,float)):
                            current_order_item_count += adet
                            current_order_revenue += adet * fiyat
                
                monthly_stats[month_key]["siparis_sayisi"] += 1
                monthly_stats[month_key]["satilan_urun_adedi"] += current_order_item_count
                monthly_stats[month_key]["toplam_gelir"] = round(monthly_stats[month_key]["toplam_gelir"] + current_order_revenue, 2)

            except json.JSONDecodeError:
                logger.warning(f"⚠️ Yıllık istatistik JSON parse hatası. Sipariş ID: {row_dict.get('id')}, Veri (ilk 50): {str(row_dict.get('sepet'))[:50]}")
            except Exception as e_inner:
                logger.error(f"⚠️ Yıllık istatistik (aylık kırılım) iç döngü hatası: {e_inner} - Sipariş ID: {row_dict.get('id')}", exc_info=True)

        logger.info(f"✅ Yıllık ({target_year}) aylık kırılımlı istatistik hesaplandı ({len(monthly_stats)} ay).")
        return {"yil": target_year, "aylik_kirilim": dict(sorted(monthly_stats.items()))}
    except Exception as e:
        logger.error(f"❌ Yıllık ({target_year}) aylık kırılımlı istatistik alınırken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"{target_year} yılı için aylık kırılımlı istatistikler alınamadı.")

@app.get("/istatistik/filtreli")
async def get_filtered_stats_endpoint(
    baslangic: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Başlangıç tarihi (YYYY-AA-GG)"),
    bitis: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Bitiş tarihi (YYYY-AA-GG)"),
    auth: bool = Depends(check_admin)
):
    logger.info(f"📊 Filtreli istatistik isteniyor: {baslangic} - {bitis}")
    try:
        start_dt = datetime.strptime(baslangic, "%Y-%m-%d")
        end_dt = datetime.strptime(bitis, "%Y-%m-%d")
        if start_dt > end_dt:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Başlangıç tarihi bitiş tarihinden sonra olamaz.")

        stats = await get_stats_for_period(baslangic, bitis)
        logger.info(f"✅ Filtreli istatistik ({baslangic} - {bitis}) hesaplandı.")
        return {"aralik": f"{baslangic} → {bitis}", **stats}
    except ValueError:
        logger.error(f"❌ Filtreli istatistik: Geçersiz tarih formatı. Başlangıç: {baslangic}, Bitiş: {bitis}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Geçersiz tarih formatı. Lütfen YYYY-AA-GG formatını kullanın.")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"❌ Filtreli istatistik ({baslangic} - {bitis}) alınırken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Belirtilen aralık için istatistikler alınamadı.")

# Sesli Yanıt
SUPPORTED_LANGUAGES = {"tr-TR", "en-US", "en-GB", "fr-FR", "de-DE"}

@app.post("/sesli-yanit")
async def generate_speech_endpoint(data: SesliYanitData):
    if not tts_client:
        logger.error("❌ Sesli yanıt: TTS istemcisi başlatılmamış.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Sesli yanıt servisi şu anda kullanılamıyor (TTS istemcisi eksik).")
    if data.language not in SUPPORTED_LANGUAGES:
        logger.warning(f"⚠️ Sesli yanıt: Desteklenmeyen dil kodu: {data.language}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Desteklenmeyen dil: {data.language}. Desteklenen diller: {', '.join(SUPPORTED_LANGUAGES)}")

    logger.info(f"🎤 Sesli yanıt isteği: Dil '{data.language}', Metin (ilk 30kr): '{data.text[:30]}...'")
    try:
        cleaned_text = temizle_emoji(data.text)
        if not cleaned_text.strip():
            logger.warning("⚠️ Sesli yanıt: Boş veya sadece emojiden oluşan metin.")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Sese dönüştürülecek geçerli bir metin bulunamadı.")

        synthesis_input = texttospeech.SynthesisInput(text=cleaned_text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=data.language,
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0
        )
        response_tts = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        logger.info(f"✅ Sesli yanıt başarıyla oluşturuldu (Dil: {data.language}).")
        return Response(content=response_tts.audio_content, media_type="audio/mpeg")
    except google_exceptions.GoogleAPIError as e:
        logger.error(f"❌ Google TTS API hatası: {e}", exc_info=True)
        detail_message = f"Google TTS servisinden ses üretilirken bir hata oluştu: {e.message if hasattr(e, 'message') else str(e)}"
        if "API key not valid" in str(e) or "permission" in str(e).lower():
            detail_message = "Google TTS servisi için kimlik bilgileri geçersiz veya yetki sorunu var."
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail_message)
    except Exception as e:
        logger.error(f"❌ Sesli yanıt endpoint'inde beklenmedik hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Sesli yanıt oluşturulurken beklenmedik bir sunucu hatası oluştu.")

@app.post("/admin/sifre-degistir")
async def change_admin_password_endpoint(creds: AdminCredentialsUpdate, auth: bool = Depends(check_admin)):
    logger.warning(f"ℹ️ Admin şifre/kullanıcı adı değiştirme endpoint'i çağrıldı (Kullanıcı: {creds.yeniKullaniciAdi}). Bu işlem için .env dosyasının manuel güncellenmesi gerekmektedir.")
    return {
        "mesaj": "Admin kullanıcı adı ve şifresini değiştirmek için lütfen sunucudaki .env dosyasını güncelleyin ve uygulamayı yeniden başlatın. Bu endpoint sadece bir hatırlatmadır ve aktif bir değişiklik yapmaz."
    }

if __name__ == "__main__":
    import uvicorn
    host_ip = os.getenv("HOST", "127.0.0.1")
    port_num = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 FastAPI uygulaması {host_ip}:{port_num} adresinde başlatılıyor (yerel geliştirme modu)...")
    uvicorn.run("main:app", host=host_ip, port=port_num, reload=True)