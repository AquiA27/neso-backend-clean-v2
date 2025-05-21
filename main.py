# main.py
from fastapi import (
    FastAPI, Request, Path, Body, Query, HTTPException, status, Depends, WebSocket, WebSocketDisconnect, Response
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm # HTTPBasic yerine OAuth2
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Set, Union, Any
from async_lru import alru_cache
from databases import Database
import os
import base64
import regex # type: ignore
import tempfile
import sqlite3 # Veri taşıma scriptleri veya fallback dışında direkt kullanılmayacak
import json
import logging
import logging.config
from datetime import datetime, timedelta, timezone as dt_timezone # timezone'u dt_timezone olarak import ettim karışmaması için
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from google.cloud import texttospeech # type: ignore
from google.api_core import exceptions as google_exceptions # type: ignore
import asyncio
import secrets
from enum import Enum

# JWT ve Şifreleme için eklenenler
from jose import JWTError, jwt
from passlib.context import CryptContext

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
            "filename": "neso_backend.log", # Log dosyası DB_DATA_DIR'dan bağımsız
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
            "level": "INFO", # Geliştirme sırasında DEBUG yapabilirsiniz
            "handlers": ["console", "file"],
            "propagate": False,
        },
    },
}
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("app_logger")

# --- Kullanıcı Rolleri ---
class KullaniciRol(str, Enum):
    ADMIN = "admin"
    KASIYER = "kasiyer"
    BARISTA = "barista"
    MUTFAK_PERSONELI = "mutfak_personeli"

# Ortam Değişkenleri Doğrulama ve Ayarlar
class Settings(BaseSettings):
    OPENAI_API_KEY: str
    GOOGLE_APPLICATION_CREDENTIALS_BASE64: str
    SECRET_KEY: str # JWT için de kullanılacak
    CORS_ALLOWED_ORIGINS: str = "http://localhost:3000,https://neso-guncel.vercel.app"
    DB_DATA_DIR: str = "." # Render gibi ortamlarda burası kalıcı disk yolu olmalı - PostgreSQL için doğrudan kullanılmayacak
    OPENAI_MODEL: str = "gpt-3.5-turbo"

    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440 # 1 gün

    DEFAULT_ADMIN_USERNAME: str = "admin"
    DEFAULT_ADMIN_PASSWORD: str = "ChangeThisDefaultPassword123!"

    # YENİ: PostgreSQL bağlantı adresi için ortam değişkeni (opsiyonel, doğrudan aşağıda kullanılacak)
    # DATABASE_URL: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

try:
    settings = Settings()
    logger.info(f"Ayarlar yüklendi.")
    if settings.DB_DATA_DIR == ".": # Bu uyarı artık daha az kritik ama loglar için hala geçerli olabilir.
        logger.warning("DB_DATA_DIR varsayılan '.' olarak ayarlı. "
                       "Render gibi bir ortamda kalıcı disk yolu (örn: /var/data/logs) log dosyaları için belirtilebilir.")
except ValueError as e:
    logger.critical(f"❌ Ortam değişkenleri eksik veya hatalı: {e}")
    raise SystemExit(f"Ortam değişkenleri eksik veya hatalı: {e}")

# Şifreleme ve OAuth2
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Yardımcı Fonksiyonlar
def temizle_emoji(text: Optional[str]) -> str:
    if not isinstance(text, str): return ""
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
    if settings.GOOGLE_APPLICATION_CREDENTIALS_BASE64:
        decoded_creds = base64.b64decode(settings.GOOGLE_APPLICATION_CREDENTIALS_BASE64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w+b') as tmp_file:
            tmp_file.write(decoded_creds)
            google_creds_path = tmp_file.name
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_creds_path
        tts_client = texttospeech.TextToSpeechClient()
        logger.info("✅ Google TTS istemcisi başlatıldı.")
    else:
        logger.warning("GOOGLE_APPLICATION_CREDENTIALS_BASE64 ortam değişkeni ayarlanmamış. TTS devre dışı.")
except Exception as e:
    logger.warning(f"❌ Google TTS istemcisi başlatılamadı: {e}. Sesli yanıt özelliği devre dışı kalabilir.")

# FastAPI Uygulaması
app = FastAPI(
    title="Neso Sipariş Asistanı API",
    version="1.3.3", # Versiyon bilgisi güncel tutulabilir
    description="Fıstık Kafe için sipariş backend servisi."
)

# Middleware Ayarları
allowed_origins_list = [origin.strip() for origin in settings.CORS_ALLOWED_ORIGINS.split(',')]
logger.info(f"📢 CORS Yapılandırması - Allowed Origins List: {allowed_origins_list} (Raw string: '{settings.CORS_ALLOWED_ORIGINS}')")
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
logger.info(f"Session Middleware etkinleştirildi.")


# --- YENİ VERİTABANI BAĞLANTI YAPILANDIRMASI ---
# Ortam değişkeninden DATABASE_URL'i alıyoruz.
# Render'da bu `postgresql://user:pass@host:port/dbname` formatında olacak.
# Eğer lokalde .env dosyasında veya sistemde DATABASE_URL tanımlı değilse,
# geliştirme için geçici bir SQLite veritabanı kullanılabilir (fallback).
DATABASE_CONNECTION_STRING = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(settings.DB_DATA_DIR, 'neso_dev_fallback.db')}")

# Bağlantı adresinin şifre kısmını loglamamak için kısaltıyoruz
log_db_url = DATABASE_CONNECTION_STRING
if "@" in log_db_url and ":" in log_db_url.split("@")[0]:
    user_pass_part = log_db_url.split("://")[1].split("@")[0]
    host_part = log_db_url.split("@")[1]
    log_db_url = f"{log_db_url.split('://')[0]}://{user_pass_part.split(':')[0]}:********@{host_part}"
logger.info(f"Ana veritabanı bağlantı adresi kullanılıyor: {log_db_url}")

db = Database(DATABASE_CONNECTION_STRING)
# Menü verileri de aynı PostgreSQL veritabanında olacağı için aynı bağlantı string'ini kullanıyoruz.
# Ayrı bir PostgreSQL instance'ı kullanılacaksa, MENU_DATABASE_URL ortam değişkeni tanımlanmalı.
MENU_DATABASE_CONNECTION_STRING = os.getenv("MENU_DATABASE_URL", DATABASE_CONNECTION_STRING)
if MENU_DATABASE_CONNECTION_STRING != DATABASE_CONNECTION_STRING:
    log_menu_db_url = MENU_DATABASE_CONNECTION_STRING
    if "@" in log_menu_db_url and ":" in log_menu_db_url.split("@")[0]:
        user_pass_part_menu = log_menu_db_url.split("://")[1].split("@")[0]
        host_part_menu = log_menu_db_url.split("@")[1]
        log_menu_db_url = f"{log_menu_db_url.split('://')[0]}://{user_pass_part_menu.split(':')[0]}:********@{host_part_menu}"
    logger.info(f"Menü veritabanı için ayrı bağlantı adresi kullanılıyor: {log_menu_db_url}")
else:
    logger.info(f"Menü veritabanı için ana bağlantı adresi ({log_db_url}) kullanılacak.")
menu_db = Database(MENU_DATABASE_CONNECTION_STRING)


# DB_DATA_DIR artık SQLite dosyaları için doğrudan kullanılmıyor olsa da,
# başka amaçlar için (örn. log dosyaları için bir üst klasör) settings'de tanımlı kalabilir.
# Eğer sadece SQLite için kullanılıyorsa aşağıdaki blok kaldırılabilir veya loglamaya özel hale getirilebilir.
try:
    # Bu dizin oluşturma artık veritabanı dosyaları için değil,
    # örneğin log dosyaları gibi başka dosyalar için settings.DB_DATA_DIR kullanılacaksa anlamlı.
    # Eğer settings.DB_DATA_DIR sadece loglar içinse, log dosyasının yolu buna göre ayarlanmalı.
    # Şu anki log config'i direkt "neso_backend.log" şeklinde.
    if not DATABASE_CONNECTION_STRING.startswith("sqlite:///"):
        logger.info(f"PostgreSQL veya benzeri bir veritabanı kullanılıyor. '{settings.DB_DATA_DIR}' dizini SQLite için oluşturulmayacak.")
    elif settings.DB_DATA_DIR != ".": # Sadece SQLite ve özel bir dizin belirtilmişse
        os.makedirs(settings.DB_DATA_DIR, exist_ok=True)
        logger.info(f"SQLite için '{settings.DB_DATA_DIR}' dizini kontrol edildi/oluşturuldu.")
except OSError as e:
    logger.error(f"'{settings.DB_DATA_DIR}' dizini oluşturulurken hata: {e}.")


TR_TZ = dt_timezone(timedelta(hours=3))

# --- Pydantic Kullanıcı Modelleri ---
class KullaniciBase(BaseModel):
    kullanici_adi: str = Field(..., min_length=3, max_length=50)
    rol: KullaniciRol
    aktif_mi: bool = True

class KullaniciCreate(KullaniciBase):
    sifre: str = Field(..., min_length=6)

class KullaniciUpdate(BaseModel):
    kullanici_adi: Optional[str] = Field(None, min_length=3, max_length=50)
    rol: Optional[KullaniciRol] = None
    aktif_mi: Optional[bool] = None
    sifre: Optional[str] = Field(None, min_length=6)

class KullaniciInDBBase(KullaniciBase):
    id: int
    class Config:
        from_attributes = True # SQLAlchemy modelleriyle uyum için (Pydantic v2)

class Kullanici(KullaniciInDBBase):
    pass

class KullaniciInDB(KullaniciInDBBase):
    sifre_hash: str

# --- Token Modelleri ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    kullanici_adi: Union[str, None] = None

# --- Şifreleme ve Kimlik Doğrulama Yardımcı Fonksiyonları ---
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(dt_timezone.utc) + expires_delta
    else:
        expire = datetime.now(dt_timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

async def get_user_from_db(username: str) -> Union[KullaniciInDB, None]:
    # Kullanıcı tablosu ana `db` nesnesi üzerinden sorgulanacak
    query = "SELECT id, kullanici_adi, sifre_hash, rol, aktif_mi FROM kullanicilar WHERE kullanici_adi = :kullanici_adi"
    user_row = await db.fetch_one(query, {"kullanici_adi": username})
    if user_row:
        return KullaniciInDB(**user_row)
    return None

async def get_current_user(token: str = Depends(oauth2_scheme)) -> Kullanici:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Kimlik bilgileri doğrulanamadı",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: Union[str, None] = payload.get("sub")
        if username is None:
            logger.warning("Token'da kullanıcı adı (sub) bulunamadı.")
            raise credentials_exception
    except JWTError as e:
        logger.warning(f"JWT decode hatası: {e}")
        raise credentials_exception

    user_in_db = await get_user_from_db(username=username)
    if user_in_db is None:
        logger.warning(f"Token'daki kullanıcı '{username}' veritabanında bulunamadı.")
        raise credentials_exception
    return Kullanici.model_validate(user_in_db)

async def get_current_active_user(current_user: Kullanici = Depends(get_current_user)) -> Kullanici:
    if not current_user.aktif_mi:
        logger.warning(f"Pasif kullanıcı '{current_user.kullanici_adi}' işlem yapmaya çalıştı.")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Hesabınız aktif değil.")
    return current_user

# --- Rol Bazlı Yetkilendirme Dependency ---
def role_checker(required_roles: List[KullaniciRol]):
    async def checker(current_user: Kullanici = Depends(get_current_active_user)) -> Kullanici:
        if current_user.rol not in required_roles:
            logger.warning(
                f"Yetkisiz erişim denemesi: Kullanıcı '{current_user.kullanici_adi}' (Rol: {current_user.rol}), "
                f"Hedeflenen Roller: {required_roles}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Bu işlemi yapmak için yeterli yetkiniz yok."
            )
        logger.debug(f"Yetkili kullanıcı '{current_user.kullanici_adi}' (Rol: {current_user.rol}) işleme devam ediyor.")
        return current_user
    return checker


@app.on_event("startup")
async def startup_event():
    try:
        await db.connect()
        # Eğer menu_db ve db aynı veritabanını işaret ediyorsa, ikinci connect gereksiz olabilir
        # Ancak ayrı Database nesneleri olduğu için ayrı ayrı yönetilmeleri daha güvenli.
        if menu_db != db : # Sadece farklı nesnelerse ve aynı bağlantıyı kullanmıyorlarsa (ki şu an aynılar)
             await menu_db.connect()
        elif not menu_db.is_connected: # Eğer db ile aynıysa ve db zaten bağlandıysa, menu_db'yi de bağla (garanti olsun)
             await menu_db.connect()

        logger.info("✅ Veritabanı bağlantıları kuruldu.")
        await init_databases() # Hem ana hem menü tablolarını oluşturacak
        await update_system_prompt() # Menü cache'lerini de temizleyip prompt'u günceller
        logger.info(f"🚀 FastAPI uygulaması başlatıldı. Sistem mesajı güncellendi.")
    except Exception as e_startup:
        logger.critical(f"❌ Uygulama başlangıcında KRİTİK HATA: {e_startup}", exc_info=True)
        # Gerekirse burada uygulamayı sonlandırabilirsiniz.
        # raise SystemExit(f"Uygulama başlatılamadı: {e_startup}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🚪 Uygulama kapatılıyor...")
    try:
        if menu_db.is_connected: await menu_db.disconnect() # Önce menu_db kapatılabilir
        if db.is_connected: await db.disconnect()
        logger.info("✅ Veritabanı bağlantıları kapatıldı.")
    except Exception as e_disconnect:
        logger.error(f"Veritabanı bağlantıları kapatılırken hata: {e_disconnect}")

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
aktif_kasa_websocketleri: Set[WebSocket] = set()

async def broadcast_message(connections: Set[WebSocket], message: Dict, ws_type_name: str):
    if not connections:
        logger.warning(f"⚠️ Broadcast: Bağlı {ws_type_name} istemcisi yok. Mesaj: {message.get('type')}")
        return

    message_json = json.dumps(message, ensure_ascii=False)
    tasks = []
    disconnected_ws = set()

    for ws in list(connections):
        try:
            tasks.append(ws.send_text(message_json))
        except RuntimeError:
            disconnected_ws.add(ws)
            logger.warning(f"⚠️ {ws_type_name} WS bağlantısı zaten kopuk (RuntimeError), listeden kaldırılıyor: {ws.client}")
        except Exception as e_send:
            disconnected_ws.add(ws)
            logger.warning(f"⚠️ {ws_type_name} WS gönderme sırasında BEKLENMEDİK hata ({ws.client}): {e_send}")

    for ws in disconnected_ws:
        connections.discard(ws)

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # İlgili websocket'i bulmak zor olabilir, bu yüzden genel bir loglama
                logger.warning(f"⚠️ {ws_type_name} WS gönderme (asyncio.gather) hatası: {result}")


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
            except json.JSONDecodeError:
                logger.warning(f"⚠️ {endpoint_name} WS: Geçersiz JSON formatında mesaj alındı: {data} from {client_info}")
            except Exception as e_inner:
                logger.error(f"❌ {endpoint_name} WS mesaj işleme hatası ({client_info}): {e_inner} - Mesaj: {data}", exc_info=True)
    except WebSocketDisconnect as e:
        if e.code == 1000 or e.code == 1001:
            logger.info(f"🔌 {endpoint_name} WS normal şekilde kapandı (Kod {e.code}): {client_info}")
        elif e.code == 1012: # Service Restart
            logger.warning(f"🔌 {endpoint_name} WS beklenmedik şekilde kapandı (Kod {e.code} - Sunucu Yeniden Başlıyor Olabilir): {client_info}")
        else:
            logger.warning(f"🔌 {endpoint_name} WS beklenmedik şekilde kapandı (Kod {e.code}): {client_info}")
    except Exception as e_outer:
        logger.error(f"❌ {endpoint_name} WS beklenmedik genel hata ({client_info}): {e_outer}", exc_info=True)
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

@app.websocket("/ws/kasa")
async def websocket_kasa_endpoint(websocket: WebSocket):
    await websocket_lifecycle(websocket, aktif_kasa_websocketleri, "Kasa")

# Veritabanı İşlemleri (update_table_status)
async def update_table_status(masa_id: str, islem: str = "Erişim"):
    now = datetime.now(TR_TZ)
    try:
        # Bu tablo ana `db` üzerinden yönetilecek
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

# Middleware (track_active_users)
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

# --- Login Endpoint ---
@app.post("/token", response_model=Token, tags=["Kimlik Doğrulama"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    logger.info(f"Giriş denemesi: Kullanıcı adı '{form_data.username}'")
    user_in_db = await get_user_from_db(username=form_data.username) # get_user_from_db zaten ana db'yi kullanıyor
    if not user_in_db or not verify_password(form_data.password, user_in_db.sifre_hash):
        logger.warning(f"Başarısız giriş: Kullanıcı '{form_data.username}' için geçersiz kimlik bilgileri.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Yanlış kullanıcı adı veya şifre",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user_in_db.aktif_mi:
        logger.warning(f"Pasif kullanıcı '{form_data.username}' giriş yapmaya çalıştı.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Hesabınız aktif değil. Lütfen yönetici ile iletişime geçin."
        )

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_in_db.kullanici_adi},
        expires_delta=access_token_expires
    )
    logger.info(f"Kullanıcı '{user_in_db.kullanici_adi}' (Rol: {user_in_db.rol}) başarıyla giriş yaptı. Token oluşturuldu.")
    return {"access_token": access_token, "token_type": "bearer"}


# Pydantic Modelleri (Sipariş, Menü vb.)
class Durum(str, Enum):
    BEKLIYOR = "bekliyor"
    HAZIRLANIYOR = "hazirlaniyor"
    HAZIR = "hazir"
    IPTAL = "iptal"
    ODENDI = "odendi"

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
    durum: Durum

class AktifMasaOzet(BaseModel):
    masa_id: str
    odenmemis_tutar: float
    aktif_siparis_sayisi: int
    siparis_detaylari: Optional[List[Dict]] = None

class KasaOdemeData(BaseModel):
    odeme_yontemi: str = Field(..., description="Ödeme yöntemi (örn: Nakit, Kredi Kartı)")

class MenuEkleData(BaseModel):
    ad: str = Field(..., min_length=1)
    fiyat: float = Field(..., gt=0)
    kategori: str = Field(..., min_length=1)

class SesliYanitData(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = Field(default="tr-TR", pattern=r"^[a-z]{2}-[A-Z]{2}$")


# --- Korunan Endpoint Örnekleri (Rol Tabanlı) ---

@app.get("/users/me", response_model=Kullanici, tags=["Kullanıcılar"])
async def read_users_me(current_user: Kullanici = Depends(get_current_active_user)):
    logger.info(f"Kullanıcı '{current_user.kullanici_adi}' kendi bilgilerini istedi.")
    return current_user

@app.get("/aktif-masalar/ws-count", tags=["Admin"])
async def get_active_tables_ws_count_endpoint(
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"Admin '{current_user.kullanici_adi}' aktif WS masa sayısını istedi.")
    try:
        return {"aktif_mutfak_ws_sayisi": len(aktif_mutfak_websocketleri),
                "aktif_admin_ws_sayisi": len(aktif_admin_websocketleri),
                "aktif_kasa_ws_sayisi": len(aktif_kasa_websocketleri)
                }
    except Exception as e:
        logger.error(f"❌ Aktif masalar WS bağlantı sayısı alınamadı: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="WS bağlantı sayısı alınamadı."
        )


# Sipariş Yönetimi (Tüm sipariş işlemleri ana `db` üzerinden yapılacak)
@app.patch("/siparis/{id}", tags=["Siparişler"])
async def patch_order_endpoint(
    id: int = Path(..., description="Güncellenecek siparişin ID'si"),
    data: SiparisGuncelleData = Body(...),
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN, KullaniciRol.MUTFAK_PERSONELI, KullaniciRol.BARISTA]))
):
    logger.info(f"🔧 PATCH /siparis/{id} ile durum güncelleme isteği (Kullanıcı: {current_user.kullanici_adi}, Rol: {current_user.rol}): {data.durum}")
    try:
        async with db.transaction(): # Ana db kullanılıyor
            order_info = await db.fetch_one("SELECT masa, odeme_yontemi FROM siparisler WHERE id = :id", {"id": id})
            if not order_info:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sipariş bulunamadı.")

            siparis_masasi = order_info["masa"]

            updated_raw = await db.fetch_one(
                "UPDATE siparisler SET durum = :durum WHERE id = :id RETURNING id, masa, durum, sepet, istek, zaman, odeme_yontemi",
                {"durum": data.durum.value, "id": id}
            )
        if not updated_raw:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sipariş bulunamadı veya güncellenemedi.")

        updated_order = dict(updated_raw)
        try:
            updated_order["sepet"] = json.loads(updated_order.get("sepet", "[]"))
        except json.JSONDecodeError:
            updated_order["sepet"] = []
            logger.warning(f"Sipariş {id} sepet JSON parse hatası (patch_order_endpoint).")

        notif_data = {**updated_order, "zaman": datetime.now(TR_TZ).isoformat()}
        notification = {"type": "durum", "data": notif_data}
        await broadcast_message(aktif_mutfak_websocketleri, notification, "Mutfak/Masa")
        await broadcast_message(aktif_admin_websocketleri, notification, "Admin")
        await broadcast_message(aktif_kasa_websocketleri, notification, "Kasa")
        await update_table_status(siparis_masasi, f"Sipariş {id} durumu güncellendi -> {updated_order['durum']} (by {current_user.kullanici_adi})")
        return {"message": f"Sipariş {id} güncellendi.", "data": updated_order}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ PATCH /siparis/{id} hatası: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Sipariş durumu güncellenirken hata oluştu.")


@app.delete("/siparis/{id}", tags=["Siparişler"])
async def delete_order_by_admin_endpoint(
    id: int = Path(..., description="İptal edilecek (silinecek) siparişin ID'si"),
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"🗑️ ADMIN DELETE (as cancel) /siparis/{id} ile iptal isteği (Kullanıcı: {current_user.kullanici_adi})")
    # Ana db kullanılıyor
    row = await db.fetch_one("SELECT zaman, masa, durum, odeme_yontemi FROM siparisler WHERE id = :id", {"id": id})
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sipariş bulunamadı.")
    if row["durum"] == Durum.IPTAL.value:
        return {"message": f"Sipariş {id} zaten iptal edilmiş."}

    try:
        async with db.transaction(): # Ana db kullanılıyor
            await db.execute("UPDATE siparisler SET durum = :durum WHERE id = :id", {"durum": Durum.IPTAL.value, "id": id})

        notif_data = { "id": id, "masa": row["masa"], "durum": Durum.IPTAL.value, "zaman": datetime.now(TR_TZ).isoformat(), "odeme_yontemi": row["odeme_yontemi"]}
        notification = {"type": "durum", "data": notif_data}
        await broadcast_message(aktif_mutfak_websocketleri, notification, "Mutfak/Masa")
        await broadcast_message(aktif_admin_websocketleri, notification, "Admin")
        await broadcast_message(aktif_kasa_websocketleri, notification, "Kasa")

        await update_table_status(row["masa"], f"Sipariş {id} admin ({current_user.kullanici_adi}) tarafından iptal edildi")
        logger.info(f"Sipariş {id} (Masa: {row['masa']}) admin ({current_user.kullanici_adi}) tarafından başarıyla iptal edildi.")
        return {"message": f"Sipariş {id} admin tarafından başarıyla iptal edildi."}
    except Exception as e:
        logger.error(f"❌ ADMIN DELETE (as cancel) /siparis/{id} hatası: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Sipariş admin tarafından iptal edilirken hata oluştu.")


@app.post("/musteri/siparis/{siparis_id}/iptal", status_code=status.HTTP_200_OK, tags=["Müşteri İşlemleri"])
async def cancel_order_by_customer_endpoint(
    siparis_id: int = Path(..., description="İptal edilecek siparişin ID'si"),
    masa_no: str = Query(..., description="Siparişin verildiği masa numarası/adı")
):
    logger.info(f"🗑️ Müşteri sipariş iptal isteği: Sipariş ID {siparis_id}, Masa No {masa_no}")
    # Ana db kullanılıyor
    order_details = await db.fetch_one(
        "SELECT id, zaman, masa, durum, odeme_yontemi FROM siparisler WHERE id = :siparis_id AND masa = :masa_no",
        {"siparis_id": siparis_id, "masa_no": masa_no}
    )
    if not order_details:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="İptal edilecek sipariş bulunamadı veya bu masaya ait değil.")
    if order_details["durum"] == "iptal":
        return {"message": "Bu sipariş zaten iptal edilmiş."}
    if order_details["durum"] not in [Durum.BEKLIYOR.value, Durum.HAZIRLANIYOR.value]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Siparişinizin durumu ({order_details['durum']}) iptal işlemi için uygun değil.")

    olusturma_zamani_str = order_details["zaman"]
    try:
        olusturma_naive = datetime.strptime(olusturma_zamani_str, "%Y-%m-%d %H:%M:%S")
        olusturma_tr = olusturma_naive.replace(tzinfo=TR_TZ)
    except ValueError: # Tarih formatı PostgreSQL'den farklı gelebilir, ISO formatına göre parse etmeyi deneyebiliriz.
        try:
            olusturma_tr = datetime.fromisoformat(olusturma_zamani_str).astimezone(TR_TZ)
        except ValueError:
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Sipariş zamanı okunamadı.")


    if datetime.now(TR_TZ) - olusturma_tr > timedelta(minutes=2):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Bu sipariş 2 dakikayı geçtiği için artık iptal edilemez.")

    try:
        async with db.transaction(): # Ana db kullanılıyor
            await db.execute("UPDATE siparisler SET durum = 'iptal' WHERE id = :id", {"id": siparis_id})
        notif_data = { "id": siparis_id, "masa": masa_no, "durum": "iptal", "zaman": datetime.now(TR_TZ).isoformat(), "odeme_yontemi": order_details["odeme_yontemi"]}
        notification = {"type": "durum", "data": notif_data}
        await broadcast_message(aktif_mutfak_websocketleri, notification, "Mutfak/Masa")
        await broadcast_message(aktif_admin_websocketleri, notification, "Admin")
        await broadcast_message(aktif_kasa_websocketleri, notification, "Kasa")
        await update_table_status(masa_no, f"Sipariş {siparis_id} müşteri tarafından iptal edildi (2dk sınırı içinde)")
        logger.info(f"Sipariş {siparis_id} (Masa: {masa_no}) müşteri tarafından başarıyla iptal edildi.")
        return {"message": f"Siparişiniz (ID: {siparis_id}) başarıyla iptal edildi."}
    except Exception as e:
        logger.error(f"❌ Müşteri sipariş iptali sırasında (Sipariş ID: {siparis_id}, Masa: {masa_no}) hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Siparişiniz iptal edilirken bir sunucu hatası oluştu.")


@app.post("/siparis-ekle", status_code=status.HTTP_201_CREATED, tags=["Müşteri İşlemleri"])
async def add_order_endpoint(data: SiparisEkleData):
    masa = data.masa
    sepet = data.sepet
    istek = data.istek
    yanit = data.yanit
    # PostgreSQL için zaman damgası genellikle ISO formatında veya TIMESTAMP tipinde direkt yönetilir.
    # Strftime yerine datetime nesnesini direkt yollamak daha iyi olabilir (databases kütüphanesi halleder).
    simdiki_zaman_obj = datetime.now(TR_TZ)
    db_zaman_kayit = simdiki_zaman_obj # Ya da .isoformat()
    yanit_zaman_iso_str = simdiki_zaman_obj.isoformat()

    logger.info(f"📥 Yeni sipariş isteği alındı: Masa {masa}, {len(sepet)} çeşit ürün. AI Yanıtı: {yanit[:200] if yanit else 'Yok'}...")
    cached_price_dict = await get_menu_price_dict() # menu_db kullanıyor (aynı PostgreSQL DB'si)
    cached_stock_dict = await get_menu_stock_dict() # menu_db kullanıyor

    processed_sepet = []
    for item in sepet:
        urun_adi_lower = item.urun.lower().strip()
        stok_kontrol_degeri = cached_stock_dict.get(urun_adi_lower)
        if stok_kontrol_degeri is None or stok_kontrol_degeri == 0: # Stokta yok veya menüde bulunmuyor
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"'{item.urun}' adlı ürün stokta yok veya menüde bulunmuyor.")
        item_dict = item.model_dump()
        cached_fiyat = cached_price_dict.get(urun_adi_lower, item.fiyat) # Fiyatı cache'den al
        if cached_fiyat != item.fiyat: logger.warning(f"Fiyat uyuşmazlığı: Ürün '{item.urun}', Frontend Fiyatı: {item.fiyat}, Cache Fiyatı: {cached_fiyat}. Cache fiyatı kullanılacak.")
        item_dict['fiyat'] = cached_fiyat
        processed_sepet.append(item_dict)

    if not processed_sepet: raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Sepette geçerli ürün yok.")
    istek_ozet = ", ".join([f"{p_item['adet']}x {p_item['urun']}" for p_item in processed_sepet])

    try:
        async with db.transaction(): # Ana db kullanılıyor
            siparis_id = await db.fetch_val("""
                INSERT INTO siparisler (masa, istek, yanit, sepet, zaman, durum, odeme_yontemi)
                VALUES (:masa, :istek, :yanit, :sepet, :zaman, 'bekliyor', NULL)
                RETURNING id
            """, { "masa": masa, "istek": istek or istek_ozet, "yanit": yanit, "sepet": json.dumps(processed_sepet, ensure_ascii=False), "zaman": db_zaman_kayit }) # datetime objesi
            if siparis_id is None: raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Sipariş kaydedilemedi.")

        siparis_bilgisi_ws = { "type": "siparis", "data": {"id": siparis_id, "masa": masa, "istek": istek or istek_ozet, "sepet": processed_sepet, "zaman": yanit_zaman_iso_str, "durum": "bekliyor", "odeme_yontemi": None}}
        await broadcast_message(aktif_mutfak_websocketleri, siparis_bilgisi_ws, "Mutfak/Masa")
        await broadcast_message(aktif_admin_websocketleri, siparis_bilgisi_ws, "Admin")
        await broadcast_message(aktif_kasa_websocketleri, siparis_bilgisi_ws, "Kasa")
        await update_table_status(masa, f"Sipariş verdi ({len(processed_sepet)} çeşit ürün)")
        logger.info(f"✅ Sipariş (ID: {siparis_id}) Masa: {masa} kaydedildi.")
        return { "mesaj": "Siparişiniz başarıyla alındı ve mutfağa iletildi.", "siparisId": siparis_id, "zaman": yanit_zaman_iso_str }
    except HTTPException as http_exc: raise http_exc
    except Exception as e:
        logger.error(f"❌ Sipariş ekleme hatası (Masa: {masa}): {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Sipariş işlenirken sunucu hatası.")


@app.post("/siparis-guncelle", tags=["Siparişler"]) # Bu endpoint ID'yi body'den değil path'ten almalıydı. PATCH /siparis/{id} daha doğru.
async def update_order_status_endpoint(
    data: SiparisGuncelleData, # Bu modelde ID yok, frontend PATCH /siparis/{id} kullanmalı.
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN, KullaniciRol.MUTFAK_PERSONELI, KullaniciRol.BARISTA]))
):
    logger.warning(f"/siparis-guncelle endpoint'i kullanıldı. Bu endpoint ID'yi body'de bekleyebilir veya /siparis/{{id}} (PATCH) kullanılmalıdır.")
    # Bu endpoint'in doğru çalışması için data içinde 'id' alanı olması beklenir veya path'ten alınmalı.
    # Mevcut SiparisGuncelleData modelinde 'id' yok. Frontend'in /siparis/{id} (PATCH) kullandığını varsayıyoruz.
    # Bu nedenle bu endpoint'in düzeltilmesi veya kaldırılması gerekir. Şimdilik hata döndürüyor.
    raise HTTPException(status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail="/siparis/{id} (PATCH) endpoint'ini kullanın.")


@app.get("/siparisler", tags=["Siparişler"])
async def get_orders_endpoint(
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN, KullaniciRol.KASIYER, KullaniciRol.MUTFAK_PERSONELI, KullaniciRol.BARISTA]))
):
    logger.info(f"📋 Tüm siparişler listeleniyor (Kullanıcı: {current_user.kullanici_adi})")
    try:
        # Ana db kullanılıyor
        orders_raw = await db.fetch_all("SELECT id, masa, istek, yanit, sepet, zaman, durum, odeme_yontemi FROM siparisler ORDER BY id DESC")
        orders_data = []
        for row in orders_raw:
            order_dict = dict(row)
            try:
                sepet_str = order_dict.get('sepet')
                order_dict['sepet'] = json.loads(sepet_str if sepet_str else '[]')
            except json.JSONDecodeError:
                order_dict['sepet'] = []
            # PostgreSQL'den zaman string'i farklı formatta gelebilir, ISO formatına çevirebiliriz.
            if isinstance(order_dict.get('zaman'), datetime):
                 order_dict['zaman'] = order_dict['zaman'].isoformat()
            orders_data.append(order_dict)
        return {"orders": orders_data}
    except Exception as e:
        logger.error(f"❌ Tüm siparişler alınırken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Siparişler veritabanından alınırken bir sorun oluştu.")

# Veritabanı Başlatma
async def init_db(): # Bu fonksiyon ana `db` nesnesini kullanarak tabloları oluşturacak
    logger.info(f"Ana veritabanı tabloları kontrol ediliyor/oluşturuluyor...")
    try:
        async with db.transaction():
            # PostgreSQL'de `AUTOINCREMENT` yerine `SERIAL PRIMARY KEY` veya `IDENTITY` kullanılır.
            # `databases` kütüphanesi bunu soyutlayabilir, değilse DDL güncellenmeli.
            # Şimdilik `AUTOINCREMENT` bırakılıyor, kütüphaneye güveniliyor.
            await db.execute("""
                CREATE TABLE IF NOT EXISTS siparisler (
                    id SERIAL PRIMARY KEY,
                    masa TEXT NOT NULL,
                    istek TEXT,
                    yanit TEXT,
                    sepet TEXT,
                    zaman TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    durum TEXT DEFAULT 'bekliyor' CHECK(durum IN ('bekliyor', 'hazirlaniyor', 'hazir', 'iptal', 'odendi')),
                    odeme_yontemi TEXT
                )""")
            await db.execute("""
                CREATE TABLE IF NOT EXISTS masa_durumlar (
                    id SERIAL PRIMARY KEY,
                    masa_id TEXT UNIQUE NOT NULL,
                    son_erisim TIMESTAMP WITH TIME ZONE NOT NULL,
                    aktif BOOLEAN DEFAULT TRUE,
                    son_islem TEXT
                )""")
            await db.execute("""
                CREATE TABLE IF NOT EXISTS kullanicilar (
                    id SERIAL PRIMARY KEY,
                    kullanici_adi TEXT UNIQUE NOT NULL,
                    sifre_hash TEXT NOT NULL,
                    rol TEXT NOT NULL CHECK(rol IN ('admin', 'kasiyer', 'barista', 'mutfak_personeli')),
                    aktif_mi BOOLEAN DEFAULT TRUE,
                    olusturulma_tarihi TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )""")
            # PostgreSQL'de index isimleri globaldir, "IF NOT EXISTS" kullanmak iyi bir pratiktir.
            await db.execute("CREATE INDEX IF NOT EXISTS idx_siparisler_masa_zaman ON siparisler(masa, zaman DESC)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_masa_durumlar_erisim ON masa_durumlar(son_erisim DESC)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_kullanicilar_kullanici_adi ON kullanicilar(kullanici_adi)")

            existing_admin = await db.fetch_one("SELECT id FROM kullanicilar WHERE kullanici_adi = :kullanici_adi", {"kullanici_adi": settings.DEFAULT_ADMIN_USERNAME})
            if not existing_admin:
                hashed_password = get_password_hash(settings.DEFAULT_ADMIN_PASSWORD)
                await db.execute(
                    """
                    INSERT INTO kullanicilar (kullanici_adi, sifre_hash, rol, aktif_mi)
                    VALUES (:kullanici_adi, :sifre_hash, :rol, TRUE)
                    """,
                    {
                        "kullanici_adi": settings.DEFAULT_ADMIN_USERNAME,
                        "sifre_hash": hashed_password,
                        "rol": KullaniciRol.ADMIN.value
                    }
                )
                logger.info(f"Varsayılan admin kullanıcısı '{settings.DEFAULT_ADMIN_USERNAME}' veritabanına eklendi.")
            else:
                logger.info(f"Varsayılan admin kullanıcısı '{settings.DEFAULT_ADMIN_USERNAME}' zaten mevcut.")
        logger.info(f"✅ Ana veritabanı tabloları başarıyla doğrulandı/oluşturuldu.")
    except Exception as e:
        logger.critical(f"❌ Ana veritabanı tabloları başlatılırken kritik hata: {e}", exc_info=True)
        raise


async def init_menu_db(): # Bu fonksiyon `menu_db` nesnesini (ki o da ana db'ye bağlı) kullanarak tabloları oluşturacak
    logger.info(f"Menü veritabanı tabloları kontrol ediliyor/oluşturuluyor...")
    try:
        async with menu_db.transaction(): # `menu_db` de aynı PostgreSQL'i kullanıyor
            # `COLLATE NOCASE` PostgreSQL'de bu şekilde kullanılamaz, kaldırıldı.
            # Case-insensitive sorgular için ILIKE veya LOWER() kullanılmalı.
            await menu_db.execute("""
                CREATE TABLE IF NOT EXISTS kategoriler (
                    id SERIAL PRIMARY KEY,
                    isim TEXT UNIQUE NOT NULL
                )""") # COLLATE NOCASE kaldırıldı
            await menu_db.execute("""
                CREATE TABLE IF NOT EXISTS menu (
                    id SERIAL PRIMARY KEY,
                    ad TEXT NOT NULL,
                    fiyat REAL NOT NULL CHECK(fiyat >= 0),
                    kategori_id INTEGER NOT NULL,
                    stok_durumu INTEGER DEFAULT 1,
                    FOREIGN KEY (kategori_id) REFERENCES kategoriler(id) ON DELETE CASCADE,
                    UNIQUE(ad, kategori_id)
                )""") # COLLATE NOCASE kaldırıldı
            await menu_db.execute("CREATE INDEX IF NOT EXISTS idx_menu_kategori_id ON menu(kategori_id)")
            await menu_db.execute("CREATE INDEX IF NOT EXISTS idx_menu_ad ON menu(ad)")
        logger.info(f"✅ Menü veritabanı tabloları başarıyla doğrulandı/oluşturuldu.")
    except Exception as e:
        logger.critical(f"❌ Menü veritabanı tabloları başlatılırken kritik hata: {e}", exc_info=True)
        raise

async def init_databases():
    await init_db() # Ana tabloları oluşturur
    await init_menu_db() # Menü tablolarını aynı veritabanında oluşturur

# Menü Yönetimi (Fonksiyonlar) - Tüm menü işlemleri `menu_db` üzerinden yapılacak (aynı PostgreSQL DB'si)
@alru_cache(maxsize=1)
async def get_menu_for_prompt_cached() -> str:
    logger.info(">>> GET_MENU_FOR_PROMPT_CACHED ÇAĞRILIYOR...")
    try:
        if not menu_db.is_connected: await menu_db.connect()
        # PostgreSQL'de case-insensitive sorgu için ILIKE veya LOWER() kullanılır.
        # Şimdilik direkt sorgu bırakılıyor, collation ayarına veya uygulama mantığına bağlı.
        # Örnek: "SELECT k.isim as kategori_isim, m.ad as urun_ad FROM menu m JOIN kategoriler k ON m.kategori_id = k.id WHERE m.stok_durumu = 1 ORDER BY LOWER(k.isim), LOWER(m.ad)"
        query = """ SELECT k.isim as kategori_isim, m.ad as urun_ad FROM menu m
                    JOIN kategoriler k ON m.kategori_id = k.id
                    WHERE m.stok_durumu = 1 ORDER BY k.isim, m.ad """
        urunler_raw = await menu_db.fetch_all(query)
        if not urunler_raw: return "Üzgünüz, şu anda menümüzde aktif ürün bulunmamaktadır."
        kategorili_menu: Dict[str, List[str]] = {}
        for row in urunler_raw:
            try: kategorili_menu.setdefault(row['kategori_isim'], []).append(row['urun_ad'])
            except Exception as e_row: logger.error(f"get_menu_for_prompt_cached: Satır işlenirken hata: {e_row}", exc_info=True)
        if not kategorili_menu: return "Üzgünüz, menü bilgisi şu anda düzgün bir şekilde formatlanamıyor."
        menu_aciklama_list = [f"- {kategori}: {', '.join(urun_listesi)}" for kategori, urun_listesi in kategorili_menu.items() if urun_listesi]
        if not menu_aciklama_list: return "Üzgünüz, menüde listelenecek ürün bulunamadı."
        logger.info(f"Menü prompt için başarıyla oluşturuldu ({len(kategorili_menu)} kategori).")
        return "\n".join(menu_aciklama_list)
    except Exception as e:
        logger.error(f"❌ Menü prompt oluşturma hatası: {e}", exc_info=True)
        return "Teknik bir sorun nedeniyle menü bilgisine şu anda ulaşılamıyor."

@alru_cache(maxsize=1)
async def get_menu_price_dict() -> Dict[str, float]:
    logger.info(">>> get_menu_price_dict ÇAĞRILIYOR...")
    try:
        if not menu_db.is_connected: await menu_db.connect()
        prices_raw = await menu_db.fetch_all("SELECT ad, fiyat FROM menu")
        # Ürün adlarını küçük harfe çevirerek cache'liyoruz, tutarlılık için
        price_dict = {row['ad'].lower().strip(): float(row['fiyat']) for row in prices_raw}
        logger.info(f"Fiyat sözlüğü {len(price_dict)} ürün için oluşturuldu/alındı.")
        return price_dict
    except Exception as e:
        logger.error(f"❌ Fiyat sözlüğü oluşturma/alma hatası: {e}", exc_info=True)
        return {}

@alru_cache(maxsize=1)
async def get_menu_stock_dict() -> Dict[str, int]:
    logger.info(">>> get_menu_stock_dict ÇAĞRILIYOR...")
    try:
        if not menu_db.is_connected: await menu_db.connect()
        stocks_raw = await menu_db.fetch_all("SELECT ad, stok_durumu FROM menu")
        if not stocks_raw: return {}
        stock_dict = {}
        for row in stocks_raw:
            try: stock_dict[str(row['ad']).lower().strip()] = int(row['stok_durumu'])
            except Exception as e_loop: logger.error(f"Stok sözlüğü oluştururken satır işleme hatası: {e_loop}", exc_info=True)
        logger.info(f">>> get_menu_stock_dict: Oluşturulan stock_dict ({len(stock_dict)} öğe).")
        return stock_dict
    except Exception as e_main:
        logger.error(f"❌ Stok sözlüğü oluşturma/alma sırasında genel hata: {e_main}", exc_info=True)
        return {}

SISTEM_MESAJI_ICERIK_TEMPLATE = (
    "Sen Fıstık Kafe için Neso adında, çok yetenekli, kibar ve hafif espirili bir sipariş asistanısın. "
    "Müşterilerin siparişlerini alacak, sorularını yanıtlayacak ve onlara yardımcı olacaksın. "
    "Siparişleri JSON formatında çıkarman gerekiyor. Eğer sipariş dışında bir soru sorulursa, normal bir şekilde yanıt ver ve JSON çıkarma. "
    "Kullanıcılarla samimi bir dil kullan ama saygıyı elden bırakma. "
    "Menüdeki ürünler ve kategorileri şunlardır (eğer bir ürün menüde yoksa, kibarca olmadığını belirt ve alternatif öner): \n"
    "{menu_prompt_data}\n"
    "Fiyat bilgisi menüde mevcut, müşteri sorarsa söyleyebilirsin ama JSON çıktısına ekleme. "
    "Sipariş alırken ürünlerin tam adlarını ve adetlerini net bir şekilde anlamaya çalış. "
    "Eğer kullanıcı bir ürünün adını eksik veya yanlış söylerse, kibarca düzeltmesini iste veya en yakın tahmini yap. "
    "Kullanıcıya her zaman pozitif ve yardımcı bir tutum sergile. "
    "Örnek JSON formatı: {\"konusma_metni\": \"Elbette, bir Türk kahvesi ve bir su. Başka bir isteğiniz var mı?\", \"sepet\": [{\"urun\": \"Türk Kahvesi\", \"adet\": 1, \"kategori\": \"Sıcak İçecekler\"}, {\"urun\": \"Su\", \"adet\": 1, \"kategori\": \"Soğuk İçecekler\"}], \"musteri_notu\": \"Kahvem şekersiz olsun lütfen.\"} "
    "Eğer kullanıcı sadece sohbet ediyor veya soru soruyorsa, JSON yerine sadece metin olarak yanıt ver. Örneğin: \"Kafemiz saat akşam 10'a kadar açık.\". "
    "JSON formatında \"konusma_metni\" alanı senin müşteriye söyleyeceğin tam yanıtı içermelidir. Bu yanıt, siparişi onaylayan veya soruya cevap veren doğal bir cümle olmalı. "
    "\"sepet\" alanı bir liste olmalı ve her bir elemanı ürün adı, adedi ve kategorisini içeren bir obje olmalı. Sadece menüde var olan ve stokta olan ürünleri sepete ekle. "
    "\"musteri_notu\" alanı, müşterinin siparişle ilgili özel isteklerini (şekersiz, buzlu vb.) içerebilir. Yoksa bu alanı ekleme. "
    "Eğer kullanıcı menü dışı bir şey isterse, kibarca olmadığını belirtip, menüden bir şey önerebilirsin ve JSON ÇIKARMA. "
    "Sipariş tamamsa ve kullanıcı onaylarsa JSON çıkar. Emin değilsen veya soru soruyorsan JSON ÇIKARMA, sadece konuşma metni döndür. "
    "Kullanıcı 'hesabı isteyebilir miyim', 'ödeyeceğim' gibi bir şey söylerse, 'Tabii, hemen ilgileniyorum.' gibi bir yanıt ver ve JSON ÇIKARMA. "
    "Şimdi kullanıcının talebini bu kurallara ve örneklere göre işle ve uygun JSON çıktısını üret."
)

SYSTEM_PROMPT: Optional[Dict[str, str]] = None

async def update_system_prompt():
    global SYSTEM_PROMPT
    logger.info("🔄 Sistem mesajı (menü bilgisi) güncelleniyor...")
    menu_data_for_prompt = "Menü bilgisi geçici olarak yüklenemedi."
    try:
        if hasattr(get_menu_for_prompt_cached, 'cache_clear'): get_menu_for_prompt_cached.cache_clear()
        if hasattr(get_menu_price_dict, 'cache_clear'): get_menu_price_dict.cache_clear()
        if hasattr(get_menu_stock_dict, 'cache_clear'): get_menu_stock_dict.cache_clear()
        logger.info("İlgili menü cache'leri temizlendi (update_system_prompt).")
        menu_data_for_prompt = await get_menu_for_prompt_cached()
        current_system_content = SISTEM_MESAJI_ICERIK_TEMPLATE.format(menu_prompt_data=menu_data_for_prompt)
        SYSTEM_PROMPT = {"role": "system", "content": current_system_content}
        logger.info(f"✅ Sistem mesajı başarıyla güncellendi.")
    except Exception as e:
        logger.error(f"❌ Sistem mesajı güncellenirken BEKLENMEDİK BİR HATA oluştu: {e}", exc_info=True)
        if SYSTEM_PROMPT is None: # Sadece ilk başlatmada veya kritik bir hatada fallback
            current_system_content = SISTEM_MESAJI_ICERIK_TEMPLATE.format(menu_prompt_data="Menü bilgisi yüklenirken hata oluştu.")
            SYSTEM_PROMPT = {"role": "system", "content": current_system_content}
            logger.warning(f"Fallback sistem mesajı (BEKLENMEDİK HATA sonrası update_system_prompt içinde) kullanılıyor.")


@app.get("/admin/clear-menu-caches", tags=["Admin İşlemleri"])
async def clear_all_caches_endpoint(current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))):
    logger.info(f"Admin '{current_user.kullanici_adi}' tarafından manuel cache temizleme isteği alındı.")
    await update_system_prompt() # Bu fonksiyon cache'leri temizleyip promptu güncelliyor.
    return {"message": "Menü, fiyat ve stok cache'leri başarıyla temizlendi. Sistem promptu güncellendi."}

@app.get("/menu", tags=["Menü"])
async def get_full_menu_endpoint():
    logger.info("Tam menü isteniyor (/menu)...")
    try:
        full_menu_data = []
        # menu_db (aynı PostgreSQL DB'si) kullanılıyor
        kategoriler_raw = await menu_db.fetch_all("SELECT id, isim FROM kategoriler ORDER BY isim")
        for kat_row in kategoriler_raw:
            urunler_raw = await menu_db.fetch_all(
                "SELECT ad, fiyat, stok_durumu FROM menu WHERE kategori_id = :id ORDER BY ad",
                {"id": kat_row['id']}
            )
            # Pydantic modeline uygun hale getirme veya direkt dict listesi
            full_menu_data.append({ "kategori": kat_row['isim'], "urunler": [dict(urun) for urun in urunler_raw]})
        logger.info(f"✅ Tam menü başarıyla alındı ({len(full_menu_data)} kategori).")
        return {"menu": full_menu_data}
    except Exception as e:
        logger.error(f"❌ Tam menü alınırken veritabanı hatası: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Menü bilgileri alınırken bir sorun oluştu.")

@app.post("/menu/ekle", status_code=status.HTTP_201_CREATED, tags=["Menü Yönetimi"])
async def add_menu_item_endpoint(
    item_data: MenuEkleData,
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"📝 Menüye yeni ürün ekleme isteği (Kullanıcı: {current_user.kullanici_adi}): {item_data.ad} ({item_data.kategori})")
    try:
        async with menu_db.transaction(): # menu_db (aynı PostgreSQL DB'si) kullanılıyor
            # PostgreSQL'de INSERT OR IGNORE yerine ON CONFLICT DO NOTHING kullanılır.
            # `databases` kütüphanesi bunu soyutlamıyorsa, sorgu güncellenmeli.
            # Şimdilik `INSERT OR IGNORE` bırakılıyor, kütüphaneye güveniliyor veya hata verebilir.
            # Doğrusu: "INSERT INTO kategoriler (isim) VALUES (:isim) ON CONFLICT (isim) DO NOTHING"
            await menu_db.execute("INSERT INTO kategoriler (isim) VALUES (:isim) ON CONFLICT (isim) DO NOTHING", {"isim": item_data.kategori})
            category_id_row = await menu_db.fetch_one("SELECT id FROM kategoriler WHERE isim = :isim", {"isim": item_data.kategori}) # ILIKE veya LOWER() gerekebilir
            if not category_id_row: raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Kategori oluşturulurken veya bulunurken bir sorun oluştu.")
            category_id = category_id_row['id']
            try:
                # RETURNING id PostgreSQL'de de çalışır.
                item_id = await menu_db.fetch_val("""
                    INSERT INTO menu (ad, fiyat, kategori_id, stok_durumu) VALUES (:ad, :fiyat, :kategori_id, 1) RETURNING id
                """, {"ad": item_data.ad, "fiyat": item_data.fiyat, "kategori_id": category_id})
            except Exception as e_db: # UNIQUE constraint hatasını daha genel yakalamak gerekebilir (PostgreSQL'de farklı hata kodu/mesajı olabilir)
                 if "UNIQUE constraint failed" in str(e_db).lower() or "duplicate key value violates unique constraint" in str(e_db).lower() : # PostgreSQL için eklendi
                    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"'{item_data.ad}' adlı ürün bu kategoride zaten mevcut.")
                 raise e_db # Diğer DB hatalarını tekrar fırlat
        await update_system_prompt() # Cache'i temizler ve prompt'u günceller
        return {"mesaj": f"'{item_data.ad}' ürünü menüye başarıyla eklendi.", "itemId": item_id}
    except HTTPException as http_exc: raise http_exc
    except Exception as e:
        logger.error(f"❌ Menüye ürün eklenirken beklenmedik genel hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Menüye ürün eklenirken sunucuda bir hata oluştu.")

@app.delete("/menu/sil", tags=["Menü Yönetimi"])
async def delete_menu_item_endpoint(
    urun_adi: str = Query(..., min_length=1, description="Silinecek ürünün tam adı."),
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"🗑️ Menüden ürün silme isteği (Kullanıcı: {current_user.kullanici_adi}): {urun_adi}")
    try:
        async with menu_db.transaction(): # menu_db (aynı PostgreSQL DB'si) kullanılıyor
            # `COLLATE NOCASE` kaldırıldığı için case-sensitive arama yapacak.
            # PostgreSQL için: "SELECT id FROM menu WHERE LOWER(ad) = LOWER(:ad)"
            item_to_delete = await menu_db.fetch_one("SELECT id FROM menu WHERE ad = :ad", {"ad": urun_adi}) # ILIKE veya LOWER() gerekebilir
            if not item_to_delete: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{urun_adi}' adlı ürün menüde bulunamadı.")
            await menu_db.execute("DELETE FROM menu WHERE id = :id", {"id": item_to_delete['id']})
        await update_system_prompt() # Cache'i temizler ve prompt'u günceller
        return {"mesaj": f"'{urun_adi}' ürünü menüden başarıyla silindi."}
    except HTTPException as http_exc: raise http_exc
    except Exception as e:
        logger.error(f"❌ Menüden ürün silinirken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Menüden ürün silinirken bir sunucu hatası oluştu.")

# AI Yanıt
@app.post("/yanitla", tags=["Yapay Zeka"])
async def handle_message_endpoint(request: Request, data: dict = Body(...)):
    user_message = data.get("text", "").strip()
    table_id = data.get("masa", "bilinmiyor") # Bu 'masa' parametresi session için mi, yoksa siparişe eklenecek mi?
    session_id = request.session.get("session_id")
    if not session_id:
        session_id = secrets.token_hex(16)
        request.session["session_id"] = session_id
        request.session["chat_history"] = [] # Her yeni session için geçmişi sıfırla
    chat_history = request.session.get("chat_history", [])
    logger.info(f"💬 AI Yanıt isteği: Masa '{table_id}', Session ID: '{session_id}', Kullanıcı Mesajı: '{user_message}'")
    if not user_message: raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Mesaj boş olamaz.")
    if SYSTEM_PROMPT is None: # Sistem prompt'u yüklenmemişse yükle
        await update_system_prompt()
        if SYSTEM_PROMPT is None: # Hala yüklenememişse hata ver
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="AI asistanı şu anda hazır değil (sistem mesajı eksik).")
    try:
        messages_for_openai = [SYSTEM_PROMPT] + chat_history + [{"role": "user", "content": user_message}]
        response = openai_client.chat.completions.create( model=settings.OPENAI_MODEL, messages=messages_for_openai, temperature=0.3, max_tokens=450) # type: ignore
        ai_reply_content = response.choices[0].message.content
        ai_reply = ai_reply_content.strip() if ai_reply_content else "Üzgünüm, şu anda bir yanıt üretemiyorum."
        # Geçmişi güncelle (sadece son 10 mesajı tutuyoruz)
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": ai_reply})
        request.session["chat_history"] = chat_history[-10:] # Son 5 çift (user+assistant)
        return {"reply": ai_reply, "sessionId": session_id}
    except OpenAIError as e: # OpenAI'ye özel hatalar
        logger.error(f"❌ OpenAI API hatası: {type(e).__name__} - {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"AI servisinden yanıt alınırken bir sorun oluştu: {type(e).__name__}")
    except Exception as e: # Diğer beklenmedik hatalar
        logger.error(f"❌ /yanitla endpoint genel hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Mesajınız işlenirken beklenmedik bir sunucu hatası oluştu.")

# Sesli Yanıt
SUPPORTED_LANGUAGES = {"tr-TR", "en-US", "en-GB", "fr-FR", "de-DE"}
@app.post("/sesli-yanit", tags=["Yapay Zeka"])
async def generate_speech_endpoint(data: SesliYanitData):
    if not tts_client: raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Sesli yanıt servisi şu anda kullanılamıyor.")
    if data.language not in SUPPORTED_LANGUAGES: raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Desteklenmeyen dil: {data.language}.")
    cleaned_text = temizle_emoji(data.text)
    if not cleaned_text.strip(): raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Sese dönüştürülecek geçerli bir metin bulunamadı.")
    try:
        synthesis_input = texttospeech.SynthesisInput(text=cleaned_text)
        # Google TTS için 'tr-TR-Standard-A' gibi bir ses adı belirtilebilir veya boş bırakılabilir.
        # voice_name = "tr-TR-Standard-A" if data.language == "tr-TR" else None
        # Chirp modelleri daha kaliteli, kullanılabilirse:
        voice_name = "tr-TR-Chirp3-HD-Laomedeia" if data.language == "tr-TR" else None # Örnek yüksek kaliteli ses
        voice_params = texttospeech.VoiceSelectionParams(
            language_code=data.language,
            name=voice_name, # Belirli bir ses modeli
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE if data.language == "tr-TR" and voice_name else texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=1.0) # speaking_rate ayarlanabilir
        response_tts = tts_client.synthesize_speech(input=synthesis_input, voice=voice_params, audio_config=audio_config)
        return Response(content=response_tts.audio_content, media_type="audio/mpeg")
    except google_exceptions.GoogleAPIError as e_google:
        detail_msg = f"Google TTS servisinden ses üretilirken bir hata oluştu: {getattr(e_google, 'message', str(e_google))}"
        status_code_tts = status.HTTP_503_SERVICE_UNAVAILABLE
        if "API key not valid" in str(e_google) or "permission" in str(e_google).lower() or "RESOURCE_EXHAUSTED" in str(e_google):
            detail_msg = "Google TTS servisi için kimlik/kota sorunu veya kaynak yetersiz."
        elif "Requested voice not found" in str(e_google) or "Invalid DefaultVoice" in str(e_google): # voice_name ile ilgili hata
            detail_msg = f"İstenen ses modeli ({voice_name}) bulunamadı veya geçersiz."; status_code_tts = status.HTTP_400_BAD_REQUEST
        logger.error(f"❌ Google TTS API hatası: {e_google}", exc_info=True)
        raise HTTPException(status_code=status_code_tts, detail=detail_msg)
    except Exception as e:
        logger.error(f"❌ Sesli yanıt endpoint'inde beklenmedik hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Sesli yanıt oluşturulurken beklenmedik bir sunucu hatası oluştu.")


# Kasa İşlemleri (Tüm kasa işlemleri ana `db` üzerinden yapılacak)
@app.post("/kasa/siparis/{siparis_id}/odendi", tags=["Kasa İşlemleri"])
async def mark_order_as_paid_endpoint(
    siparis_id: int = Path(..., description="Ödendi olarak işaretlenecek siparişin ID'si"),
    odeme_bilgisi: KasaOdemeData = Body(...),
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN, KullaniciRol.KASIYER]))
):
    logger.info(f"💰 Kasa: Sipariş {siparis_id} ödendi olarak işaretleniyor (Kullanıcı: {current_user.kullanici_adi}). Ödeme: {odeme_bilgisi.odeme_yontemi}")
    try:
        async with db.transaction(): # Ana db kullanılıyor
            order_check = await db.fetch_one("SELECT id, masa, durum FROM siparisler WHERE id = :id", {"id": siparis_id})
            if not order_check:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sipariş bulunamadı.")
            if order_check["durum"] == Durum.ODENDI.value:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Sipariş zaten ödendi.")
            if order_check["durum"] == Durum.IPTAL.value:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="İptal edilmiş sipariş ödenemez.")

            updated_order_raw = await db.fetch_one(
                """UPDATE siparisler
                   SET durum = :yeni_durum, odeme_yontemi = :odeme_yontemi
                   WHERE id = :id
                   RETURNING id, masa, durum, sepet, istek, zaman, odeme_yontemi""",
                {
                    "yeni_durum": Durum.ODENDI.value,
                    "odeme_yontemi": odeme_bilgisi.odeme_yontemi,
                    "id": siparis_id
                }
            )
        if not updated_order_raw:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sipariş güncellenemedi.")

        updated_order = dict(updated_order_raw)
        updated_order["sepet"] = json.loads(updated_order.get("sepet", "[]"))
        if isinstance(updated_order.get('zaman'), datetime): # Zamanı ISO formatına çevir
            updated_order['zaman'] = updated_order['zaman'].isoformat()


        notif_data = {**updated_order, "zaman": datetime.now(TR_TZ).isoformat()} # Bildirim için anlık zaman
        notification = {"type": "durum", "data": notif_data}
        await broadcast_message(aktif_mutfak_websocketleri, notification, "Mutfak/Masa")
        await broadcast_message(aktif_admin_websocketleri, notification, "Admin")
        await broadcast_message(aktif_kasa_websocketleri, notification, "Kasa")

        await update_table_status(updated_order["masa"], f"Sipariş {siparis_id} ödendi (by {current_user.kullanici_adi}, Yöntem: {updated_order['odeme_yontemi']})")
        return {"message": f"Sipariş {siparis_id} ödendi.", "data": updated_order}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"❌ Kasa: Sipariş {siparis_id} ödendi olarak işaretlenirken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Sipariş durumu güncellenirken sunucu hatası oluştu.")

@app.get("/kasa/odemeler", tags=["Kasa İşlemleri"])
async def get_payable_orders_endpoint(
    durum: Optional[str] = Query(None),
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN, KullaniciRol.KASIYER]))
):
    logger.info(f"💰 Kasa: Ödeme bekleyen siparişler listeleniyor (Kullanıcı: {current_user.kullanici_adi}, Filtre: {durum}).")
    try:
        base_query_str = "SELECT id, masa, istek, sepet, zaman, durum, odeme_yontemi FROM siparisler WHERE "
        values = {}
        valid_statuses_for_payment = [Durum.HAZIR.value, Durum.BEKLIYOR.value, Durum.HAZIRLANIYOR.value]

        if durum:
            if durum not in valid_statuses_for_payment:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Geçersiz durum filtresi. Kullanılabilecekler: {', '.join(valid_statuses_for_payment)}")
            query = base_query_str + "durum = :durum ORDER BY zaman ASC"
            values = {"durum": durum}
        else:
            # PostgreSQL'de IN (...) için parametreler farklı şekilde ele alınabilir.
            # `databases` kütüphanesi bunu `wherein` ile halleder ya da `ANY(:status_list)` gibi bir yapı kullanılır.
            # Şimdilik placeholder'lı yapı korunuyor, `databases`'in bunu handle etmesi bekleniyor.
            status_placeholders = ", ".join([f":status_{i}" for i in range(len(valid_statuses_for_payment))])
            query = base_query_str + f"durum IN ({status_placeholders}) ORDER BY zaman ASC"
            values = {f"status_{i}": s for i, s in enumerate(valid_statuses_for_payment)}

        orders_raw = await db.fetch_all(query, values) # Ana db kullanılıyor
        orders_data = []
        for row in orders_raw:
            order_dict = dict(row)
            order_dict["sepet"] = json.loads(order_dict.get('sepet','[]'))
            if isinstance(order_dict.get('zaman'), datetime):
                 order_dict['zaman'] = order_dict['zaman'].isoformat()
            orders_data.append(order_dict)
        return {"orders": orders_data}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"❌ Kasa: Ödeme bekleyen siparişler alınırken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Siparişler alınırken bir hata oluştu.")

@app.get("/kasa/masa/{masa_id}/hesap", tags=["Kasa İşlemleri"])
async def get_table_bill_endpoint(
    masa_id: str = Path(...),
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN, KullaniciRol.KASIYER]))
):
    logger.info(f"💰 Kasa: Masa {masa_id} için hesap isteniyor (Kullanıcı: {current_user.kullanici_adi}).")
    try:
        active_payable_statuses = [Durum.BEKLIYOR.value, Durum.HAZIRLANIYOR.value, Durum.HAZIR.value]
        # PostgreSQL için IN clause `databases` tarafından yönetilmeli.
        status_placeholders = ", ".join([f":status_{i}" for i in range(len(active_payable_statuses))])
        values = {f"status_{i}": s_val for i, s_val in enumerate(active_payable_statuses)}
        values["masa_id"] = masa_id

        query = f"SELECT id, masa, istek, sepet, zaman, durum, yanit, odeme_yontemi FROM siparisler WHERE masa = :masa_id AND durum IN ({status_placeholders}) ORDER BY zaman ASC"

        orders_raw = await db.fetch_all(query, values) # Ana db kullanılıyor
        orders_data = []
        toplam_tutar = 0.0
        for row in orders_raw:
            order_dict = dict(row)
            sepet_items = json.loads(order_dict.get('sepet', '[]'))
            order_dict['sepet'] = sepet_items
            if isinstance(order_dict.get('zaman'), datetime): # Zamanı ISO formatına çevir
                 order_dict['zaman'] = order_dict['zaman'].isoformat()
            for item in sepet_items:
                if isinstance(item,dict) and isinstance(item.get('adet',0),(int,float)) and isinstance(item.get('fiyat',0.0),(int,float)):
                    toplam_tutar += item['adet'] * item['fiyat']
            orders_data.append(order_dict)
        return {"masa_id": masa_id, "siparisler": orders_data, "toplam_tutar": round(toplam_tutar, 2)}
    except Exception as e:
        logger.error(f"❌ Kasa: Masa {masa_id} hesabı alınırken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Masa hesabı alınırken bir hata oluştu.")


# --- KULLANICI YÖNETİMİ ENDPOINT'LERİ (Admin için) ---
# Tüm kullanıcı işlemleri ana `db` üzerinden yapılacak
@app.post("/admin/kullanicilar", response_model=Kullanici, status_code=status.HTTP_201_CREATED, tags=["Kullanıcı Yönetimi"])
async def create_new_user(
    user_data: KullaniciCreate,
    current_admin: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"Admin '{current_admin.kullanici_adi}' yeni kullanıcı oluşturuyor: {user_data.kullanici_adi}, Rol: {user_data.rol}")
    existing_user = await get_user_from_db(user_data.kullanici_adi)
    if existing_user:
        logger.warning(f"Yeni kullanıcı oluşturma hatası: '{user_data.kullanici_adi}' zaten mevcut.")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Bu kullanıcı adı zaten mevcut.")

    hashed_password = get_password_hash(user_data.sifre)
    # `olusturulma_tarihi` PostgreSQL'de DEFAULT CURRENT_TIMESTAMP ile otomatik ayarlanacak.
    query = """
        INSERT INTO kullanicilar (kullanici_adi, sifre_hash, rol, aktif_mi)
        VALUES (:kullanici_adi, :sifre_hash, :rol, :aktif_mi)
        RETURNING id, kullanici_adi, rol, aktif_mi
    """
    values = {
        "kullanici_adi": user_data.kullanici_adi,
        "sifre_hash": hashed_password,
        "rol": user_data.rol.value,
        "aktif_mi": user_data.aktif_mi
    }
    try:
        created_user_row = await db.fetch_one(query, values)
        if not created_user_row:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Kullanıcı oluşturulurken bir sorun oluştu (DB).")
        logger.info(f"Kullanıcı '{created_user_row['kullanici_adi']}' başarıyla oluşturuldu (ID: {created_user_row['id']}).")
        return Kullanici(**created_user_row)
    except Exception as e: # UNIQUE constraint hatası için özel kontrol gerekebilir
        if "UNIQUE constraint failed" in str(e).lower() or "duplicate key value violates unique constraint" in str(e).lower():
            logger.warning(f"Yeni kullanıcı oluşturma hatası (DB): '{user_data.kullanici_adi}' zaten mevcut.")
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Bu kullanıcı adı veritabanında zaten mevcut.")
        logger.error(f"Yeni kullanıcı ({user_data.kullanici_adi}) DB'ye eklenirken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Kullanıcı veritabanına eklenirken hata: {str(e)}")


@app.get("/admin/kullanicilar", response_model=List[Kullanici], tags=["Kullanıcı Yönetimi"])
async def list_all_users(
    current_admin: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"Admin '{current_admin.kullanici_adi}' tüm kullanıcıları listeliyor.")
    query = "SELECT id, kullanici_adi, rol, aktif_mi FROM kullanicilar ORDER BY id"
    user_rows = await db.fetch_all(query)
    return [Kullanici(**row) for row in user_rows]


@app.put("/admin/kullanicilar/{user_id}", response_model=Kullanici, tags=["Kullanıcı Yönetimi"])
async def update_existing_user(
    user_id: int,
    user_update_data: KullaniciUpdate,
    current_admin: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"Admin '{current_admin.kullanici_adi}', kullanıcı ID {user_id} için güncelleme yapıyor: {user_update_data.model_dump_json(exclude_none=True, exclude_unset=True)}")

    target_user = await db.fetch_one("SELECT id, kullanici_adi, rol, aktif_mi FROM kullanicilar WHERE id = :user_id", {"user_id": user_id})
    if not target_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Güncellenecek kullanıcı bulunamadı.")

    update_fields = {}
    if user_update_data.kullanici_adi is not None and user_update_data.kullanici_adi != target_user["kullanici_adi"]:
        # Yeni kullanıcı adının başka bir kullanıcı tarafından kullanılıp kullanılmadığını kontrol et
        existing_user_with_new_name = await db.fetch_one("SELECT id FROM kullanicilar WHERE kullanici_adi = :k_adi AND id != :u_id", {"k_adi": user_update_data.kullanici_adi, "u_id": user_id})
        if existing_user_with_new_name:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Bu kullanıcı adı zaten başka bir kullanıcı tarafından kullanılıyor.")
        update_fields["kullanici_adi"] = user_update_data.kullanici_adi

    if user_update_data.rol is not None:
        update_fields["rol"] = user_update_data.rol.value

    if user_update_data.aktif_mi is not None:
        update_fields["aktif_mi"] = user_update_data.aktif_mi

    if user_update_data.sifre is not None:
        update_fields["sifre_hash"] = get_password_hash(user_update_data.sifre)

    if not update_fields:
        logger.info(f"Kullanıcı ID {user_id} için güncellenecek bir alan belirtilmedi.")
        return Kullanici(**target_user)

    set_clause_parts = [f"{key} = :{key}" for key in update_fields.keys()]
    set_clause = ", ".join(set_clause_parts)

    query = f"""
        UPDATE kullanicilar
        SET {set_clause}
        WHERE id = :user_id_param
        RETURNING id, kullanici_adi, rol, aktif_mi
    """
    values = {**update_fields, "user_id_param": user_id}

    try:
        updated_user_row = await db.fetch_one(query, values)
        if not updated_user_row: # Normalde RETURNING varsa ve WHERE eşleşirse satır döner.
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Kullanıcı güncellenirken bulunamadı (veya değişiklik olmadı).")

        logger.info(f"Kullanıcı ID {user_id} başarıyla güncellendi. Yeni değerler: {dict(updated_user_row)}")
        return Kullanici(**updated_user_row)
    except Exception as e:
        if "UNIQUE constraint failed" in str(e).lower() or "duplicate key value violates unique constraint" in str(e).lower(): # Kullanıcı adı unique ise
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Bu kullanıcı adı zaten kullanılıyor (güncelleme sırasında).")
        logger.error(f"Kullanıcı ID {user_id} güncellenirken DB hatası: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Kullanıcı güncellenirken veritabanı hatası: {str(e)}")


@app.delete("/admin/kullanicilar/{user_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Kullanıcı Yönetimi"])
async def delete_existing_user(
    user_id: int,
    current_admin: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"Admin '{current_admin.kullanici_adi}', kullanıcı ID {user_id}'yi siliyor.")

    if current_admin.id == user_id:
        logger.warning(f"Admin '{current_admin.kullanici_adi}' kendini silmeye çalıştı.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Admin kendini silemez.")

    user_to_delete = await db.fetch_one("SELECT id FROM kullanicilar WHERE id = :user_id", {"user_id": user_id})
    if not user_to_delete:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Silinecek kullanıcı bulunamadı.")

    try:
        await db.execute("DELETE FROM kullanicilar WHERE id = :user_id", {"user_id": user_id})
        logger.info(f"Kullanıcı ID {user_id} başarıyla silindi.")
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except Exception as e:
        logger.error(f"Kullanıcı ID {user_id} silinirken DB hatası: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Kullanıcı silinirken veritabanı hatası: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    host_ip = os.getenv("HOST", "127.0.0.1") # Render gibi platformlar HOST'u genellikle kendi ayarlar.
    port_num = int(os.getenv("PORT", 8000)) # Render PORT'u otomatik atar.
    logger.info(f"🚀 FastAPI uygulaması {host_ip}:{port_num} adresinde başlatılıyor (yerel geliştirme modu)...")
    # log_config=None yapılabilir, çünkü zaten yukarıda dictConfig ile ayarladık.
    # Uvicorn'un kendi loglamasını da LOGGING_CONFIG'e dahil ettik.
    uvicorn.run("main:app", host=host_ip, port=port_num, reload=True, log_config=LOGGING_CONFIG)