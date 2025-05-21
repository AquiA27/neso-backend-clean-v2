# main.py
from fastapi import (
    FastAPI, Request, Path, Body, Query, HTTPException, status, Depends, WebSocket, WebSocketDisconnect, Response
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
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
# import sqlite3 # Direkt kullanılmayacak, PostgreSQL'e geçildi
import json
import logging
import logging.config
from datetime import datetime, timedelta, date as VeliDate # date için alias
from datetime import timezone as dt_timezone # timezone'u dt_timezone olarak import ettim karışmaması için
from collections import Counter as VeliCounter # Counter için alias
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from google.cloud import texttospeech
from google.api_core import exceptions as google_exceptions
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
    SECRET_KEY: str
    CORS_ALLOWED_ORIGINS: str = "http://localhost:3000,https://neso-guncel.vercel.app"
    DB_DATA_DIR: str = "." # PostgreSQL için doğrudan kullanılmayacak
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    DEFAULT_ADMIN_USERNAME: str = "admin"
    DEFAULT_ADMIN_PASSWORD: str = "ChangeThisDefaultPassword123!"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

try:
    settings = Settings()
    logger.info(f"Ayarlar yüklendi.")
    if settings.DB_DATA_DIR == ".":
        logger.warning("DB_DATA_DIR varsayılan '.' olarak ayarlı.")
except ValueError as e:
    logger.critical(f"❌ Ortam değişkenleri eksik veya hatalı: {e}")
    raise SystemExit(f"Ortam değişkenleri eksik veya hatalı: {e}")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def temizle_emoji(text: Optional[str]) -> str:
    if not isinstance(text, str): return ""
    try:
        emoji_pattern = regex.compile(r"[\p{Emoji_Presentation}\p{Extended_Pictographic}]+", regex.UNICODE)
        return emoji_pattern.sub('', text)
    except Exception as e:
        logger.error(f"Emoji temizleme hatası: {e}")
        return text

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

app = FastAPI(
    title="Neso Sipariş Asistanı API",
    version="1.3.5", # Önceki düzeltmeler ve NameError düzeltmesi
    description="Fıstık Kafe için sipariş backend servisi."
)

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

DATABASE_CONNECTION_STRING = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(settings.DB_DATA_DIR, 'neso_dev_fallback.db')}")
log_db_url = DATABASE_CONNECTION_STRING
if "@" in log_db_url and ":" in log_db_url.split("@")[0]:
    user_pass_part = log_db_url.split("://")[1].split("@")[0]
    host_part = log_db_url.split("@")[1]
    log_db_url = f"{log_db_url.split('://')[0]}://{user_pass_part.split(':')[0]}:********@{host_part}"
logger.info(f"Ana veritabanı bağlantı adresi kullanılıyor: {log_db_url}")

db = Database(DATABASE_CONNECTION_STRING)
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
menu_db = Database(MENU_DATABASE_CONNECTION_STRING) # menu_db ve db aynı PostgreSQL'i işaret ediyor olacak

try:
    if not DATABASE_CONNECTION_STRING.startswith("sqlite:///"):
        logger.info(f"PostgreSQL veya benzeri bir veritabanı kullanılıyor. '{settings.DB_DATA_DIR}' dizini SQLite için oluşturulmayacak.")
    elif settings.DB_DATA_DIR != ".":
        os.makedirs(settings.DB_DATA_DIR, exist_ok=True)
        logger.info(f"SQLite için '{settings.DB_DATA_DIR}' dizini kontrol edildi/oluşturuldu.")
except OSError as e:
    logger.error(f"'{settings.DB_DATA_DIR}' dizini oluşturulurken hata: {e}.")

TR_TZ = dt_timezone(timedelta(hours=3))

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
        from_attributes = True

class Kullanici(KullaniciInDBBase):
    pass

class KullaniciInDB(KullaniciInDBBase):
    sifre_hash: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    kullanici_adi: Union[str, None] = None

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
        # menu_db ve db aynı bağlantıyı kullandığı için, menu_db.connect() tekrar çağrılabilir
        # veya sadece db.is_connected kontrolü sonrası menu_db'nin de bağlı olduğu varsayılabilir.
        # Güvenli olması için, eğer ayrı bir nesne ise ve bağlı değilse bağlayalım.
        if menu_db != db or not menu_db.is_connected: # Eğer farklı nesnelerse VEYA aynı olup bağlı değilse
             await menu_db.connect()
        logger.info("✅ Veritabanı bağlantıları kuruldu.")
        await init_databases()
        await update_system_prompt()
        logger.info(f"🚀 FastAPI uygulaması başlatıldı. Sistem mesajı güncellendi.")
    except Exception as e_startup:
        logger.critical(f"❌ Uygulama başlangıcında KRİTİK HATA: {e_startup}", exc_info=True)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🚪 Uygulama kapatılıyor...")
    try:
        if menu_db.is_connected: await menu_db.disconnect()
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
        if e.code == 1000 or e.code == 1001: # Normal kapanış veya endpoint'in gidişi
            logger.info(f"🔌 {endpoint_name} WS normal şekilde kapandı (Kod {e.code}): {client_info}")
        elif e.code == 1012: # Service Restart
            logger.warning(f"🔌 {endpoint_name} WS beklenmedik şekilde kapandı (Kod {e.code} - Sunucu Yeniden Başlıyor Olabilir): {client_info}")
        else: # Diğer beklenmedik kapanışlar
            logger.warning(f"🔌 {endpoint_name} WS beklenmedik şekilde kapandı (Kod {e.code}): {client_info}")
    except Exception as e_outer: # Diğer tüm hatalar
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

async def update_table_status(masa_id: str, islem: str = "Erişim"):
    now = datetime.now(TR_TZ)
    try:
        await db.execute("""
            INSERT INTO masa_durumlar (masa_id, son_erisim, aktif, son_islem)
            VALUES (:masa_id, :son_erisim, TRUE, :islem)
            ON CONFLICT(masa_id) DO UPDATE SET
                son_erisim = excluded.son_erisim,
                aktif = excluded.aktif,
                son_islem = excluded.son_islem
        """, {"masa_id": masa_id, "son_erisim": now, "islem": islem})
        await broadcast_message(aktif_admin_websocketleri, {
            "type": "masa_durum",
            "data": {"masaId": masa_id, "sonErisim": now.isoformat(), "aktif": True, "sonIslem": islem}
        }, "Admin")
    except Exception as e:
        logger.error(f"❌ Masa durumu ({masa_id}) güncelleme hatası: {e}")

@app.middleware("http")
async def track_active_users(request: Request, call_next):
    masa_id_param = request.path_params.get("masaId")
    masa_id_query = request.query_params.get("masa_id")
    masa_id = masa_id_param or masa_id_query
    if masa_id:
        endpoint_name = request.scope.get("endpoint", {}).__name__ if request.scope.get("endpoint") else request.url.path
        await update_table_status(str(masa_id), f"{request.method} {endpoint_name}")
    try:
        response = await call_next(request)
        return response
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception(f"❌ HTTP Middleware genel hata ({request.url.path}): {e}")
        return Response("Sunucuda bir hata oluştu.", status_code=500, media_type="text/plain")

@app.get("/ping")
async def ping_endpoint():
    logger.info("📢 /ping endpoint'ine istek geldi!")
    return {"message": "Neso backend pong! Service is running."}

@app.post("/token", response_model=Token, tags=["Kimlik Doğrulama"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    logger.info(f"Giriş denemesi: Kullanıcı adı '{form_data.username}'")
    user_in_db = await get_user_from_db(username=form_data.username)
    if not user_in_db or not verify_password(form_data.password, user_in_db.sifre_hash):
        logger.warning(f"Başarısız giriş: Kullanıcı '{form_data.username}' için geçersiz kimlik bilgileri.")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Yanlış kullanıcı adı veya şifre", headers={"WWW-Authenticate": "Bearer"})
    if not user_in_db.aktif_mi:
        logger.warning(f"Pasif kullanıcı '{form_data.username}' giriş yapmaya çalıştı.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Hesabınız aktif değil. Lütfen yönetici ile iletişime geçin.")
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user_in_db.kullanici_adi}, expires_delta=access_token_expires)
    logger.info(f"Kullanıcı '{user_in_db.kullanici_adi}' (Rol: {user_in_db.rol}) başarıyla giriş yaptı. Token oluşturuldu.")
    return {"access_token": access_token, "token_type": "bearer"}

class Durum(str, Enum):
    BEKLIYOR = "bekliyor"
    HAZIRLANIYOR = "hazirlaniyor"
    HAZIR = "hazir"
    IPTAL = "iptal"
    ODENDI = "odendi"

class SepetItem(BaseModel):
    urun: str = Field(..., min_length=1)
    adet: int = Field(..., gt=0)
    fiyat: float = Field(..., ge=0)
    kategori: Optional[str] = Field(None)

class SiparisEkleData(BaseModel):
    masa: str = Field(..., min_length=1)
    sepet: List[SepetItem] = Field(..., min_items=1)
    istek: Optional[str] = None
    yanit: Optional[str] = None

class SiparisGuncelleData(BaseModel):
    durum: Durum

class AktifMasaOzet(BaseModel):
    masa_id: str
    odenmemis_tutar: float
    aktif_siparis_sayisi: int
    siparis_detaylari: Optional[List[Dict]] = None

class KasaOdemeData(BaseModel):
    odeme_yontemi: str

class MenuEkleData(BaseModel):
    ad: str = Field(..., min_length=1)
    fiyat: float = Field(..., gt=0)
    kategori: str = Field(..., min_length=1)

class SesliYanitData(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = Field(default="tr-TR", pattern=r"^[a-z]{2}-[A-Z]{2}$")

class IstatistikBase(BaseModel):
    siparis_sayisi: int
    toplam_gelir: float
    satilan_urun_adedi: int

class GunlukIstatistik(IstatistikBase):
    tarih: str

class AylikIstatistik(IstatistikBase):
    ay: str

class YillikAylikKirilimDetay(BaseModel):
    toplam_gelir: float
    satilan_urun_adedi: int

class YillikAylikKirilimResponse(BaseModel):
    aylik_kirilim: Dict[str, YillikAylikKirilimDetay]

class EnCokSatilanUrun(BaseModel):
    urun: str
    adet: int

@app.get("/users/me", response_model=Kullanici, tags=["Kullanıcılar"])
async def read_users_me(current_user: Kullanici = Depends(get_current_active_user)):
    logger.info(f"Kullanıcı '{current_user.kullanici_adi}' kendi bilgilerini istedi.")
    return current_user

@app.get("/aktif-masalar/ws-count", tags=["Admin İşlemleri"])
async def get_active_tables_ws_count_endpoint(
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"Admin '{current_user.kullanici_adi}' aktif WS masa sayısını istedi.")
    return {"aktif_mutfak_ws_sayisi": len(aktif_mutfak_websocketleri),
            "aktif_admin_ws_sayisi": len(aktif_admin_websocketleri),
            "aktif_kasa_ws_sayisi": len(aktif_kasa_websocketleri)}

@app.get("/istatistik/gunluk", response_model=GunlukIstatistik, tags=["İstatistikler"])
async def get_gunluk_istatistik(
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN])),
    tarih_str: Optional[str] = Query(None, description="YYYY-MM-DD formatında tarih. Boş bırakılırsa bugün alınır.")
):
    logger.info(f"Admin '{current_user.kullanici_adi}' günlük istatistikleri istedi (Tarih: {tarih_str or 'Bugün'}).")
    try:
        if tarih_str:
            try:
                gun_baslangic_dt = datetime.strptime(tarih_str, "%Y-%m-%d").replace(tzinfo=TR_TZ)
            except ValueError:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Geçersiz tarih formatı. YYYY-MM-DD kullanın.")
        else:
            gun_baslangic_dt = datetime.now(TR_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
        gun_bitis_dt = gun_baslangic_dt + timedelta(days=1)
        query = """
            SELECT sepet, durum FROM siparisler
            WHERE zaman >= :baslangic AND zaman < :bitis AND durum = 'odendi'
        """
        odenen_siparisler = await db.fetch_all(query, {"baslangic": gun_baslangic_dt, "bitis": gun_bitis_dt})
        siparis_sayisi = len(odenen_siparisler)
        toplam_gelir = 0.0
        satilan_urun_adedi = 0
        for siparis in odenen_siparisler:
            try:
                sepet_items = json.loads(siparis["sepet"] or "[]")
                for item in sepet_items:
                    adet = item.get("adet", 0)
                    fiyat = item.get("fiyat", 0.0)
                    toplam_gelir += adet * fiyat
                    satilan_urun_adedi += adet
            except json.JSONDecodeError:
                logger.warning(f"Günlük istatistik: Sepet parse hatası, Sipariş durumu: {siparis['durum']}, Sepet: {siparis['sepet']}")
                continue
        return GunlukIstatistik(
            tarih=gun_baslangic_dt.strftime("%Y-%m-%d"),
            siparis_sayisi=siparis_sayisi,
            toplam_gelir=round(toplam_gelir, 2),
            satilan_urun_adedi=satilan_urun_adedi
        )
    except Exception as e:
        logger.error(f"❌ Günlük istatistik alınırken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Günlük istatistikler alınırken bir sorun oluştu.")

@app.get("/istatistik/aylik", response_model=AylikIstatistik, tags=["İstatistikler"])
async def get_aylik_istatistik(
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN])),
    yil: Optional[int] = Query(None, description="YYYY formatında yıl. Boş bırakılırsa bu yıl alınır."),
    ay: Optional[int] = Query(None, description="1-12 arası ay. Boş bırakılırsa bu ay alınır.")
):
    logger.info(f"Admin '{current_user.kullanici_adi}' aylık istatistikleri istedi (Yıl: {yil or 'Bu Yıl'}, Ay: {ay or 'Bu Ay'}).")
    try:
        simdi_tr = datetime.now(TR_TZ)
        target_yil = yil if yil else simdi_tr.year
        target_ay = ay if ay else simdi_tr.month
        if not (1 <= target_ay <= 12):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Geçersiz ay değeri. 1-12 arasında olmalıdır.")
        ay_baslangic_dt = datetime(target_yil, target_ay, 1, tzinfo=TR_TZ)
        if target_ay == 12:
            ay_bitis_dt = datetime(target_yil + 1, 1, 1, tzinfo=TR_TZ)
        else:
            ay_bitis_dt = datetime(target_yil, target_ay + 1, 1, tzinfo=TR_TZ)
        query = """
            SELECT sepet, durum FROM siparisler
            WHERE zaman >= :baslangic AND zaman < :bitis AND durum = 'odendi'
        """
        odenen_siparisler = await db.fetch_all(query, {"baslangic": ay_baslangic_dt, "bitis": ay_bitis_dt})
        siparis_sayisi = len(odenen_siparisler)
        toplam_gelir = 0.0
        satilan_urun_adedi = 0
        for siparis in odenen_siparisler:
            try:
                sepet_items = json.loads(siparis["sepet"] or "[]")
                for item in sepet_items:
                    adet = item.get("adet", 0)
                    fiyat = item.get("fiyat", 0.0)
                    toplam_gelir += adet * fiyat
                    satilan_urun_adedi += adet
            except json.JSONDecodeError:
                logger.warning(f"Aylık istatistik: Sepet parse hatası, Sipariş durumu: {siparis['durum']}, Sepet: {siparis['sepet']}")
                continue
        return AylikIstatistik(
            ay=ay_baslangic_dt.strftime("%Y-%m"),
            siparis_sayisi=siparis_sayisi,
            toplam_gelir=round(toplam_gelir, 2),
            satilan_urun_adedi=satilan_urun_adedi
        )
    except Exception as e:
        logger.error(f"❌ Aylık istatistik alınırken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Aylık istatistikler alınırken bir sorun oluştu.")

@app.get("/istatistik/yillik-aylik-kirilim", response_model=YillikAylikKirilimResponse, tags=["İstatistikler"])
async def get_yillik_aylik_kirilim(
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN])),
    yil: Optional[int] = Query(None, description="YYYY formatında yıl. Boş bırakılırsa bu yılın verileri getirilir.")
):
    logger.info(f"Admin '{current_user.kullanici_adi}' yıllık aylık kırılım istatistiklerini istedi (Yıl: {yil or 'Bu Yıl'}).")
    try:
        target_yil = yil if yil else datetime.now(TR_TZ).year
        yil_baslangic_tr = datetime(target_yil, 1, 1, tzinfo=TR_TZ)
        yil_bitis_tr = datetime(target_yil + 1, 1, 1, tzinfo=TR_TZ)
        query_all_year = """
            SELECT sepet, zaman FROM siparisler
            WHERE durum = 'odendi' AND zaman >= :baslangic AND zaman < :bitis
        """
        odenen_siparisler_yil = await db.fetch_all(query_all_year, {"baslangic": yil_baslangic_tr, "bitis": yil_bitis_tr})
        aylik_kirilim_data: Dict[str, Dict[str, Any]] = {}
        for siparis in odenen_siparisler_yil:
            siparis_zamani = siparis["zaman"]
            if siparis_zamani.tzinfo is None:
                siparis_zamani = siparis_zamani.replace(tzinfo=TR_TZ)
            else:
                siparis_zamani = siparis_zamani.astimezone(TR_TZ)
            ay_key = siparis_zamani.strftime("%Y-%m")
            if ay_key not in aylik_kirilim_data:
                aylik_kirilim_data[ay_key] = {"toplam_gelir": 0.0, "satilan_urun_adedi": 0}
            try:
                sepet_items = json.loads(siparis["sepet"] or "[]")
                for item in sepet_items:
                    adet = item.get("adet", 0)
                    fiyat = item.get("fiyat", 0.0)
                    aylik_kirilim_data[ay_key]["toplam_gelir"] += adet * fiyat
                    aylik_kirilim_data[ay_key]["satilan_urun_adedi"] += adet
            except json.JSONDecodeError:
                logger.warning(f"Yıllık kırılım: Sepet parse hatası, Sipariş zamanı: {siparis['zaman']}, Sepet: {siparis['sepet']}")
                continue
        response_data = {
            key: YillikAylikKirilimDetay(**value)
            for key, value in aylik_kirilim_data.items()
        }
        return YillikAylikKirilimResponse(aylik_kirilim=response_data)
    except Exception as e:
        logger.error(f"❌ Yıllık aylık kırılım istatistikleri alınırken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Yıllık istatistikler alınırken bir sorun oluştu.")

@app.get("/istatistik/en-cok-satilan", response_model=List[EnCokSatilanUrun], tags=["İstatistikler"])
async def get_en_cok_satilan_urunler(
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN])),
    limit: int = Query(5, ge=1, le=20, description="Listelenecek ürün sayısı")
):
    logger.info(f"Admin '{current_user.kullanici_adi}' en çok satılan {limit} ürünü istedi.")
    try:
        query = "SELECT sepet FROM siparisler WHERE durum = 'odendi'"
        odenen_siparisler = await db.fetch_all(query)
        urun_sayaclari = VeliCounter()
        for siparis in odenen_siparisler:
            try:
                sepet_items = json.loads(siparis["sepet"] or "[]")
                for item in sepet_items:
                    urun_adi = item.get("urun")
                    adet = item.get("adet", 0)
                    if urun_adi and adet > 0:
                        urun_sayaclari[urun_adi] += adet
            except json.JSONDecodeError:
                logger.warning(f"En çok satılan: Sepet parse hatası, Sepet: {siparis['sepet']}")
                continue
        en_cok_satilanlar = [
            EnCokSatilanUrun(urun=item[0], adet=item[1])
            for item in urun_sayaclari.most_common(limit)
        ]
        return en_cok_satilanlar
    except Exception as e:
        logger.error(f"❌ En çok satılan ürünler alınırken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="En çok satılan ürünler alınırken bir sorun oluştu.")

@app.get("/admin/aktif-masa-tutarlari", response_model=List[AktifMasaOzet], tags=["Admin İşlemleri"])
async def get_aktif_masa_tutarlari(
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"Admin '{current_user.kullanici_adi}' aktif masa tutarlarını istedi.")
    try:
        odenmemis_durumlar = [Durum.BEKLIYOR.value, Durum.HAZIRLANIYOR.value, Durum.HAZIR.value]

        # === DÜZELTİLMİŞ SORGU ===
        # PostgreSQL'de bir diziye karşı IN kontrolü için ANY kullanılır.
        # :statuses_list placeholder'ına Python listesi/tuple'ı verilecek.
        query_str = "SELECT masa, sepet FROM siparisler WHERE durum = ANY(:statuses_list)"
        values = {"statuses_list": odenmemis_durumlar} # tuple() yapmaya gerek yok, asyncpg listeyi de anlar.
        # ==========================

        aktif_siparisler = await db.fetch_all(query=query_str, values=values)

        masa_ozetleri: Dict[str, Dict[str, Any]] = {}
        for siparis in aktif_siparisler:
            masa_id = siparis["masa"]
            if masa_id not in masa_ozetleri:
                masa_ozetleri[masa_id] = {"aktif_siparis_sayisi": 0, "odenmemis_tutar": 0.0}
            masa_ozetleri[masa_id]["aktif_siparis_sayisi"] += 1
            try:
                sepet_items = json.loads(siparis["sepet"] or "[]")
                for item in sepet_items:
                    adet = item.get("adet", 0)
                    fiyat = item.get("fiyat", 0.0)
                    masa_ozetleri[masa_id]["odenmemis_tutar"] += adet * fiyat
            except json.JSONDecodeError:
                logger.warning(f"Aktif masalar: Sepet parse hatası, Masa: {masa_id}, Sepet: {siparis['sepet']}")
                continue

        response_list = [
            AktifMasaOzet(
                masa_id=masa,
                aktif_siparis_sayisi=data["aktif_siparis_sayisi"],
                odenmemis_tutar=round(data["odenmemis_tutar"], 2)
            ) for masa, data in masa_ozetleri.items()
        ]
        return response_list

    except Exception as e:
        logger.error(f"❌ Aktif masa tutarları alınırken hata: {e}", exc_info=True) # exc_info=True önemli
        # Frontend'in AxiosError'dan alacağı mesaj için:
        if isinstance(e, google_exceptions.PostgresSyntaxError): # asyncpg.exceptions.PostgresSyntaxError olacak
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Veritabanı sorgu hatası: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Aktif masa tutarları alınırken bir sorun oluştu.")

@app.patch("/siparis/{id}", tags=["Siparişler"])
async def patch_order_endpoint(
    id: int = Path(..., description="Güncellenecek siparişin ID'si"),
    data: SiparisGuncelleData = Body(...),
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN, KullaniciRol.MUTFAK_PERSONELI, KullaniciRol.BARISTA]))
):
    logger.info(f"🔧 PATCH /siparis/{id} ile durum güncelleme isteği (Kullanıcı: {current_user.kullanici_adi}, Rol: {current_user.rol}): {data.durum}")
    try:
        async with db.transaction():
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
        if isinstance(updated_order.get('zaman'), datetime):
             updated_order['zaman'] = updated_order['zaman'].isoformat()
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
    row = await db.fetch_one("SELECT zaman, masa, durum, odeme_yontemi FROM siparisler WHERE id = :id", {"id": id})
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sipariş bulunamadı.")
    if row["durum"] == Durum.IPTAL.value:
        return {"message": f"Sipariş {id} zaten iptal edilmiş."}
    try:
        async with db.transaction():
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
    olusturma_zamani = order_details["zaman"]
    if isinstance(olusturma_zamani, str):
        try:
            olusturma_zamani_dt = datetime.fromisoformat(olusturma_zamani)
        except ValueError:
            olusturma_zamani_dt = datetime.strptime(olusturma_zamani, "%Y-%m-%d %H:%M:%S").replace(tzinfo=TR_TZ)
    else:
        olusturma_zamani_dt = olusturma_zamani
    if olusturma_zamani_dt.tzinfo is None:
        olusturma_zamani_dt = olusturma_zamani_dt.replace(tzinfo=TR_TZ)
    else:
        olusturma_zamani_dt = olusturma_zamani_dt.astimezone(TR_TZ)
    if datetime.now(TR_TZ) - olusturma_zamani_dt > timedelta(minutes=2):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Bu sipariş 2 dakikayı geçtiği için artık iptal edilemez.")
    try:
        async with db.transaction():
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

# --- Menü Yönetimi (Fonksiyonlar) - Tanımlar buraya taşındı ---
@alru_cache(maxsize=1)
async def get_menu_price_dict() -> Dict[str, float]:
    logger.info(">>> get_menu_price_dict ÇAĞRILIYOR...")
    try:
        if not menu_db.is_connected: await menu_db.connect()
        prices_raw = await menu_db.fetch_all("SELECT ad, fiyat FROM menu")
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

@alru_cache(maxsize=1)
async def get_menu_for_prompt_cached() -> str:
    logger.info(">>> GET_MENU_FOR_PROMPT_CACHED ÇAĞRILIYOR (Fiyatlar Dahil Edilecek)...")
    try:
        if not menu_db.is_connected:
            await menu_db.connect()
            # logger.info("get_menu_for_prompt_cached içinde menu_db bağlantısı kuruldu.")
        query = """
            SELECT k.isim as kategori_isim, m.ad as urun_ad, m.fiyat as urun_fiyat
            FROM menu m
            JOIN kategoriler k ON m.kategori_id = k.id
            WHERE m.stok_durumu = 1
            ORDER BY k.isim, m.ad
        """
        urunler_raw = await menu_db.fetch_all(query)
        if not urunler_raw:
            return "Üzgünüz, şu anda menümüzde aktif ürün bulunmamaktadır."
        kategorili_menu: Dict[str, List[str]] = {}
        for row in urunler_raw:
            try:
                urun_adi = row['urun_ad']
                urun_fiyati_str = f"{float(row['urun_fiyat']):.2f} TL"
                kategori_ismi = row['kategori_isim']
                kategorili_menu.setdefault(kategori_ismi, []).append(f"{urun_adi} ({urun_fiyati_str})")
            except Exception as e_row:
                logger.error(f"get_menu_for_prompt_cached (fiyatlı): Satır işlenirken hata: {e_row} - Satır: {row}", exc_info=True)
        if not kategorili_menu:
            return "Üzgünüz, menü bilgisi şu anda düzgün bir şekilde formatlanamıyor."
        menu_aciklama_list = [
            f"- {kategori}: {', '.join(urun_listesi_detayli)}"
            for kategori, urun_listesi_detayli in kategorili_menu.items() if urun_listesi_detayli
        ]
        if not menu_aciklama_list:
            return "Üzgünüz, menüde listelenecek ürün bulunamadı."
        logger.info(f"Menü (fiyatlar dahil) prompt için başarıyla oluşturuldu ({len(kategorili_menu)} kategori).")
        return "\n".join(menu_aciklama_list)
    except Exception as e:
        logger.error(f"❌ Menü (fiyatlar dahil) prompt oluşturma hatası: {e}", exc_info=True)
        return "Teknik bir sorun nedeniyle menü bilgisine ve fiyatlara şu anda ulaşılamıyor. Lütfen daha sonra tekrar deneyin veya personelden yardım isteyin."
# --- Menü Yönetimi Fonksiyonları SONU ---

@app.post("/siparis-ekle", status_code=status.HTTP_201_CREATED, tags=["Müşteri İşlemleri"])
async def add_order_endpoint(data: SiparisEkleData):
    masa = data.masa
    sepet = data.sepet
    istek = data.istek
    yanit = data.yanit
    simdiki_zaman_obj = datetime.now(TR_TZ)
    db_zaman_kayit = simdiki_zaman_obj
    yanit_zaman_iso_str = simdiki_zaman_obj.isoformat()
    logger.info(f"📥 Yeni sipariş isteği alındı: Masa {masa}, {len(sepet)} çeşit ürün. AI Yanıtı: {yanit[:200] if yanit else 'Yok'}...")
    
    # Fonksiyonların doğru tanımlandığından ve çağrıldığından emin oluyoruz.
    cached_price_dict = await get_menu_price_dict()
    cached_stock_dict = await get_menu_stock_dict()
    
    processed_sepet = []
    for item in sepet:
        urun_adi_lower = item.urun.lower().strip()
        stok_kontrol_degeri = cached_stock_dict.get(urun_adi_lower)
        if stok_kontrol_degeri is None or stok_kontrol_degeri == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"'{item.urun}' adlı ürün stokta yok veya menüde bulunmuyor.")
        item_dict = item.model_dump()
        cached_fiyat = cached_price_dict.get(urun_adi_lower, item.fiyat)
        if cached_fiyat != item.fiyat: logger.warning(f"Fiyat uyuşmazlığı: Ürün '{item.urun}', Frontend Fiyatı: {item.fiyat}, Cache Fiyatı: {cached_fiyat}. Cache fiyatı kullanılacak.")
        item_dict['fiyat'] = cached_fiyat
        processed_sepet.append(item_dict)
    if not processed_sepet: raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Sepette geçerli ürün yok.")
    istek_ozet = ", ".join([f"{p_item['adet']}x {p_item['urun']}" for p_item in processed_sepet])
    try:
        async with db.transaction():
            siparis_id = await db.fetch_val("""
                INSERT INTO siparisler (masa, istek, yanit, sepet, zaman, durum, odeme_yontemi)
                VALUES (:masa, :istek, :yanit, :sepet, :zaman, 'bekliyor', NULL)
                RETURNING id
            """, { "masa": masa, "istek": istek or istek_ozet, "yanit": yanit, "sepet": json.dumps(processed_sepet, ensure_ascii=False), "zaman": db_zaman_kayit })
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

@app.post("/siparis-guncelle", tags=["Siparişler"])
async def update_order_status_endpoint(
    data: SiparisGuncelleData,
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN, KullaniciRol.MUTFAK_PERSONELI, KullaniciRol.BARISTA]))
):
    raise HTTPException(status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail="/siparis/{id} (PATCH) endpoint'ini kullanın.")

@app.get("/siparisler", tags=["Siparişler"])
async def get_orders_endpoint(
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN, KullaniciRol.KASIYER, KullaniciRol.MUTFAK_PERSONELI, KullaniciRol.BARISTA]))
):
    logger.info(f"📋 Tüm siparişler listeleniyor (Kullanıcı: {current_user.kullanici_adi})")
    try:
        orders_raw = await db.fetch_all("SELECT id, masa, istek, yanit, sepet, zaman, durum, odeme_yontemi FROM siparisler ORDER BY id DESC")
        orders_data = []
        for row in orders_raw:
            order_dict = dict(row)
            try:
                sepet_str = order_dict.get('sepet')
                order_dict['sepet'] = json.loads(sepet_str if sepet_str else '[]')
            except json.JSONDecodeError:
                order_dict['sepet'] = []
            if isinstance(order_dict.get('zaman'), datetime):
                 order_dict['zaman'] = order_dict['zaman'].isoformat()
            orders_data.append(order_dict)
        return {"orders": orders_data}
    except Exception as e:
        logger.error(f"❌ Tüm siparişler alınırken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Siparişler veritabanından alınırken bir sorun oluştu.")

async def init_db():
    logger.info(f"Ana veritabanı tabloları kontrol ediliyor/oluşturuluyor...")
    try:
        async with db.transaction():
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
                    {"kullanici_adi": settings.DEFAULT_ADMIN_USERNAME, "sifre_hash": hashed_password, "rol": KullaniciRol.ADMIN.value}
                )
                logger.info(f"Varsayılan admin kullanıcısı '{settings.DEFAULT_ADMIN_USERNAME}' veritabanına eklendi.")
            else:
                logger.info(f"Varsayılan admin kullanıcısı '{settings.DEFAULT_ADMIN_USERNAME}' zaten mevcut.")
        logger.info(f"✅ Ana veritabanı tabloları başarıyla doğrulandı/oluşturuldu.")
    except Exception as e:
        logger.critical(f"❌ Ana veritabanı tabloları başlatılırken kritik hata: {e}", exc_info=True)
        raise

async def init_menu_db():
    logger.info(f"Menü veritabanı tabloları kontrol ediliyor/oluşturuluyor...")
    try:
        async with menu_db.transaction(): # menu_db de aynı PostgreSQL'i kullanıyor
            await menu_db.execute("""
                CREATE TABLE IF NOT EXISTS kategoriler (
                    id SERIAL PRIMARY KEY,
                    isim TEXT UNIQUE NOT NULL
                )""")
            await menu_db.execute("""
                CREATE TABLE IF NOT EXISTS menu (
                    id SERIAL PRIMARY KEY,
                    ad TEXT NOT NULL,
                    fiyat REAL NOT NULL CHECK(fiyat >= 0),
                    kategori_id INTEGER NOT NULL,
                    stok_durumu INTEGER DEFAULT 1,
                    FOREIGN KEY (kategori_id) REFERENCES kategoriler(id) ON DELETE CASCADE,
                    UNIQUE(ad, kategori_id)
                )""")
            await menu_db.execute("CREATE INDEX IF NOT EXISTS idx_menu_kategori_id ON menu(kategori_id)")
            await menu_db.execute("CREATE INDEX IF NOT EXISTS idx_menu_ad ON menu(ad)")
        logger.info(f"✅ Menü veritabanı tabloları başarıyla doğrulandı/oluşturuldu.")
    except Exception as e:
        logger.critical(f"❌ Menü veritabanı tabloları başlatılırken kritik hata: {e}", exc_info=True)
        raise

async def init_databases():
    await init_db()
    await init_menu_db()

SISTEM_MESAJI_ICERIK_TEMPLATE = (
    "Sen Fıstık Kafe için Neso adında, çok yetenekli, kibar ve hafif espirili bir sipariş asistanısın. "
    "Fıstık Kafe, ikinci nesil kahveler, özel çaylar, sıcak ve soğuk içecekler ile lezzetli atıştırmalıklar (kek, kurabiye vb.) sunan bir mekandır; KAFEDE YEMEK (pizza, kebap, ana yemek vb.) SERVİSİ BULUNMAMAKTADIR. "
    "Görevin, müşterilerin taleplerini doğru anlayıp, SANA VERİLEN STOKTAKİ ÜRÜNLER LİSTESİNDE yer alan ürünlerle eşleştirerek siparişlerini JSON formatında hazırlamak ve kafe deneyimini keyifli hale getirmektir. "
    "Müşterilerin ruh haline, bağlama (ör. hava durumu) ve yöresel dillere duyarlı ol.\n\n"
    "# LANGUAGE DETECTION & RESPONSE\n"
    "1. Müşterinin kullandığı dili otomatik olarak algıla ve tüm metin yanıtlarını aynı dilde üret. "
    "Desteklediğin diller: Türkçe, English, العربية, Deutsch, Français, Español vb.\n"
    "2. İlk karşılamada ve hatırlatmalarda nazik, hafif espirili bir üslup kullan:\n"
    "   - Türkçe: \"Merhaba, ben Neso! Fıstık Kafe’de sana enfes bir deneyim yaşatmak için buradayım, ne sipariş edelim?\"\n"
    "   - English: \"Hello, I’m Neso! Ready to make your time at Fıstık Kafe delightful. What can I get started for you?\"\n\n"
    "# STOKTAKİ ÜRÜNLER\n"
    "STOKTAKİ ÜRÜNLERİN TAM LİSTESİ (KATEGORİ: ÜRÜNLER VE FİYATLARI) - Fıstık Kafe sadece içecek ve hafif atıştırmalıklar sunar:\n"
    "{menu_prompt_data}\n"
    "# ÖNEMLİ NOT: Buraya enjekte edilen {menu_prompt_data} içeriğinin güncel ve doğru olduğundan emin ol. Örneklerdeki ürünler de bu listede VAR OLMALIDIR veya örnekler menüde olmayan ürün senaryosunu doğru işlemelidir.\n\n"
    "# ÖNEMLİ KURALLAR\n\n"
    "## Genel Sipariş Kuralları:\n"
    "1. SADECE yukarıdaki STOKTAKİ ÜRÜNLER listesinde açıkça belirtilen ürünleri ve onların özelliklerini kabul et. Listelenen tüm ürünler stoktadır.\n"
    "2. Ürün adı tam eşleşmese bile (anlamsal olarak %75+ benzerlik varsa) STOKTAKİ ÜRÜNLER listesindeki en yakın ürünü seç. "
    "Müşterinin belirttiği ek özellikleri (örn: sade, şekerli, duble, yanında süt vb.) ilgili ürünün “musteri_notu” alanına ekle.\n"
    "   ÖRNEK: “2 sade türk kahvesi, 1 şekerli” -> Bu durumda kahveleri ayrı JSON kalemleri olarak işle (birini 'sade', diğerini 'şekerli' notuyla).\n"
    "2.1. **Karma İstekler (Menüde Olan ve Olmayan):** Eğer müşteri hem menüde olan bir içecek/atıştırmalık hem de menüde olmayan bir YEMEK türü isterse (örn: 'Bir latte ve bir de Adana Kebap'), menüde olanları sepete ekle. `konusma_metni` içinde hem onayladığın ürünleri belirt hem de menüde olmayan YEMEK için 'Fıstık Kafe'de yemek servisimiz bulunmuyor' şeklinde bilgilendirme yapıp Fıstık Kafe'ye uygun bir kahve/içecek/atıştırmalık alternatifi öner.\n"
    "3. Yöresel ifadeleri (“rafık”, “baa”, “kurban olim” gibi) veya argoyu görmezden gelerek asıl sipariş niyetine odaklan.\n"
    "4. Birden fazla ürün siparişinde, her birinin özelliklerini ve adetlerini ayrı ayrı JSON kalemleri olarak işle.\n"
    "5. Belirtilmeyen özellikler için (eğer varsa) STOKTAKİ ÜRÜNLER listesinde belirtilen varsayılanları kullan veya genel kabul görmüş standartları (örn. Türk kahvesi için 'orta şekerli', Çay için 'normal dem') uygula. Eğer bir varsayılan yoksa ve özellik önemliyse (örn. kahve çekirdeği türü), müşteriye sorarak netleştir (Kural 11).\n" 
    "6. Fiyat ve kategori bilgilerini HER ZAMAN STOKTAKİ ÜRÜNLER listesinden al, asla tahmin etme veya uydurma yapma. Birim fiyatları kullan.\n"
    "7. Siparişteki her bir ürün için toplam tutarı (adet × birim_fiyat) doğru hesapla ve tüm siparişin genel `toplam_tutar`ını oluştur.\n\n"
    "## Soru Sorma, Öneri İstekleri ve Menüde Olmayan Ürünlerin Ele Alınması:\n"
    "8. **Menüde Olmayan Ürün:** Müşteri STOKTAKİ ÜRÜNLER listesinde olmayan bir ürün (özellikle YEMEK türü) isterse VEYA bir ürünün menüde olup olmadığı sorulur VE BU ÜRÜN LİSTEDE YOKSA, kesinlikle 'menüde var' YANITI VERME. JSON `sepet` alanını boş liste `[]` olarak, `toplam_tutar`ı `0.0` olarak ayarla ve `konusma_metni` alanında nazikçe ürünün menüde bulunmadığını (eğer yemekse 'Fıstık Kafe'de yemek servisimiz bulunmuyor' şeklinde) bildir. Ardından, **Fıstık Kafe konseptine uygun (kahve, çay, soğuk içecek, tatlı/atıştırmalık) bir alternatif sunmayı TEKLİF ET.**\n"
    "   ÖRNEK (Menüde Olmayan YEMEK İsteği): Kullanıcı: “Pizza alabilir miyim?” -> `konusma_metni`: “Maalesef Fıstık Kafe'de pizza gibi yemek çeşitlerimiz bulunmuyor. Size bunun yerine özel demleme bir kahvemizi veya taptaze bir dilim kekimizi önerebilirim. Ne dersiniz?”\n"
    "   ÖRNEK (Menüde Olmayan İçecek Sorgusu): Kullanıcı: “Menünüzde Vişneli Gazoz var mı?” (Eğer Vişneli Gazoz {menu_prompt_data}'da yoksa) -> `konusma_metni`: \"Hemen kontrol ediyorum... Maalesef menümüzde şu an için Vişneli Gazoz bulunmuyor. Size menümüzden başka bir soğuk içecek, örneğin ev yapımı limonatamızı veya taze sıkılmış meyve sularımızı önermemi ister misiniz?\"\n"
    "9. **Öneri İstekleri:** Eğer kullanıcı bir veya birkaç özellik belirterek (örneğin 'çilekli bir şeyler', 'soğuk bir içecek', 'hafif bir tatlı') VE SONUNDA 'ne önerirsin?', 'ne tavsiye edersin?', 'ne yesem/içsem?', 'ne alabilirim?' gibi bir soruyla veya ifadeyle öneri istiyorsa, **KESİNLİKLE doğrudan sipariş alma.** JSON `sepet` alanını boş liste `[]` olarak, `toplam_tutar`ı `0.0` olarak ayarla. Bunun yerine, STOKTAKİ ÜRÜNLER listesinden bu özelliklere uygun, GERÇEKTE VAR OLAN bir veya birkaç ürünü `konusma_metni` alanında metin olarak öner. Önerini sunduktan sonra müşterinin onayını veya seçimini bekle.\n"
    "10. **Genel Sorular ve Menü Listeleme:** Eğer kullanıcı genel bir soru soruyorsa (örn. “Menüde neler var?”, “Kahveleriniz nelerdir?”, “Bugün hava nasıl?”), siparişle ilgisi yoksa veya menüyü istiyorsa, JSON `sepet` alanını boş liste `[]` olarak, `toplam_tutar`ı `0.0` olarak ayarla ve sadece `konusma_metni` alanında sorusuna uygun şekilde (gerekirse menüyü kategorilere göre listeleyerek) bilgi ver.\n"
    "11. **Belirsiz Siparişler ve Onay Soruları:** Ürün, adet veya özelliklerden tam emin değilsen veya sipariş belirsizse, doğrudan sipariş almak yerine JSON `sepet` alanını boş liste `[]` olarak, `toplam_tutar`ı `0.0` olarak ayarla ve `konusma_metni` alanında kibar bir onay sorusu sor (örn. “Türk kahveniz sade mi olsun, yoksa başka bir özellik mi ekleyelim?”).\n"
    "12. **Sipariş Dışı Genel Sohbet ve Tavsiyeler:** Müşteri sipariş dışı bir talepte bulunursa (örn. “Hastayım, ne içmeliyim?”, “Sevgilimden ayrıldım.”), JSON `sepet` alanını boş liste `[]` olarak, `toplam_tutar`ı `0.0` olarak ayarla. Bağlama uygun, STOKTAKİ ÜRÜNLER listesinden bir öneriyi (Fıstık Kafe konseptine uygun olarak kahve, çay, bitki çayı, taze meyve suyu vb.) `konusma_metni` alanında sun. Hava durumu bilgisi verilirse bunu dikkate al.\n"
    "     - Örnek: “Hastayım” → `konusma_metni`: “Çok geçmiş olsun! Hızlı iyileşmenize yardımcı olması için menümüzdeki taze sıkılmış portakal suyunu veya bir bitki çayını (papatya, adaçayı gibi seçeneklerimiz var) denemenizi önerebilirim. Hangisini istersiniz?”\n"
    "     - Örnek: “Sevgilimden ayrıldım” (Hava sıcaksa) → `konusma_metni`: “Ooo, üzüldüm ama canınız sağ olsun! Belki şöyle bol köpüklü bir Türk kahvesi ya da serinletici bir naneli limonata keyfinizi biraz yerine getirir? Ne dersiniz?”\n\n"
    "## Sipariş Onayı ve JSON Üretimi:\n"
    "13. Sadece kullanıcı net bir şekilde bir ürünü ve adedini belirterek sipariş verirse VEYA daha önce sunduğun bir öneriyi açıkça kabul ederse (örn. ‘Evet, naneli limonata alayım.’), o zaman sipariş için aşağıdaki formatta JSON üret. Diğer tüm durumlarda (soru, belirsiz istek, öneri isteme, menüde olmayan ürün) `sepet` boş olmalı ve yanıt `konusma_metni` üzerinden verilmelidir.\n\n"
    "# JSON ÇIKTISI ve METİN YANITLARI (YENİ TALİMATLAR)\n" # YENİ TALİMAT BAŞLIĞI
    "1.  **Net Sipariş Durumu (Kural 13):** Eğer kullanıcı açıkça bir veya daha fazla menü ürününü adetleriyle birlikte sipariş ediyorsa veya daha önce sunduğun bir sipariş önerisini net olarak kabul ediyorsa, YALNIZCA ve YALNIZCA aşağıdaki JSON formatında yanıt ver. Bu JSON dışında BAŞKA HİÇBİR METİN EKLEME.\n"
    "    {{\n"
    "      \"sepet\": [ {{\n"
    "        \"urun\": \"MENÜDEKİ TAM ÜRÜN ADI\",\n"
    "        \"adet\": ADET_SAYISI (integer),\n"
    "        \"fiyat\": BIRIM_FIYAT (float),\n"
    "        \"kategori\": \"KATEGORI_ADI\",\n"
    "        \"musteri_notu\": \"EK ÖZELLİKLER (sade, şekerli, vb.) veya ''\"\n"
    "      }} ],\n"
    "      \"toplam_tutar\": TOPLAM_TUTAR (float),\n"
    "      \"musteri_notu\": \"SİPARİŞİN GENELİ İÇİN NOT (örn: hepsi paket olsun) veya ''\",\n"
    "      \"konusma_metni\": \"Siparişi onaylayan kısa ve nazik bir metin (müşterinin konuştuğu dilde).\"\n"
    "    }}\n"
    "2.  **Sipariş Dışı Durumlar (Kural 8, 9, 10, 11, 12 - Menüde olmayan ürün, öneri isteği, genel soru, belirsiz sipariş, genel sohbet, \"Merhaba\" gibi selamlaşmalar):** Bu durumlarda KESİNLİKLE JSON FORMATINDA BİR ÇIKTI ÜRETME. Bunun yerine, sadece müşteriye söyleyeceğin uygun diyalog metnini DÜZ METİN olarak yaz. Örneğin, \"Merhaba! Size nasıl yardımcı olabilirim?\" veya \"Maalesef o ürün menümüzde bulunmuyor.\" gibi.\n\n" # JSON ÇIKARMA TALİMATI GÜÇLENDİRİLDİ
    "# ÖRNEKLER\n\n"
    "## Örnek 1: Spesifik Özelliklerle Öneri İsteği (Menüdeki Gerçek Ürünlerle Öner) -> DÜZ METİN YANIT\n" # ÖRNEK GÜNCELLENDİ
    "Kullanıcı: \"Çilekli Soğuk birşeyler istiyorum ne önerirsin?\"\n"
    "Çıktı (DÜZ METİN):\n" # JSON DEĞİL
    "Elbette! Çilekli ve soğuk bir şeyler arıyorsunuz. Menümüzdeki çilekli soğuk içeceklerden size örneğin Çilekli Milkshake'i (eğer menümüzde varsa ve stoktaysa) önerebilirim. Ya da dilerseniz diğer soğuk içecek seçeneklerimize birlikte bakalım. Ne dersiniz?\n\n"
    "## Örnek 2: Öneriyi Kabul Etme ve Sipariş Oluşturma -> JSON YANIT\n"
    "Kullanıcı: (Önceki öneriye istinaden) \"Tamam, çilekli milkshake alayım bir tane.\"\n"
    "Çıktı (JSON):\n"
    "{{\n"
    '  "sepet": [\n'
    '    {{\n'
    '      "urun": "Çilekli Milkshake",\n'
    '      "adet": 1,\n'
    '      "fiyat": 25.0,\n'
    '      "kategori": "Soğuk İçecekler",\n'
    '      "musteri_notu": ""\n'
    '    }}\n'
    '  ],\n'
    '  "toplam_tutar": 25.0,\n'
    '  "musteri_notu": "",\n'
    '  "konusma_metni": "Harika seçim! Bir adet Çilekli Milkshake hemen hazırlanıyor. Başka bir arzunuz var mıydı?"\n'
    "}}\n\n"
    "## Örnek 3: Karma İstek (Menüde Olan İçecek ve Olmayan Yemek) -> JSON YANIT (Sadece geçerli ürünlerle)\n"
    "Kullanıcı: \"Rafık baa 2 Türk kahvesi, 1’i şekersiz olsun 1’i az şekerli, bir de yanına şırdan atsana bol acılı.\"\n"
    "Çıktı (JSON):\n"
    "{{\n"
    '  "sepet": [\n'
    '    {{\n'
    '      "urun": "Türk Kahvesi",\n'
    '      "adet": 1,\n'
    '      "fiyat": 15.0,\n'
    '      "kategori": "Sıcak İçecekler",\n'
    '      "musteri_notu": "şekersiz"\n'
    '    }},\n'
    '    {{\n'
    '      "urun": "Türk Kahvesi",\n'
    '      "adet": 1,\n'
    '      "fiyat": 15.0,\n'
    '      "kategori": "Sıcak İçecekler",\n'
    '      "musteri_notu": "az şekerli"\n'
    '    }}\n'
    '  ],\n'
    '  "toplam_tutar": 30.0,\n'
    '  "musteri_notu": "",\n'
    '  "konusma_metni": "Hemen geliyor şefim! Bir şekersiz, bir de az şekerli Türk kahveniz hazırlanıyor. Şırdan gibi yemek çeşitlerimiz Fıstık Kafe\'de maalesef bulunmuyor. Kahvelerinizin yanına belki taze pişmiş bir kurabiye veya günlük keklerimizden ikram edebilirim?"\n'
    "}}\n\n"
    "## Örnek 4: Menüde Olmayan YEMEK İsteği -> DÜZ METİN YANIT\n" # ÖRNEK GÜNCELLENDİ
    "Kullanıcı: \"Bana bir büyük boy Adana Dürüm yollar mısın?\"\n"
    "Çıktı (DÜZ METİN):\n" # JSON DEĞİL
    "Fıstık Kafe'de Adana dürüm gibi yemek servisimiz bulunmuyor, üzgünüz. Size bunun yerine özel harman bir filtre kahve veya serinletici bir ice latte hazırlamamı ister misiniz?\n\n"
    "## Örnek 5: Genel Menü Sorusu -> DÜZ METİN YANIT\n" # ÖRNEK GÜNCELLENDİ
    "Kullanıcı: \"Menüde neler var?\"\n"
    "Çıktı (DÜZ METİN):\n" # JSON DEĞİL
    "Tabii, hemen Fıstık Kafe menümüzü sizinle paylaşıyorum: [AI BURADA {menu_prompt_data}\'dan ALDIĞI BİLGİLERLE KATEGORİLERE GÖRE MENÜ ÖZETİ SUNAR, YEMEK OLMADIĞINI VURGULARAK İÇECEK VE ATIŞTIRMALIKLARI ÖNE ÇIKARIR] ... Özellikle denemek istediğiniz bir kahve, çay veya atıştırmalık var mı?\n\n"
    "## Örnek 6: Basit Selamlama -> DÜZ METİN YANIT\n" # YENİ ÖRNEK
    "Kullanıcı: \"Merhaba Neso nasılsın?\"\n"
    "Çıktı (DÜZ METİN):\n" # JSON DEĞİL
    "Merhaba! İyiyim, teşekkür ederim. Fıstık Kafe'de size yardımcı olmak için hazırım. Ne arzu edersiniz?\n\n"
    "Şimdi kullanıcının talebini bu kurallara ve örneklere göre işle ve uygun JSON veya DÜZ METİN çıktısını üret." # SON TALİMAT GÜNCELLENDİ
)

# @alru_cache(maxsize=1) # Bu fonksiyonların tanımları zaten aşağıda mevcut, tekrar etmeye gerek yok.
# async def get_menu_price_dict() -> Dict[str, float]: ...
# async def get_menu_stock_dict() -> Dict[str, int]: ...
# async def get_menu_for_prompt_cached() -> str: ... # Bu da aşağıda tanımlı

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
    except KeyError as ke:
        logger.error(f"❌ Sistem mesajı güncellenirken KeyError oluştu: {ke}. Şablonda eksik/yanlış anahtar olabilir.", exc_info=True)
        try:
            current_system_content_fallback = SISTEM_MESAJI_ICERIK_TEMPLATE.replace("{menu_prompt_data}", "Menü bilgisi yüklenirken hata oluştu (fallback).")
            SYSTEM_PROMPT = {"role": "system", "content": current_system_content_fallback}
            logger.warning(f"Fallback sistem mesajı (KeyError sonrası) kullanılıyor.")
        except Exception as fallback_e:
            logger.error(f"❌ Fallback sistem mesajı oluşturulurken de hata oluştu: {fallback_e}", exc_info=True)
            SYSTEM_PROMPT = {"role": "system", "content": "Ben Neso, Fıstık Kafe sipariş asistanıyım. Size nasıl yardımcı olabilirim? (Sistem mesajı yüklenemedi.)"}
    except Exception as e:
        logger.error(f"❌ Sistem mesajı güncellenirken BEKLENMEDİK BİR HATA oluştu: {e}", exc_info=True)
        if SYSTEM_PROMPT is None:
            try:
                current_system_content_fallback = SISTEM_MESAJI_ICERIK_TEMPLATE.replace("{menu_prompt_data}", "Menü bilgisi yüklenirken genel hata oluştu (fallback).")
                SYSTEM_PROMPT = {"role": "system", "content": current_system_content_fallback}
                logger.warning(f"Fallback sistem mesajı (BEKLENMEDİK HATA sonrası) kullanılıyor.")
            except Exception as fallback_e:
                 logger.error(f"❌ Fallback sistem mesajı oluşturulurken de (genel hata sonrası) hata oluştu: {fallback_e}", exc_info=True)
                 SYSTEM_PROMPT = {"role": "system", "content": "Ben Neso, Fıstık Kafe sipariş asistanıyım. Size nasıl yardımcı olabilirim? (Sistem mesajı yüklenemedi.)"}

@app.get("/admin/clear-menu-caches", tags=["Admin İşlemleri"])
async def clear_all_caches_endpoint(current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))):
    logger.info(f"Admin '{current_user.kullanici_adi}' tarafından manuel cache temizleme isteği alındı.")
    await update_system_prompt()
    return {"message": "Menü, fiyat ve stok cache'leri başarıyla temizlendi. Sistem promptu güncellendi."}

@app.get("/menu", tags=["Menü"])
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
        async with menu_db.transaction():
            await menu_db.execute("INSERT INTO kategoriler (isim) VALUES (:isim) ON CONFLICT (isim) DO NOTHING", {"isim": item_data.kategori})
            category_id_row = await menu_db.fetch_one("SELECT id FROM kategoriler WHERE LOWER(isim) = LOWER(:isim)", {"isim": item_data.kategori})
            if not category_id_row: raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Kategori oluşturulurken veya bulunurken bir sorun oluştu.")
            category_id = category_id_row['id']
            try:
                item_id = await menu_db.fetch_val("""
                    INSERT INTO menu (ad, fiyat, kategori_id, stok_durumu) VALUES (:ad, :fiyat, :kategori_id, 1) RETURNING id
                """, {"ad": item_data.ad, "fiyat": item_data.fiyat, "kategori_id": category_id})
            except Exception as e_db:
                 if "duplicate key value violates unique constraint" in str(e_db).lower() or "UNIQUE constraint failed" in str(e_db).lower():
                    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"'{item_data.ad}' adlı ürün bu kategoride zaten mevcut.")
                 logger.error(f"DB Hatası /menu/ekle: {e_db}", exc_info=True)
                 raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Veritabanı hatası: {str(e_db)}")
        await update_system_prompt()
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
        async with menu_db.transaction():
            item_to_delete = await menu_db.fetch_one("SELECT id FROM menu WHERE LOWER(ad) = LOWER(:ad)", {"ad": urun_adi})
            if not item_to_delete: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{urun_adi}' adlı ürün menüde bulunamadı.")
            await menu_db.execute("DELETE FROM menu WHERE id = :id", {"id": item_to_delete['id']})
        await update_system_prompt()
        return {"mesaj": f"'{urun_adi}' ürünü menüden başarıyla silindi."}
    except HTTPException as http_exc: raise http_exc
    except Exception as e:
        logger.error(f"❌ Menüden ürün silinirken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Menüden ürün silinirken bir sunucu hatası oluştu.")

@app.post("/yanitla", tags=["Yapay Zeka"])
async def handle_message_endpoint(request: Request, data: dict = Body(...)):
    user_message = data.get("text", "").strip()
    table_id = data.get("masa", "bilinmiyor")
    session_id = request.session.get("session_id")
    if not session_id:
        session_id = secrets.token_hex(16)
        request.session["session_id"] = session_id
        request.session["chat_history"] = []
    chat_history = request.session.get("chat_history", [])
    logger.info(f"💬 AI Yanıt isteği: Masa '{table_id}', Session ID: '{session_id}', Kullanıcı Mesajı: '{user_message}'")
    if not user_message: raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Mesaj boş olamaz.")
    if SYSTEM_PROMPT is None:
        await update_system_prompt()
        if SYSTEM_PROMPT is None:
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="AI asistanı şu anda hazır değil (sistem mesajı eksik).")
    try:
        messages_for_openai = [SYSTEM_PROMPT] + chat_history + [{"role": "user", "content": user_message}]
        response = openai_client.chat.completions.create( model=settings.OPENAI_MODEL, messages=messages_for_openai, temperature=0.3, max_tokens=450)
        ai_reply_content = response.choices[0].message.content
        ai_reply = ai_reply_content.strip() if ai_reply_content else "Üzgünüm, şu anda bir yanıt üretemiyorum."
        
        # AI'nın düz metin mi yoksa JSON mu döndürdüğünü kontrol etmeye çalışalım (basit bir kontrol)
        is_json_response = ai_reply.startswith("{") and ai_reply.endswith("}")
        if is_json_response:
            try:
                # JSON'ı parse etmeye çalışarak gerçekten geçerli olup olmadığını teyit et
                json.loads(ai_reply) 
                logger.info(f"AI JSON formatında yanıt verdi: {ai_reply[:200]}...")
            except json.JSONDecodeError:
                is_json_response = False # Geçersiz JSON ise düz metin kabul et
                logger.warning(f"AI JSON gibi görünen ama geçersiz bir yanıt verdi, düz metin olarak işlenecek: {ai_reply[:200]}...")
        else:
             logger.info(f"AI düz metin formatında yanıt verdi: {ai_reply[:200]}...")

        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": ai_reply}) # AI'nın ham yanıtını sakla
        request.session["chat_history"] = chat_history[-10:]
        
        # Yanıtı frontend'e gönderirken, AI'nın talimata uyup uymadığını kontrol etmiyoruz,
        # sadece AI'nın ürettiği ham yanıtı yolluyoruz. Frontend bunu işleyecek.
        return {"reply": ai_reply, "sessionId": session_id}

    except OpenAIError as e:
        logger.error(f"❌ OpenAI API hatası: {type(e).__name__} - {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"AI servisinden yanıt alınırken bir sorun oluştu: {type(e).__name__}")
    except Exception as e:
        logger.error(f"❌ /yanitla endpoint genel hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Mesajınız işlenirken beklenmedik bir sunucu hatası oluştu.")

SUPPORTED_LANGUAGES = {"tr-TR", "en-US", "en-GB", "fr-FR", "de-DE"}
@app.post("/sesli-yanit", tags=["Yapay Zeka"])
async def generate_speech_endpoint(data: SesliYanitData):
    if not tts_client: raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Sesli yanıt servisi şu anda kullanılamıyor.")
    if data.language not in SUPPORTED_LANGUAGES: raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Desteklenmeyen dil: {data.language}.")
    cleaned_text = temizle_emoji(data.text)
    
    # Eğer gelen metin JSON formatında ise, sadece "konusma_metni" alanını seslendir.
    # Bu, AI'nın yanlışlıkla JSON döndürdüğü ama seslendirilmesi gereken bir konuşma metni olduğu durumlar için.
    try:
        if cleaned_text.strip().startswith("{") and cleaned_text.strip().endswith("}"):
            parsed_json = json.loads(cleaned_text)
            if "konusma_metni" in parsed_json and isinstance(parsed_json["konusma_metni"], str):
                cleaned_text = parsed_json["konusma_metni"]
                logger.info(f"Sesli yanıt için JSON'dan 'konusma_metni' çıkarıldı: {cleaned_text[:100]}...")
            else: # Geçerli bir konuşma metni yoksa veya JSON değilse, olduğu gibi kullan.
                logger.warning("Sesli yanıt için gelen JSON'da 'konusma_metni' bulunamadı veya string değil, ham metin kullanılacak.")
    except json.JSONDecodeError:
        pass # JSON değilse, orijinal cleaned_text kullanılır.

    if not cleaned_text.strip(): raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Sese dönüştürülecek geçerli bir metin bulunamadı.")
    
    try:
        synthesis_input = texttospeech.SynthesisInput(text=cleaned_text)
        voice_name = "tr-TR-Chirp3-HD-Laomedeia" if data.language == "tr-TR" else None
        voice_params = texttospeech.VoiceSelectionParams(language_code=data.language, name=voice_name, ssml_gender=texttospeech.SsmlVoiceGender.FEMALE if data.language == "tr-TR" and voice_name else texttospeech.SsmlVoiceGender.NEUTRAL)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=1.0)
        response_tts = tts_client.synthesize_speech(input=synthesis_input, voice=voice_params, audio_config=audio_config)
        return Response(content=response_tts.audio_content, media_type="audio/mpeg")
    except google_exceptions.GoogleAPIError as e_google:
        detail_msg = f"Google TTS servisinden ses üretilirken bir hata oluştu: {getattr(e_google, 'message', str(e_google))}"
        status_code_tts = status.HTTP_503_SERVICE_UNAVAILABLE
        if "API key not valid" in str(e_google) or "permission" in str(e_google).lower() or "RESOURCE_EXHAUSTED" in str(e_google):
            detail_msg = "Google TTS servisi için kimlik/kota sorunu veya kaynak yetersiz."
        elif "Requested voice not found" in str(e_google) or "Invalid DefaultVoice" in str(e_google):
            detail_msg = f"İstenen ses modeli ({voice_name}) bulunamadı veya geçersiz."; status_code_tts = status.HTTP_400_BAD_REQUEST
        logger.error(f"❌ Google TTS API hatası: {e_google}", exc_info=True)
        raise HTTPException(status_code=status_code_tts, detail=detail_msg)
    except Exception as e:
        logger.error(f"❌ Sesli yanıt endpoint'inde beklenmedik hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Sesli yanıt oluşturulurken beklenmedik bir sunucu hatası oluştu.")

# ... (Kasa, Kullanıcı Yönetimi ve diğer endpointleriniz aynı kalacak) ...
# Kasa İşlemleri (önceki gibi)
@app.post("/kasa/siparis/{siparis_id}/odendi", tags=["Kasa İşlemleri"])
async def mark_order_as_paid_endpoint(
    siparis_id: int = Path(..., description="Ödendi olarak işaretlenecek siparişin ID'si"),
    odeme_bilgisi: KasaOdemeData = Body(...),
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN, KullaniciRol.KASIYER]))
):
    logger.info(f"💰 Kasa: Sipariş {siparis_id} ödendi olarak işaretleniyor (Kullanıcı: {current_user.kullanici_adi}). Ödeme: {odeme_bilgisi.odeme_yontemi}")
    try:
        async with db.transaction():
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
                {"yeni_durum": Durum.ODENDI.value, "odeme_yontemi": odeme_bilgisi.odeme_yontemi, "id": siparis_id}
            )
        if not updated_order_raw:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sipariş güncellenemedi.")
        updated_order = dict(updated_order_raw)
        updated_order["sepet"] = json.loads(updated_order.get("sepet", "[]"))
        if isinstance(updated_order.get('zaman'), datetime):
            updated_order['zaman'] = updated_order['zaman'].isoformat()
        notif_data = {**updated_order, "zaman": datetime.now(TR_TZ).isoformat()}
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
        base_query = "SELECT id, masa, istek, sepet, zaman, durum, odeme_yontemi FROM siparisler WHERE "
        values = {}
        valid_statuses = [Durum.HAZIR.value, Durum.BEKLIYOR.value, Durum.HAZIRLANIYOR.value]
        if durum:
            if durum not in valid_statuses:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Geçersiz durum. Kullanılabilecekler: {', '.join(valid_statuses)}")
            query_str = base_query + "durum = :durum ORDER BY zaman ASC"
            values["durum"] = durum
        else:
            # === DÜZELTİLMİŞ SORGU ===
            query_str = base_query + "durum = ANY(:statuses_list) ORDER BY zaman ASC"
            values["statuses_list"] = valid_statuses # tuple() yapmaya gerek yok
            # ==========================

        orders_raw = await db.fetch_all(query=query_str, values=values)
        orders_data = []
        for row in orders_raw:
            order_dict = dict(row)
            order_dict["sepet"] = json.loads(order_dict.get('sepet','[]'))
            if isinstance(order_dict.get('zaman'), datetime):
                 order_dict['zaman'] = order_dict['zaman'].isoformat()
            orders_data.append(order_dict)
        return {"orders": orders_data}
    except Exception as e: # exc_info=True önemli
        logger.error(f"❌ Kasa: Ödeme bekleyen siparişler alınırken hata: {e}", exc_info=True)
        if isinstance(e, google_exceptions.PostgresSyntaxError): # asyncpg.exceptions.PostgresSyntaxError olacak
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Veritabanı sorgu hatası: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Siparişler alınırken bir hata oluştu.")

@app.get("/kasa/masa/{masa_id}/hesap", tags=["Kasa İşlemleri"])
async def get_table_bill_endpoint(
    masa_id: str = Path(...),
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN, KullaniciRol.KASIYER]))
):
    logger.info(f"💰 Kasa: Masa {masa_id} için hesap isteniyor (Kullanıcı: {current_user.kullanici_adi}).")
    try:
        active_payable_statuses = [Durum.BEKLIYOR.value, Durum.HAZIRLANIYOR.value, Durum.HAZIR.value]
        query_str = "SELECT id, masa, istek, sepet, zaman, durum, yanit, odeme_yontemi FROM siparisler WHERE masa = :masa_id AND durum IN :statuses ORDER BY zaman ASC"
        values = {"masa_id": masa_id, "statuses": tuple(active_payable_statuses)}
        
        orders_raw = await db.fetch_all(query=query_str, values=values)
        orders_data = []
        toplam_tutar = 0.0
        for row in orders_raw:
            order_dict = dict(row)
            sepet_items = json.loads(order_dict.get('sepet', '[]'))
            order_dict['sepet'] = sepet_items
            if isinstance(order_dict.get('zaman'), datetime):
                 order_dict['zaman'] = order_dict['zaman'].isoformat()
            for item in sepet_items:
                if isinstance(item,dict) and isinstance(item.get('adet',0),(int,float)) and isinstance(item.get('fiyat',0.0),(int,float)):
                    toplam_tutar += item['adet'] * item['fiyat']
            orders_data.append(order_dict)
        return {"masa_id": masa_id, "siparisler": orders_data, "toplam_tutar": round(toplam_tutar, 2)}
    except Exception as e:
        logger.error(f"❌ Kasa: Masa {masa_id} hesabı alınırken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Masa hesabı alınırken bir hata oluştu.")

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
    query = """
        INSERT INTO kullanicilar (kullanici_adi, sifre_hash, rol, aktif_mi)
        VALUES (:kullanici_adi, :sifre_hash, :rol, :aktif_mi)
        RETURNING id, kullanici_adi, rol, aktif_mi
    """
    values = {"kullanici_adi": user_data.kullanici_adi, "sifre_hash": hashed_password, "rol": user_data.rol.value, "aktif_mi": user_data.aktif_mi}
    try:
        created_user_row = await db.fetch_one(query, values)
        if not created_user_row:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Kullanıcı oluşturulurken bir sorun oluştu (DB).")
        logger.info(f"Kullanıcı '{created_user_row['kullanici_adi']}' başarıyla oluşturuldu (ID: {created_user_row['id']}).")
        return Kullanici(**created_user_row)
    except Exception as e:
        if "duplicate key value violates unique constraint" in str(e).lower() or "UNIQUE constraint failed" in str(e).lower():
            logger.warning(f"Yeni kullanıcı oluşturma hatası (DB): '{user_data.kullanici_adi}' zaten mevcut.")
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Bu kullanıcı adı veritabanında zaten mevcut.")
        logger.error(f"Yeni kullanıcı ({user_data.kullanici_adi}) DB'ye eklenirken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Kullanıcı veritabanına eklenirken hata: {str(e)}")

@app.get("/admin/kullanicilar", response_model=List[Kullanici], tags=["Kullanıcı Yönetimi"])
async def list_all_users(current_admin: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))):
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
    target_user_row = await db.fetch_one("SELECT id, kullanici_adi, rol, aktif_mi FROM kullanicilar WHERE id = :user_id", {"user_id": user_id})
    if not target_user_row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Güncellenecek kullanıcı bulunamadı.")
    target_user = dict(target_user_row)
    update_fields = {}
    if user_update_data.kullanici_adi is not None and user_update_data.kullanici_adi != target_user["kullanici_adi"]:
        existing_user_with_new_name = await db.fetch_one("SELECT id FROM kullanicilar WHERE kullanici_adi = :k_adi AND id != :u_id", {"k_adi": user_update_data.kullanici_adi, "u_id": user_id})
        if existing_user_with_new_name:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Bu kullanıcı adı zaten başka bir kullanıcı tarafından kullanılıyor.")
        update_fields["kullanici_adi"] = user_update_data.kullanici_adi
    if user_update_data.rol is not None: update_fields["rol"] = user_update_data.rol.value
    if user_update_data.aktif_mi is not None: update_fields["aktif_mi"] = user_update_data.aktif_mi
    if user_update_data.sifre is not None: update_fields["sifre_hash"] = get_password_hash(user_update_data.sifre)
    if not update_fields:
        logger.info(f"Kullanıcı ID {user_id} için güncellenecek bir alan belirtilmedi.")
        return Kullanici(**target_user)
    set_clause_parts = [f"{key} = :{key}" for key in update_fields.keys()]
    set_clause = ", ".join(set_clause_parts)
    query = f"UPDATE kullanicilar SET {set_clause} WHERE id = :user_id_param RETURNING id, kullanici_adi, rol, aktif_mi"
    values = {**update_fields, "user_id_param": user_id}
    try:
        updated_user_row = await db.fetch_one(query, values)
        if not updated_user_row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Kullanıcı güncellenirken bulunamadı.")
        logger.info(f"Kullanıcı ID {user_id} başarıyla güncellendi. Yeni değerler: {dict(updated_user_row)}")
        return Kullanici(**updated_user_row)
    except Exception as e:
        if "duplicate key value violates unique constraint" in str(e).lower() or "UNIQUE constraint failed" in str(e).lower():
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
    host_ip = os.getenv("HOST", "127.0.0.1")
    port_num = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 FastAPI uygulaması {host_ip}:{port_num} adresinde başlatılıyor (yerel geliştirme modu)...")
    uvicorn.run("main:app", host=host_ip, port=port_num, reload=True, log_config=LOGGING_CONFIG)