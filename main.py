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
import sqlite3
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

# Loglama Yapılandırması (Değişiklik yok)
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

# --- YENİ: Kullanıcı Rolleri ---
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
    DB_DATA_DIR: str = "."
    OPENAI_MODEL: str = "gpt-3.5-turbo"

    # YENİ: JWT Ayarları
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440 # 1 gün

    # YENİ: Varsayılan Admin Kullanıcısı (sadece ilk çalıştırmada DB'ye eklemek için)
    DEFAULT_ADMIN_USERNAME: str = "admin"
    DEFAULT_ADMIN_PASSWORD: str = "ChangeThisDefaultPassword123!"


    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

try:
    settings = Settings()
except ValueError as e:
    logger.critical(f"❌ Ortam değişkenleri eksik: {e}")
    raise SystemExit(f"Ortam değişkenleri eksik: {e}")

# --- YENİ: Şifreleme ve OAuth2 ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token") # "/token" endpoint'i login için kullanılacak

# Yardımcı Fonksiyonlar (Değişiklik yok)
def temizle_emoji(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    try:
        emoji_pattern = regex.compile(r"[\p{Emoji_Presentation}\p{Extended_Pictographic}]+", regex.UNICODE)
        return emoji_pattern.sub('', text)
    except Exception as e:
        logger.error(f"Emoji temizleme hatası: {e}")
        return text

# API İstemcileri Başlatma (Değişiklik yok)
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
    version="1.3.0", # Rol tabanlı erişim ile versiyon güncellendi
    description="Fıstık Kafe için sipariş backend servisi."
)
# security = HTTPBasic() # Eski HTTPBasic kaldırıldı

# Middleware Ayarları (Değişiklik yok)
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
    secret_key=settings.SECRET_KEY, # Session için de aynı SECRET_KEY kullanılabilir
    session_cookie="neso_session"
)
logger.info(f"Session Middleware etkinleştirildi.")


# Veritabanı Bağlantı Havuzu (Değişiklik yok)
DB_NAME = "neso.db"
MENU_DB_NAME = "neso_menu.db"
DB_PATH = os.path.join(settings.DB_DATA_DIR, DB_NAME)
MENU_DB_PATH = os.path.join(settings.DB_DATA_DIR, MENU_DB_NAME)
os.makedirs(settings.DB_DATA_DIR, exist_ok=True)

db = Database(f"sqlite:///{DB_PATH}")
menu_db = Database(f"sqlite:///{MENU_DB_PATH}")

# Türkiye Saat Dilimi (UTC+3)
TR_TZ = dt_timezone(timedelta(hours=3))

# --- YENİ: Pydantic Kullanıcı Modelleri ---
class KullaniciBase(BaseModel):
    kullanici_adi: str
    rol: KullaniciRol
    aktif_mi: Optional[bool] = True

class KullaniciCreate(KullaniciBase): # Admin tarafından kullanıcı oluşturma
    sifre: str

class KullaniciUpdate(BaseModel): # Admin tarafından kullanıcı güncelleme
    kullanici_adi: Optional[str] = None
    rol: Optional[KullaniciRol] = None
    aktif_mi: Optional[bool] = None
    sifre: Optional[str] = None # Şifre de güncellenebilir

class KullaniciInDBBase(KullaniciBase):
    id: int
    class Config:
        from_attributes = True

class Kullanici(KullaniciInDBBase): # API yanıtlarında kullanılacak model (şifresiz)
    pass

class KullaniciInDB(KullaniciInDBBase): # Veritabanında saklanacak model (şifre hash'li)
    sifre_hash: str

# --- YENİ: Token Modelleri ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    kullanici_adi: Union[str, None] = None
    # scopes: List[str] = [] # Rolleri scope olarak da kullanabilirdik, şimdilik sadece kullanıcı adı yeterli

# --- YENİ: Şifreleme ve Kimlik Doğrulama Yardımcı Fonksiyonları ---
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
        username: str = payload.get("sub")
        if username is None:
            logger.warning("Token'da kullanıcı adı (sub) bulunamadı.")
            raise credentials_exception
        # token_scopes = payload.get("scopes", []) # Eğer scope kullanılsaydı
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

# --- YENİ: Rol Bazlı Yetkilendirme Dependency ---
def role_checker(required_roles: List[KullaniciRol]):
    async def checker(current_user: Kullanici = Depends(get_current_active_user)) -> Kullanici:
        if current_user.rol not in required_roles:
            logger.warning(
                f"Yetkisiz erişim denemesi: Kullanıcı '{current_user.kullanici_adi}' (Rol: {current_user.rol}), "
                f"Gereken Roller: {required_roles}"
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
    await db.connect()
    await menu_db.connect()
    logger.info("✅ Veritabanı bağlantıları kuruldu.")
    await init_databases() # init_databases içinde init_db ve init_menu_db çağrılacak
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

# WebSocket Yönetimi (Değişiklik yok)
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
                elif message.get("type") == "status_update" and endpoint_name == "Admin": # Admin WS için örnek özel mesaj işleme
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
async def websocket_admin_endpoint(websocket: WebSocket): # TODO: Buraya da token/session ile yetkilendirme eklenebilir
    await websocket_lifecycle(websocket, aktif_admin_websocketleri, "Admin")

@app.websocket("/ws/mutfak")
async def websocket_mutfak_endpoint(websocket: WebSocket): # TODO: Buraya da token/session ile yetkilendirme eklenebilir
    await websocket_lifecycle(websocket, aktif_mutfak_websocketleri, "Mutfak/Masa")

@app.websocket("/ws/kasa")
async def websocket_kasa_endpoint(websocket: WebSocket): # TODO: Buraya da token/session ile yetkilendirme eklenebilir
    await websocket_lifecycle(websocket, aktif_kasa_websocketleri, "Kasa")

# Veritabanı İşlemleri (update_table_status - Değişiklik yok)
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
        """, {"masa_id": masa_id, "son_erisim": now.strftime("%Y-%m-%d %H:%M:%S"), "islem": islem})

        await broadcast_message(aktif_admin_websocketleri, {
            "type": "masa_durum",
            "data": {"masaId": masa_id, "sonErisim": now.isoformat(), "aktif": True, "sonIslem": islem}
        }, "Admin")
    except Exception as e:
        logger.error(f"❌ Masa durumu ({masa_id}) güncelleme hatası: {e}")

# Middleware (track_active_users - Değişiklik yok)
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

# --- YENİ: Login Endpoint ---
@app.post("/token", response_model=Token, tags=["Kimlik Doğrulama"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    logger.info(f"Giriş denemesi: Kullanıcı adı '{form_data.username}'")
    user_in_db = await get_user_from_db(username=form_data.username)
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
        data={"sub": user_in_db.kullanici_adi}, # Token'a sadece kullanıcı adı (subject) ekleniyor
        expires_delta=access_token_expires
    )
    logger.info(f"Kullanıcı '{user_in_db.kullanici_adi}' (Rol: {user_in_db.rol}) başarıyla giriş yaptı. Token oluşturuldu.")
    return {"access_token": access_token, "token_type": "bearer"}


# Pydantic Modelleri (Sipariş, Menü vb. - Değişiklik yok, Kullanici modelleri yukarı eklendi)
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

class SiparisGuncelleData(BaseModel): # Admin, Mutfak, Barista kullanabilir
    masa: str = Field(..., min_length=1, description="Durumu güncellenecek siparişin masa numarası.")
    durum: Durum = Field(..., description="Siparişin yeni durumu.")
    id: Optional[int] = Field(None, description="Durumu güncellenecek siparişin ID'si (belirli bir sipariş için).")

class AktifMasaOzet(BaseModel):
    masa_id: str
    odenmemis_tutar: float
    aktif_siparis_sayisi: int
    siparis_detaylari: Optional[List[Dict]] = None

class KasaOdemeData(BaseModel): # Kasiyer kullanır
    odeme_yontemi: Optional[str] = Field(None, description="Ödeme yöntemi (örn: nakit, kart)")

class MenuEkleData(BaseModel): # Admin kullanır
    ad: str = Field(..., min_length=1, description="Menüye eklenecek ürünün adı.")
    fiyat: float = Field(..., gt=0, description="Ürünün fiyatı.")
    kategori: str = Field(..., min_length=1, description="Ürünün kategorisi.")

class AdminCredentialsUpdate(BaseModel): # Artık kullanılmayacak, şifre değişikliği DB üzerinden olmalı
    yeniKullaniciAdi: str = Field(..., min_length=1)
    yeniSifre: str = Field(..., min_length=8)

class SesliYanitData(BaseModel):
    text: str = Field(..., min_length=1, description="Sese dönüştürülecek metin.")
    language: str = Field(default="tr-TR", pattern=r"^[a-z]{2}-[A-Z]{2}$", description="Metnin dili (örn: tr-TR, en-US).")


# --- YENİ: Korunan Endpoint Örnekleri (Rol Tabanlı) ---

@app.get("/users/me", response_model=Kullanici, tags=["Kullanıcılar"])
async def read_users_me(current_user: Kullanici = Depends(get_current_active_user)):
    """ Mevcut giriş yapmış kullanıcının bilgilerini döndürür. """
    logger.info(f"Kullanıcı '{current_user.kullanici_adi}' kendi bilgilerini istedi.")
    return current_user

# Admin Doğrulama yerine JWT ve Rol Kullanımı
# def check_admin(credentials: HTTPBasicCredentials = Depends(security)): ... (Bu fonksiyon artık kullanılmıyor)

@app.get("/aktif-masalar/ws-count", tags=["Admin"]) # Eski /aktif-masalar endpoint'i, sadece WS sayısını verir
async def get_active_tables_ws_count_endpoint(
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    """ Sadece Admin tarafından erişilebilir. Aktif Mutfak WebSocket bağlantı sayısını döndürür. """
    logger.info(f"Admin '{current_user.kullanici_adi}' aktif WS masa sayısını istedi.")
    try:
        return {"aktif_mutfak_ws_sayisi": len(aktif_mutfak_websocketleri)}
    except Exception as e:
        logger.error(f"❌ Aktif masalar WS bağlantı sayısı alınamadı: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="WS bağlantı sayısı alınamadı."
        )

# Sipariş Yönetimi
@app.patch("/siparis/{id}", tags=["Siparişler"]) # response_model eklenebilir
async def patch_order_endpoint(
    id: int = Path(..., description="Güncellenecek siparişin ID'si"),
    data: SiparisGuncelleData = Body(...),
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN, KullaniciRol.MUTFAK_PERSONELI, KullaniciRol.BARISTA])) # Admin, Mutfak, Barista
):
    logger.info(f"🔧 PATCH /siparis/{id} ile durum güncelleme isteği (Kullanıcı: {current_user.kullanici_adi}, Rol: {current_user.rol}): {data.durum}")
    try:
        async with db.transaction():
            updated = await db.fetch_one(
                """
                UPDATE siparisler
                SET durum = :durum
                WHERE id = :id
                RETURNING id, masa, durum, sepet, istek, zaman
                """,
                {"durum": data.durum.value, "id": id}
            )
        if not updated:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sipariş bulunamadı.")

        order = dict(updated)
        try:
            order["sepet"] = json.loads(order.get("sepet", "[]"))
        except json.JSONDecodeError:
            order["sepet"] = []
            logger.warning(f"Sipariş {id} sepet JSON parse hatası (patch_order_endpoint).")

        notif_data = {
            "id": order["id"],
            "masa": order["masa"],
            "durum": order["durum"],
            "sepet": order["sepet"],
            "istek": order["istek"],
            "zaman": datetime.now(TR_TZ).isoformat() # Güncelleme zamanı
        }
        notification = {"type": "durum", "data": notif_data}
        await broadcast_message(aktif_mutfak_websocketleri, notification, "Mutfak/Masa")
        await broadcast_message(aktif_admin_websocketleri, notification, "Admin")
        await broadcast_message(aktif_kasa_websocketleri, notification, "Kasa")
        await update_table_status(order["masa"], f"Sipariş {id} durumu güncellendi -> {order['durum']} (by {current_user.kullanici_adi})")
        return {"message": f"Sipariş {id} güncellendi.", "data": order}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ PATCH /siparis/{id} hatası: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Sipariş durumu güncellenirken hata oluştu.")

@app.delete("/siparis/{id}", tags=["Siparişler"]) # Sadece Admin
async def delete_order_by_admin_endpoint(
    id: int = Path(..., description="İptal edilecek siparişin ID'si"),
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"🗑️ ADMIN DELETE /siparis/{id} ile iptal isteği (Kullanıcı: {current_user.kullanici_adi})")
    row = await db.fetch_one("SELECT zaman, masa FROM siparisler WHERE id = :id", {"id": id})
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sipariş bulunamadı.")

    olusturma_zamani_str = row["zaman"]
    try:
        olusturma_naive = datetime.strptime(olusturma_zamani_str, "%Y-%m-%d %H:%M:%S")
        olusturma_tr = olusturma_naive.replace(tzinfo=TR_TZ)
    except ValueError:
        logger.error(f"Sipariş {id} için geçersiz zaman formatı: {olusturma_zamani_str}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Sipariş zamanı okunamadı.")

    # Admin için iptal süresi kontrolü (isteğe bağlı, şimdilik kaldırıldı, müşteri iptalinde var)
    # if datetime.now(TR_TZ) - olusturma_tr > timedelta(minutes=2):
    #     logger.warning(f"Sipariş {id} (Masa: {row['masa']}) admin tarafından iptal edilmek istendi ancak 2 dakikalık süre aşıldı.")
    #     raise HTTPException(
    #         status_code=status.HTTP_400_BAD_REQUEST,
    #         detail="Bu sipariş 2 dakikayı geçtiği için artık iptal edilemez (Admin)."
    #     )

    try:
        async with db.transaction():
            await db.execute("UPDATE siparisler SET durum = 'iptal' WHERE id = :id", {"id": id})

        notif_data = {
            "id": id,
            "masa": row["masa"],
            "durum": "iptal",
            "zaman": datetime.now(TR_TZ).isoformat()
        }
        notification = {"type": "durum", "data": notif_data}
        await broadcast_message(aktif_mutfak_websocketleri, notification, "Mutfak/Masa")
        await broadcast_message(aktif_admin_websocketleri, notification, "Admin")
        await broadcast_message(aktif_kasa_websocketleri, notification, "Kasa")

        await update_table_status(row["masa"], f"Sipariş {id} admin ({current_user.kullanici_adi}) tarafından iptal edildi")
        logger.info(f"Sipariş {id} (Masa: {row['masa']}) admin ({current_user.kullanici_adi}) tarafından başarıyla iptal edildi.")
        return {"message": f"Sipariş {id} admin tarafından başarıyla iptal edildi."}
    except Exception as e:
        logger.error(f"❌ ADMIN DELETE /siparis/{id} hatası: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Sipariş admin tarafından iptal edilirken hata oluştu.")


@app.post("/musteri/siparis/{siparis_id}/iptal", status_code=status.HTTP_200_OK, tags=["Müşteri İşlemleri"])
async def cancel_order_by_customer_endpoint(
    siparis_id: int = Path(..., description="İptal edilecek siparişin ID'si"),
    masa_no: str = Query(..., description="Siparişin verildiği masa numarası/adı")
):
    logger.info(f"🗑️ Müşteri sipariş iptal isteği: Sipariş ID {siparis_id}, Masa No {masa_no}")
    # Bu endpoint için JWT koruması yok, çünkü müşteri yapıyor.
    # Gerekirse masa bazlı bir session veya basit bir token mekanizması eklenebilir.
    # Şimdilik olduğu gibi bırakıyoruz.

    order_details = await db.fetch_one(
        "SELECT id, zaman, masa, durum FROM siparisler WHERE id = :siparis_id AND masa = :masa_no",
        {"siparis_id": siparis_id, "masa_no": masa_no}
    )

    if not order_details:
        logger.warning(f"İptal edilecek sipariş (ID: {siparis_id}) bulunamadı veya Masa No ({masa_no}) ile eşleşmedi.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="İptal edilecek sipariş bulunamadı veya bu masaya ait değil."
        )

    if order_details["durum"] == "iptal":
        logger.info(f"Sipariş {siparis_id} (Masa: {masa_no}) zaten iptal edilmiş.")
        return {"message": "Bu sipariş zaten iptal edilmiş."}

    if order_details["durum"] != "bekliyor" and order_details["durum"] != "hazirlaniyor":
        logger.warning(f"Sipariş {siparis_id} (Masa: {masa_no}) durumu ({order_details['durum']}) iptal için uygun değil.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Siparişinizin durumu ({order_details['durum']}) iptal işlemi için uygun değil."
        )

    olusturma_zamani_str = order_details["zaman"]
    try:
        olusturma_naive = datetime.strptime(olusturma_zamani_str, "%Y-%m-%d %H:%M:%S")
        olusturma_tr = olusturma_naive.replace(tzinfo=TR_TZ)
    except ValueError:
        logger.error(f"Sipariş {siparis_id} için geçersiz zaman formatı: {olusturma_zamani_str}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Sipariş zamanı okunamadı."
        )

    if datetime.now(TR_TZ) - olusturma_tr > timedelta(minutes=2):
        logger.warning(f"Sipariş {siparis_id} (Masa: {masa_no}) 2 dakikalık müşteri iptal süresini aştı.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bu sipariş 2 dakikayı geçtiği için artık iptal edilemez."
        )

    try:
        async with db.transaction():
            await db.execute("UPDATE siparisler SET durum = 'iptal' WHERE id = :id", {"id": siparis_id})

        notif_data = {
            "id": siparis_id,
            "masa": masa_no,
            "durum": "iptal",
            "zaman": datetime.now(TR_TZ).isoformat()
        }
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


@app.post("/siparis-ekle", status_code=status.HTTP_201_CREATED, tags=["Müşteri İşlemleri"]) # Kimlik doğrulaması yok (müşteri yapıyor)
async def add_order_endpoint(data: SiparisEkleData):
    masa = data.masa
    sepet = data.sepet
    istek = data.istek
    yanit = data.yanit

    db_zaman_str = datetime.now(TR_TZ).strftime("%Y-%m-%d %H:%M:%S")
    yanit_zaman_iso_str = datetime.now(TR_TZ).isoformat()

    logger.info(f"📥 Yeni sipariş isteği alındı: Masa {masa}, {len(sepet)} çeşit ürün. DB Zaman: {db_zaman_str}. AI Yanıtı: {yanit[:200] if yanit else 'Yok'}...")

    cached_price_dict = await get_menu_price_dict()
    cached_stock_dict = await get_menu_stock_dict()

    processed_sepet = []
    for item in sepet:
        urun_adi_lower = item.urun.lower().strip()
        stok_kontrol_degeri = cached_stock_dict.get(urun_adi_lower)
        if stok_kontrol_degeri is None or stok_kontrol_degeri == 0: # Stokta olmayan veya tanımsız ürün
            logger.warning(f"⚠️ Stokta olmayan veya tanımsız ürün: '{item.urun}' (Masa: {masa}).")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"'{item.urun}' adlı ürün stokta yok veya menüde bulunmuyor.")

        item_dict = item.model_dump()
        cached_fiyat = cached_price_dict.get(urun_adi_lower, item.fiyat) # Cache'den fiyat al, yoksa gelen fiyatı kullan
        if cached_fiyat != item.fiyat:
            logger.warning(f"Fiyat uyuşmazlığı: Ürün '{item.urun}', Frontend Fiyatı: {item.fiyat}, Cache Fiyatı: {cached_fiyat}. Cache fiyatı kullanılacak.")
        item_dict['fiyat'] = cached_fiyat
        if item_dict['fiyat'] == 0 and item.fiyat == 0:
            logger.warning(f"⚠️ '{item.urun}' için fiyat 0.")
        processed_sepet.append(item_dict)

    if not processed_sepet:
        logger.warning(f"⚠️ Sipariş boş (Masa: {masa}).")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Sepette geçerli ürün yok.")

    istek_ozet = ", ".join([f"{p_item['adet']}x {p_item['urun']}" for p_item in processed_sepet])
    try:
        async with db.transaction():
            siparis_id = await db.fetch_val("""
                INSERT INTO siparisler (masa, istek, yanit, sepet, zaman, durum)
                VALUES (:masa, :istek, :yanit, :sepet, :zaman, 'bekliyor')
                RETURNING id
            """, {
                "masa": masa,
                "istek": istek or istek_ozet, # Eğer müşteri isteği yoksa, sipariş özetini istek olarak kaydet
                "yanit": yanit,
                "sepet": json.dumps(processed_sepet, ensure_ascii=False),
                "zaman": db_zaman_str
            })
            if siparis_id is None:
                logger.error(f"❌ Sipariş ID'si alınamadı. Masa: {masa}")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Sipariş kaydedilemedi.")

            siparis_bilgisi_ws = {
                "type": "siparis",
                "data": {"id": siparis_id, "masa": masa, "istek": istek or istek_ozet, "sepet": processed_sepet, "zaman": db_zaman_str, "durum": "bekliyor"}
            }
            await broadcast_message(aktif_mutfak_websocketleri, siparis_bilgisi_ws, "Mutfak/Masa")
            await broadcast_message(aktif_admin_websocketleri, siparis_bilgisi_ws, "Admin")
            await broadcast_message(aktif_kasa_websocketleri, siparis_bilgisi_ws, "Kasa")
            await update_table_status(masa, f"Sipariş verdi ({len(processed_sepet)} çeşit ürün)")
            logger.info(f"✅ Sipariş (ID: {siparis_id}) Masa: {masa} kaydedildi.")

            return {
                "mesaj": "Siparişiniz başarıyla alındı ve mutfağa iletildi.",
                "siparisId": siparis_id,
                "zaman": yanit_zaman_iso_str # Frontend'e ISO formatında gönder
            }
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"❌ Sipariş ekleme hatası (Masa: {masa}): {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Sipariş işlenirken sunucu hatası.")


@app.post("/siparis-guncelle", tags=["Siparişler"]) # Artık patch_order_endpoint kullanılıyor, bu kaldırılabilir veya yönlendirilebilir. Şimdilik yetkilendirme ekleyelim.
async def update_order_status_endpoint(
    data: SiparisGuncelleData,
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN, KullaniciRol.MUTFAK_PERSONELI, KullaniciRol.BARISTA]))
):
    logger.info(f"🔄 Sipariş durum güncelleme isteği (Kullanıcı: {current_user.kullanici_adi}): ID {data.id or 'Son'}, Masa {data.masa}, Durum {data.durum}")
    try:
        async with db.transaction():
            if data.id:
                query = "UPDATE siparisler SET durum = :durum WHERE id = :id RETURNING id, masa, durum, sepet, istek, zaman"
                values = {"durum": data.durum.value, "id": data.id}
            else: # ID yoksa, masanın son aktif siparişini güncelle
                query = """
                    UPDATE siparisler SET durum = :durum
                    WHERE id = (
                        SELECT id FROM siparisler
                        WHERE masa = :masa AND durum NOT IN ('hazir', 'iptal', 'odendi')
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

                notif_data = {
                    "id": updated_order_dict.get("id"),
                    "masa": updated_order_dict.get("masa"),
                    "durum": updated_order_dict.get("durum"),
                    "sepet": updated_order_dict.get("sepet"),
                    "istek": updated_order_dict.get("istek"),
                    "zaman": datetime.now(TR_TZ).isoformat() # Güncelleme zamanı
                }
                notification = {"type": "durum", "data": notif_data}
                await broadcast_message(aktif_mutfak_websocketleri, notification, "Mutfak/Masa")
                await broadcast_message(aktif_admin_websocketleri, notification, "Admin")
                await broadcast_message(aktif_kasa_websocketleri, notification, "Kasa")
                await update_table_status(updated_order_dict.get("masa", data.masa), f"Sipariş durumu güncellendi -> {data.durum.value} (by {current_user.kullanici_adi})")
                logger.info(f"✅ Sipariş (ID: {updated_order_dict.get('id')}) durumu '{data.durum.value}' olarak güncellendi.")
                return {"message": f"Sipariş (ID: {updated_order_dict.get('id')}) durumu '{data.durum.value}' olarak güncellendi.", "data": updated_order_dict}
            else:
                logger.warning(f"⚠️ Güncellenecek sipariş bulunamadı: ID {data.id or 'Son'}, Masa {data.masa}")
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Güncellenecek uygun bir sipariş bulunamadı.")
    except Exception as e:
        logger.error(f"❌ Sipariş durumu güncelleme hatası (Masa: {data.masa}): {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Sipariş durumu güncellenirken bir hata oluştu.")

@app.get("/siparisler", tags=["Siparişler"]) # Admin, Kasiyer, Mutfak, Barista
async def get_orders_endpoint(
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN, KullaniciRol.KASIYER, KullaniciRol.MUTFAK_PERSONELI, KullaniciRol.BARISTA]))
):
    logger.info(f"📋 Tüm siparişler listeleniyor (Kullanıcı: {current_user.kullanici_adi})")
    try:
        orders_raw = await db.fetch_all("SELECT id, masa, istek, yanit, sepet, zaman, durum FROM siparisler ORDER BY id DESC")
        orders_data = []
        for row in orders_raw:
            order_dict = dict(row)
            try:
                sepet_str = order_dict.get('sepet')
                order_dict['sepet'] = json.loads(sepet_str if sepet_str else '[]')
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
async def init_db(): # Burası init_databases içinde çağrılıyor
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
                    durum TEXT DEFAULT 'bekliyor' CHECK(durum IN ('bekliyor', 'hazirlaniyor', 'hazir', 'iptal', 'odendi'))
                )""")
            await db.execute("""
                CREATE TABLE IF NOT EXISTS masa_durumlar (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    masa_id TEXT UNIQUE NOT NULL,
                    son_erisim TIMESTAMP NOT NULL,
                    aktif BOOLEAN DEFAULT TRUE,
                    son_islem TEXT
                )""")
            # YENİ: Kullanıcılar tablosu
            await db.execute("""
                CREATE TABLE IF NOT EXISTS kullanicilar (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kullanici_adi TEXT UNIQUE NOT NULL,
                    sifre_hash TEXT NOT NULL,
                    rol TEXT NOT NULL, -- KullaniciRol enum değerleri (admin, kasiyer, barista, mutfak_personeli)
                    aktif_mi BOOLEAN DEFAULT TRUE,
                    olusturulma_tarihi TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )""")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_siparisler_masa_zaman ON siparisler(masa, zaman DESC)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_masa_durumlar_erisim ON masa_durumlar(son_erisim DESC)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_kullanicilar_kullanici_adi ON kullanicilar(kullanici_adi)") # Yeni index

            # YENİ: Varsayılan admin kullanıcısını ekle (eğer yoksa)
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

        logger.info(f"✅ Ana veritabanı ({DB_PATH}) başarıyla doğrulandı/oluşturuldu.")
    except Exception as e:
        logger.critical(f"❌ Ana veritabanı başlatılırken kritik hata: {e}", exc_info=True)
        raise

async def init_menu_db(): # Değişiklik yok
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

# Menü Yönetimi (Fonksiyonlarda değişiklik yok, olduğu gibi kalıyor - get_menu_for_prompt_cached, get_menu_price_dict, get_menu_stock_dict)
@alru_cache(maxsize=1)
async def get_menu_for_prompt_cached() -> str:
    logger.info(">>> GET_MENU_FOR_PROMPT_CACHED ÇAĞRILIYOR...")
    try:
        if not menu_db.is_connected:
            logger.info(">>> get_menu_for_prompt_cached: menu_db BAĞLI DEĞİL, bağlanıyor...")
            await menu_db.connect()
        query = """
            SELECT k.isim as kategori_isim, m.ad as urun_ad FROM menu m
            JOIN kategoriler k ON m.kategori_id = k.id
            WHERE m.stok_durumu = 1 ORDER BY k.isim, m.ad
        """
        urunler_raw = await menu_db.fetch_all(query)
        if not urunler_raw:
            logger.warning(">>> get_menu_for_prompt_cached: Menü prompt için stokta olan HİÇ ÜRÜN BULUNAMADI.")
            return "Üzgünüz, şu anda menümüzde aktif ürün bulunmamaktadır."
        kategorili_menu: Dict[str, List[str]] = {}
        for row in urunler_raw:
            try:
                kategori_ismi = row['kategori_isim']
                urun_adi = row['urun_ad']
                if kategori_ismi and urun_adi:
                    kategorili_menu.setdefault(kategori_ismi, []).append(urun_adi)
            except Exception as e_row:
                logger.error(f"get_menu_for_prompt_cached: Satır işlenirken hata: {e_row}", exc_info=True)
        if not kategorili_menu:
            return "Üzgünüz, menü bilgisi şu anda düzgün bir şekilde formatlanamıyor."
        menu_aciklama_list = [f"- {kategori}: {', '.join(urun_listesi)}" for kategori, urun_listesi in kategorili_menu.items() if urun_listesi]
        if not menu_aciklama_list:
            return "Üzgünüz, menüde listelenecek ürün bulunamadı."
        menu_aciklama = "\n".join(menu_aciklama_list)
        logger.info(f"Menü prompt için başarıyla oluşturuldu ({len(kategorili_menu)} kategori).")
        return menu_aciklama
    except Exception as e:
        logger.error(f"❌ Menü prompt oluşturma hatası: {e}", exc_info=True)
        return "Teknik bir sorun nedeniyle menü bilgisine şu anda ulaşılamıyor."

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
        if not menu_db.is_connected:
            await menu_db.connect()
        stocks_raw = await menu_db.fetch_all("SELECT ad, stok_durumu FROM menu")
        if not stocks_raw:
            logger.warning(">>> get_menu_stock_dict: Stok bilgisi için HİÇ ürün çekilemedi!")
            return {}
        stock_dict = {}
        for row in stocks_raw:
            try:
                stock_dict[str(row['ad']).lower().strip()] = int(row['stok_durumu'])
            except Exception as e_loop:
                logger.error(f"Stok sözlüğü oluştururken satır işleme hatası: {e_loop}", exc_info=True)
        logger.info(f">>> get_menu_stock_dict: Oluşturulan stock_dict ({len(stock_dict)} öğe).")
        return stock_dict
    except Exception as e_main:
        logger.error(f"❌ Stok sözlüğü oluşturma/alma sırasında genel hata: {e_main}", exc_info=True)
        return {}

SISTEM_MESAJI_ICERIK_TEMPLATE = ( # Değişiklik yok
    "Sen Fıstık Kafe için Neso adında, çok yetenekli bir sipariş asistanısın. "
    "Görevin, müşterilerin taleplerini doğru anlayıp, SANA VERİLEN STOKTAKİ ÜRÜNLER LİSTESİNDE yer alan ürünlerle eşleştirerek siparişlerini JSON formatında hazırlamaktır.\n\n"
    "# LANGUAGE DETECTION & RESPONSE\n"
    "1. Müşterinin kullandığı dili otomatik olarak algıla ve tüm metin yanıtlarını aynı dilde üret. "
    "Desteklediğin diller: Türkçe, English, العربية, Deutsch, Français, Español vb.\n"
    "2. İlk karşılamada ve hatırlatmalarda yine bu dilde selamlaş ve nazik ol:\n"
    "   - Türkçe: “Merhaba, ben Neso! Size nasıl yardımcı olabilirim?”\n"
    "   - English: “Hello, I’m Neso! How can I assist you today?”\n\n"
    "# STOKTAKİ ÜRÜNLER\n"
    "STOKTAKİ ÜRÜNLERİN TAM LİSTESİ (KATEGORİ: ÜRÜNLER):\n"
    "{menu_prompt_data}\n\n"
    "# ÖNEMLİ KURALLAR\n"
    "1. SADECE yukarıdaki listede varsa ürün kabul et. Hepsi stokta.\n"
    "2. Tam eşleşme olmasa bile (%75+ benzerlikle) en yakın ürünü seç. "
    "Müşterinin ek özelliklerini (sade, şekerli, büyük, dondurmalı, vb.) “musteri_notu” alanına ekle.\n"
    "   ÖRNEK: “2 sade türk kahvesi, 1 şekerli” ⇒ adet ve notları ayrı ayrı topla.\n"
    "3. Listede benzer ürün yoksa (örn. “pizza”), JSON ÜRETME; sadece nazikçe bildir: “Maalesef menümüzde pizza yok.”\n"
    "4. Ürün ve adetlerden emin değilsen önce onay sorusu sor (örn. “Türk kahveniz sade mi olsun?”).\n"
    "5. Fiyat ve kategori bilgilerini kesinlikle menü listesinden al, asla uydurma yapma.\n"
    "6. Toplam tutarı (adet × birim_fiyat) doğru hesapla.\n"
    "7. Müşteri soru soruyorsa (örn. “Menüde neler var?”), JSON üretme, sadece uygun yanıt ver. "
    "Menüyü kategorilere göre listele.\n\n"
    "# JSON ÇIKTISI\n"
    "Eğer sipariş net ve ürünler stokta ise, sadece aşağıdaki formatta JSON ver, başka hiçbir şey yazma:\n"
    "{{\n"
    '  "sepet": [\n'
    '    {{\n'
    '      "urun": "MENÜDEKİ TAM ÜRÜN ADI",\n'
    '      "adet": ADET_SAYISI,\n'
    '      "fiyat": BIRIM_FIYAT,\n'
    '      "kategori": "KATEGORI_ADI"\n'
    '    }}\n'
    '  ],\n'
    '  "toplam_tutar": TOPLAM_TUTAR,\n'
    '  "musteri_notu": "EK ÖZELLİKLER (sade, şekerli, vb.) veya \'\'",\n'
    '  "konusma_metni": "Kısa, nazik onay mesajı (aynı dilde)."\n'
    "}}\n"
)

SYSTEM_PROMPT: Optional[Dict[str, str]] = None

async def update_system_prompt(): # Değişiklik yok
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
        if SYSTEM_PROMPT is None:
            current_system_content = SISTEM_MESAJI_ICERIK_TEMPLATE.format(menu_prompt_data=menu_data_for_prompt)
            SYSTEM_PROMPT = {"role": "system", "content": current_system_content}
            logger.warning(f"Fallback sistem mesajı (BEKLENMEDİK HATA sonrası update_system_prompt içinde) kullanılıyor.")

@app.get("/admin/clear-menu-caches", tags=["Admin İşlemleri"]) # Sadece Admin
async def clear_all_caches_endpoint(current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))):
    logger.info(f"Admin '{current_user.kullanici_adi}' tarafından manuel cache temizleme isteği alındı.")
    await update_system_prompt()
    return {"message": "Menü, fiyat ve stok cache'leri başarıyla temizlendi. Sistem promptu güncellendi."}

@app.get("/menu", tags=["Menü"]) # Kimlik doğrulaması yok, herkes erişebilir
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

@app.post("/menu/ekle", status_code=status.HTTP_201_CREATED, tags=["Menü Yönetimi"]) # Sadece Admin
async def add_menu_item_endpoint(
    item_data: MenuEkleData,
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"📝 Menüye yeni ürün ekleme isteği (Kullanıcı: {current_user.kullanici_adi}): {item_data.ad} ({item_data.kategori})")
    try:
        async with menu_db.transaction():
            await menu_db.execute("INSERT OR IGNORE INTO kategoriler (isim) VALUES (:isim)", {"isim": item_data.kategori})
            category_id_row = await menu_db.fetch_one("SELECT id FROM kategoriler WHERE isim = :isim", {"isim": item_data.kategori})
            if not category_id_row:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Kategori oluşturulurken bir sorun oluştu.")
            category_id = category_id_row['id']
            try:
                item_id = await menu_db.fetch_val("""
                    INSERT INTO menu (ad, fiyat, kategori_id, stok_durumu)
                    VALUES (:ad, :fiyat, :kategori_id, 1)
                    RETURNING id
                """, {"ad": item_data.ad, "fiyat": item_data.fiyat, "kategori_id": category_id})
            except sqlite3.IntegrityError as ie:
                if "UNIQUE constraint failed" in str(ie).lower():
                    logger.warning(f"Menü ekleme başarısız: '{item_data.ad}' ürünü '{item_data.kategori}' kategorisinde zaten mevcut.")
                    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"'{item_data.ad}' adlı ürün bu kategoride zaten mevcut.")
                raise
            except Exception as e_db: # databases.IntegrityError vs.
                 if hasattr(e_db, 'args') and "UNIQUE constraint failed" in str(e_db.args[0]).lower():
                    logger.warning(f"Menü ekleme başarısız (DB Exception): '{item_data.ad}' ürünü '{item_data.kategori}' kategorisinde zaten mevcut.")
                    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"'{item_data.ad}' adlı ürün bu kategoride zaten mevcut.")
                 raise e_db

        await update_system_prompt() # Cache'i ve sistem mesajını güncelle
        logger.info(f"✅ '{item_data.ad}' menüye başarıyla eklendi (ID: {item_id}). Sistem mesajı güncellendi.")
        return {"mesaj": f"'{item_data.ad}' ürünü menüye başarıyla eklendi.", "itemId": item_id}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"❌ Menüye ürün eklenirken beklenmedik genel hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Menüye ürün eklenirken sunucuda bir hata oluştu.")


@app.delete("/menu/sil", tags=["Menü Yönetimi"]) # Sadece Admin
async def delete_menu_item_endpoint(
    urun_adi: str = Query(..., min_length=1, description="Silinecek ürünün tam adı."),
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"🗑️ Menüden ürün silme isteği (Kullanıcı: {current_user.kullanici_adi}): {urun_adi}")
    try:
        async with menu_db.transaction():
            item_to_delete = await menu_db.fetch_one("SELECT id FROM menu WHERE ad = :ad COLLATE NOCASE", {"ad": urun_adi})
            if not item_to_delete:
                logger.warning(f"Silinecek ürün bulunamadı: '{urun_adi}'")
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{urun_adi}' adlı ürün menüde bulunamadı.")
            await menu_db.execute("DELETE FROM menu WHERE id = :id", {"id": item_to_delete['id']})
        await update_system_prompt() # Cache'i ve sistem mesajını güncelle
        logger.info(f"✅ '{urun_adi}' menüden başarıyla silindi. Sistem mesajı güncellendi.")
        return {"mesaj": f"'{urun_adi}' ürünü menüden başarıyla silindi."}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"❌ Menüden ürün silinirken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Menüden ürün silinirken bir sunucu hatası oluştu.")

# AI Yanıt (Değişiklik yok - kimlik doğrulaması gerektirmiyor)
@app.post("/yanitla", tags=["Yapay Zeka"])
async def handle_message_endpoint(request: Request, data: dict = Body(...)):
    user_message = data.get("text", "").strip()
    table_id = data.get("masa", "bilinmiyor") # Masa asistanından geliyor
    session_id = request.session.get("session_id")

    if not session_id:
        session_id = secrets.token_hex(16)
        request.session["session_id"] = session_id
        request.session["chat_history"] = []
        logger.info(f"Yeni session başlatıldı: {session_id} Masa: {table_id}")

    chat_history = request.session.get("chat_history", [])
    logger.info(f"💬 AI Yanıt isteği: Masa '{table_id}', Session ID: '{session_id}', Kullanıcı Mesajı: '{user_message}'")

    if not user_message:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Mesaj boş olamaz.")
    if SYSTEM_PROMPT is None:
        logger.error("❌ AI Yanıt: Sistem promptu yüklenmemiş!")
        await update_system_prompt()
        if SYSTEM_PROMPT is None:
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="AI asistanı şu anda hazır değil (sistem mesajı eksik).")

    try:
        messages_for_openai = [SYSTEM_PROMPT] + chat_history + [{"role": "user", "content": user_message}]
        logger.debug(f"OpenAI'ye gönderilecek mesajlar (ilk mesaj hariç son 3): {messages_for_openai[-3:] if len(messages_for_openai) > 3 else messages_for_openai}")
        response = openai_client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages_for_openai, # type: ignore
            temperature=0.3,
            max_tokens=450,
        )
        ai_reply_content = response.choices[0].message.content
        ai_reply = ai_reply_content.strip() if ai_reply_content else "Üzgünüm, şu anda bir yanıt üretemiyorum."
        if not ai_reply_content: logger.warning("OpenAI'den boş yanıt (None) alındı.")

        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": ai_reply})
        request.session["chat_history"] = chat_history[-10:]

        logger.info(f"🤖 HAM AI Yanıtı (Masa: {table_id}, Session: {session_id}): {ai_reply}")
        return {"reply": ai_reply, "sessionId": session_id}
    except OpenAIError as e:
        logger.error(f"❌ OpenAI API ile iletişim hatası (Masa: {table_id}): {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"AI servisinden yanıt alınırken bir sorun oluştu: {type(e).__name__}")
    except Exception as e:
        logger.error(f"❌ AI yanıt endpoint'inde beklenmedik hata (Masa: {table_id}): {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Mesajınız işlenirken beklenmedik bir sunucu hatası oluştu.")

# İstatistikler (Fonksiyonlarda değişiklik yok, sadece endpoint'ler yetkilendirilecek)
def calculate_statistics(orders_data: List[dict]) -> tuple[int, int, float]: # Değişiklik yok
    total_orders_count = len(orders_data)
    total_items_sold = 0
    total_revenue = 0.0
    for order_row in orders_data:
        try:
            sepet_items_str = order_row.get('sepet')
            items = []
            if isinstance(sepet_items_str, str):
                if sepet_items_str.strip(): items = json.loads(sepet_items_str)
            elif isinstance(sepet_items_str, list):
                items = sepet_items_str
            if not isinstance(items, list):
                logger.warning(f"⚠️ İstatistik: Sepet öğesi beklenen liste formatında değil: {type(items)} - Sipariş ID: {order_row.get('id')}")
                items = []
            for item in items:
                if isinstance(item, dict):
                    adet = item.get("adet", 0)
                    fiyat = item.get("fiyat", 0.0)
                    if isinstance(adet, (int, float)) and isinstance(fiyat, (int, float)):
                        total_items_sold += int(adet)
                        total_revenue += adet * fiyat
                    else:
                        logger.warning(f"⚠️ İstatistik: Sepet öğesinde geçersiz adet/fiyat: {item} - Sipariş ID: {order_row.get('id')}")
                else:
                    logger.warning(f"⚠️ İstatistik: Sepet öğesi dict değil: {item} - Sipariş ID: {order_row.get('id')}")
        except json.JSONDecodeError:
            logger.warning(f"⚠️ İstatistik: Sepet JSON parse hatası. Sipariş ID: {order_row.get('id')}, Sepet Verisi (ilk 50 krkt): {str(order_row.get('sepet'))[:50]}")
        except KeyError:
            logger.warning(f"⚠️ İstatistik: 'sepet' anahtarı bulunamadı veya başka bir key hatası. Sipariş ID: {order_row.get('id')}")
        except Exception as e:
            logger.error(f"⚠️ İstatistik hesaplama sırasında beklenmedik hata: {e} - Sipariş ID: {order_row.get('id')}", exc_info=True)
    return total_orders_count, total_items_sold, round(total_revenue, 2)

@app.get("/admin/aktif-masa-tutarlari", response_model=List[AktifMasaOzet], tags=["Admin İşlemleri"]) # Sadece Admin
async def get_aktif_masa_tutarlari_endpoint(current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))):
    logger.info(f"📊 Admin '{current_user.kullanici_adi}': Aktif masa tutarları isteniyor.")
    try:
        query = f"""
            SELECT masa, id, sepet, durum, zaman
            FROM siparisler
            WHERE durum IN ('{Durum.BEKLIYOR.value}', '{Durum.HAZIRLANIYOR.value}', '{Durum.HAZIR.value}')
            ORDER BY masa, zaman ASC
        """
        aktif_siparisler_raw = await db.fetch_all(query)

        if not aktif_siparisler_raw:
            return []

        masalar_data: Dict[str, Dict[str, Any]] = {}
        for row_dict in [dict(row) for row in aktif_siparisler_raw]:
            masa_id = row_dict["masa"]
            if masa_id not in masalar_data:
                masalar_data[masa_id] = {"odenmemis_tutar": 0.0, "aktif_siparis_sayisi": 0}

            siparis_tutari = 0.0
            try:
                sepet_items_str = row_dict.get('sepet')
                sepet_items = json.loads(sepet_items_str if sepet_items_str else '[]')
                for item in sepet_items:
                    adet = item.get('adet', 0)
                    fiyat = item.get('fiyat', 0.0)
                    if isinstance(adet, (int, float)) and isinstance(fiyat, (int, float)):
                        siparis_tutari += adet * fiyat
            except json.JSONDecodeError:
                logger.warning(f"Aktif masa tutarları: Sepet JSON parse hatası. Sipariş ID: {row_dict.get('id')}")
            except Exception as e_item:
                logger.error(f"Aktif masa tutarları: Sepet öğesi işlenirken hata: {e_item}. Sipariş ID: {row_dict.get('id')}", exc_info=True)

            masalar_data[masa_id]["odenmemis_tutar"] += siparis_tutari
            masalar_data[masa_id]["aktif_siparis_sayisi"] += 1

        response_list = [
            AktifMasaOzet(
                masa_id=masa,
                odenmemis_tutar=round(data["odenmemis_tutar"], 2),
                aktif_siparis_sayisi=data["aktif_siparis_sayisi"]
            ) for masa, data in masalar_data.items()
        ]
        logger.info(f"✅ Aktif masa tutarları başarıyla hesaplandı ({len(response_list)} masa).")
        return response_list
    except Exception as e:
        logger.error(f"❌ Aktif masa tutarları alınırken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Aktif masa tutarları alınırken bir hata oluştu.")

@app.get("/istatistik/en-cok-satilan", tags=["İstatistikler"]) # Sadece Admin
async def get_popular_items_endpoint(
    limit: int = Query(5, ge=1, le=20),
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"📊 En çok satılan {limit} ürün istatistiği isteniyor (Kullanıcı: {current_user.kullanici_adi}).")
    item_counts: Dict[str, int] = {}
    try:
        orders_raw = await db.fetch_all("SELECT sepet FROM siparisler WHERE durum != 'iptal'") # Sadece iptal olmayanlar
        for row_record in orders_raw:
            row_as_dict = dict(row_record)
            try:
                sepet_items_str = row_as_dict.get('sepet')
                items = json.loads(sepet_items_str if sepet_items_str else '[]')
                if not isinstance(items, list): items = []

                for item in items:
                    if isinstance(item, dict):
                        item_name = item.get("urun")
                        quantity = item.get("adet", 0)
                        if item_name and isinstance(quantity, (int, float)) and quantity > 0:
                            item_counts[item_name] = item_counts.get(item_name, 0) + int(quantity)
            except json.JSONDecodeError:
                logger.warning(f"⚠️ Popüler ürünler: Sepet JSON parse hatası. Veri (ilk 50): {str(sepet_items_str)[:50] if sepet_items_str else 'Yok'}")
            except Exception as e_inner:
                logger.error(f"⚠️ Popüler ürünler: Sepet işleme sırasında beklenmedik iç hata: {e_inner} - Satır: {row_as_dict}", exc_info=True)
        sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        logger.info(f"✅ En çok satılan {len(sorted_items)} ürün bulundu.")
        return [{"urun": item, "adet": count} for item, count in sorted_items]
    except Exception as e_outer:
        logger.error(f"❌ Popüler ürünler istatistiği alınırken genel hata: {e_outer}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Popüler ürün istatistikleri alınamadı.")

async def get_stats_for_period(start_date_str: str, end_date_str: Optional[str] = None) -> dict: # Değişiklik yok
    start_datetime_str = f"{start_date_str} 00:00:00"
    query = "SELECT id, sepet, zaman FROM siparisler WHERE durum = 'odendi' AND zaman >= :start_dt" # Sadece 'odendi' durumundakiler
    values: Dict[str, any] = {"start_dt": start_datetime_str}
    if end_date_str:
        end_datetime_obj = datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1) # Bitiş tarihini dahil etmek için ertesi gün 00:00
        end_datetime_str = end_datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
        query += " AND zaman < :end_dt"
        values["end_dt"] = end_datetime_str
    orders_for_stats_records = await db.fetch_all(query, values)
    orders_list = [dict(record) for record in orders_for_stats_records]
    total_orders_count, total_items_sold, total_revenue = calculate_statistics(orders_list)
    return {
        "siparis_sayisi": total_orders_count,
        "satilan_urun_adedi": total_items_sold,
        "toplam_gelir": total_revenue,
        "veri_adedi": len(orders_list)
    }

@app.get("/istatistik/gunluk", tags=["İstatistikler"]) # Sadece Admin
async def get_daily_stats_endpoint(
    tarih: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="Belirli bir günün istatistiği (YYYY-AA-GG). Boş bırakılırsa bugünün istatistiği."),
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    target_date_str = tarih if tarih else datetime.now(TR_TZ).strftime("%Y-%m-%d")
    logger.info(f"📊 Günlük istatistik isteniyor (Kullanıcı: {current_user.kullanici_adi}): {target_date_str}")
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

@app.get("/istatistik/aylik", tags=["İstatistikler"]) # Sadece Admin
async def get_monthly_stats_endpoint(
    yil: Optional[int] = Query(None, ge=2000, le=datetime.now(TR_TZ).year + 1),
    ay: Optional[int] = Query(None, ge=1, le=12),
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    now = datetime.now(TR_TZ)
    target_year = yil if yil else now.year
    target_month = ay if ay else now.month
    logger.info(f"📊 Aylık istatistik isteniyor (Kullanıcı: {current_user.kullanici_adi}): {target_year}-{target_month:02d}")
    try:
        start_date = datetime(target_year, target_month, 1)
        if target_month == 12:
            end_date = datetime(target_year, 12, 31)
        else:
            end_date = datetime(target_year, target_month + 1, 1) - timedelta(days=1)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        stats = await get_stats_for_period(start_date_str, end_date_str)
        logger.info(f"✅ Aylık istatistik ({target_year}-{target_month:02d}) hesaplandı.")
        return {"yil": target_year, "ay": target_month, **stats}
    except ValueError as ve:
        logger.error(f"❌ Aylık istatistik: Geçersiz yıl/ay değeri: Yıl={yil}, Ay={ay}. Hata: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Geçersiz yıl veya ay değeri. {ve}")
    except Exception as e:
        logger.error(f"❌ Aylık istatistik ({target_year}-{target_month:02d}) alınırken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Aylık istatistikler alınamadı.")

@app.get("/istatistik/yillik-aylik-kirilim", tags=["İstatistikler"]) # Sadece Admin
async def get_yearly_stats_by_month_endpoint(
    yil: Optional[int] = Query(None, ge=2000, le=datetime.now(TR_TZ).year + 1),
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    target_year = yil if yil else datetime.now(TR_TZ).year
    logger.info(f"📊 Yıllık ({target_year}) aylık kırılımlı istatistik isteniyor (Kullanıcı: {current_user.kullanici_adi}).")
    try:
        start_of_year_str = f"{target_year}-01-01 00:00:00"
        end_of_year_exclusive_str = f"{target_year+1}-01-01 00:00:00"
        query = """
            SELECT id, sepet, zaman FROM siparisler
            WHERE durum = 'odendi' AND zaman >= :start AND zaman < :end_exclusive
            ORDER BY zaman ASC
        """ # Sadece 'odendi' durumundakiler
        orders_raw_records = await db.fetch_all(query, {"start": start_of_year_str, "end_exclusive": end_of_year_exclusive_str})
        monthly_stats: Dict[str, Dict[str, any]] = {}
        orders_as_dicts = [dict(record) for record in orders_raw_records]

        for row_dict in orders_as_dicts:
            try:
                order_time_str = row_dict.get('zaman', '')
                order_datetime = datetime.strptime(order_time_str.split('.')[0], "%Y-%m-%d %H:%M:%S") # Milisaniyeyi atla
                month_key = order_datetime.strftime("%Y-%m")

                if month_key not in monthly_stats:
                    monthly_stats[month_key] = {"siparis_sayisi": 0, "satilan_urun_adedi": 0, "toplam_gelir": 0.0}

                sepet_items_str = row_dict.get('sepet')
                items = json.loads(sepet_items_str if sepet_items_str else '[]')
                if not isinstance(items, list): items = []

                current_order_item_count = 0
                current_order_revenue = 0.0
                for item in items:
                    if isinstance(item, dict):
                        adet = item.get("adet", 0)
                        fiyat = item.get("fiyat", 0.0)
                        if isinstance(adet, (int,float)) and isinstance(fiyat, (int,float)):
                            current_order_item_count += int(adet)
                            current_order_revenue += adet * fiyat

                monthly_stats[month_key]["siparis_sayisi"] += 1
                monthly_stats[month_key]["satilan_urun_adedi"] += current_order_item_count
                monthly_stats[month_key]["toplam_gelir"] = round(monthly_stats[month_key]["toplam_gelir"] + current_order_revenue, 2)
            except json.JSONDecodeError:
                logger.warning(f"⚠️ Yıllık istatistik JSON parse hatası. Sipariş ID: {row_dict.get('id')}")
            except ValueError: # Zaman formatı hatası
                 logger.warning(f"Yıllık istatistik: Geçersiz zaman formatı: {order_time_str} Sipariş ID: {row_dict.get('id')}")
            except Exception as e_inner:
                logger.error(f"⚠️ Yıllık istatistik (aylık kırılım) iç döngü hatası: {e_inner}", exc_info=True)

        logger.info(f"✅ Yıllık ({target_year}) aylık kırılımlı istatistik hesaplandı ({len(monthly_stats)} ay).")
        return {"yil": target_year, "aylik_kirilim": dict(sorted(monthly_stats.items()))}
    except Exception as e:
        logger.error(f"❌ Yıllık ({target_year}) aylık kırılımlı istatistik alınırken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"{target_year} yılı için aylık kırılımlı istatistikler alınamadı.")


@app.get("/istatistik/filtreli", tags=["İstatistikler"]) # Sadece Admin
async def get_filtered_stats_endpoint(
    baslangic: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Başlangıç tarihi (YYYY-AA-GG)"),
    bitis: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Bitiş tarihi (YYYY-AA-GG)"),
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"📊 Filtreli istatistik isteniyor (Kullanıcı: {current_user.kullanici_adi}): {baslangic} - {bitis}")
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

# Sesli Yanıt (Değişiklik yok - kimlik doğrulaması gerektirmiyor)
SUPPORTED_LANGUAGES = {"tr-TR", "en-US", "en-GB", "fr-FR", "de-DE"}
@app.post("/sesli-yanit", tags=["Yapay Zeka"])
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
        voice_name = "tr-TR-Chirp3-HD-Laomedeia" if data.language == "tr-TR" else None
        if voice_name: logger.info(f"Türkçe için özel ses modeli seçildi: {voice_name}")

        voice_params = texttospeech.VoiceSelectionParams(
            language_code=data.language,
            name=voice_name,
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE if data.language == "tr-TR" and voice_name else texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=1.0)
        logger.debug(f"TTS İsteği -> Dil: {voice_params.language_code}, Ses Adı: {voice_params.name or 'Varsayılan'}, Cinsiyet: {voice_params.ssml_gender}, Encoding: MP3")

        response_tts = tts_client.synthesize_speech(input=synthesis_input, voice=voice_params, audio_config=audio_config)
        logger.info(f"✅ Sesli yanıt başarıyla oluşturuldu (Dil: {data.language}, Format: MP3, Ses: {voice_params.name or 'Varsayılan'}).")
        return Response(content=response_tts.audio_content, media_type="audio/mpeg")

    except google_exceptions.GoogleAPIError as e_google:
        logger.error(f"❌ Google TTS API hatası: {e_google}", exc_info=True)
        detail_msg = f"Google TTS servisinden ses üretilirken bir hata oluştu: {e_google.message if hasattr(e_google, 'message') else str(e_google)}"
        if "API key not valid" in str(e_google) or "permission" in str(e_google).lower() or "RESOURCE_EXHAUSTED" in str(e_google):
            detail_msg = "Google TTS servisi için kimlik/kota sorunu veya kaynak yetersiz."
        elif "Requested voice not found" in str(e_google) or "Invalid DefaultVoice" in str(e_google):
            detail_msg = f"İstenen ses modeli ({voice_name}) bulunamadı veya geçersiz. Lütfen ses adını kontrol edin."
            logger.error(detail_msg)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail_msg)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail_msg)
    except Exception as e:
        logger.error(f"❌ Sesli yanıt endpoint'inde beklenmedik hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Sesli yanıt oluşturulurken beklenmedik bir sunucu hatası oluştu.")

# Kasa İşlemleri
@app.post("/kasa/siparis/{siparis_id}/odendi", tags=["Kasa İşlemleri"]) # Admin, Kasiyer
async def mark_order_as_paid_endpoint(
    siparis_id: int = Path(..., description="Ödendi olarak işaretlenecek siparişin ID'si"),
    odeme_bilgisi: Optional[KasaOdemeData] = Body(None, description="Ödeme ile ilgili ek bilgiler (isteğe bağlı)"),
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN, KullaniciRol.KASIYER]))
):
    logger.info(f"💰 Kasa: Sipariş {siparis_id} ödendi olarak işaretleniyor (Kullanıcı: {current_user.kullanici_adi}). Ödeme bilgisi: {odeme_bilgisi}")
    try:
        async with db.transaction():
            order_check = await db.fetch_one(
                "SELECT id, masa, durum, sepet, istek, zaman FROM siparisler WHERE id = :id",
                {"id": siparis_id}
            )
            if not order_check:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sipariş bulunamadı.")
            if order_check["durum"] == Durum.ODENDI.value:
                logger.warning(f"Sipariş {siparis_id} zaten 'ödendi' olarak işaretlenmiş.")
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Sipariş zaten ödendi olarak işaretlenmiş.")
            if order_check["durum"] == Durum.IPTAL.value:
                logger.warning(f"İptal edilmiş sipariş ({siparis_id}) ödenemez.")
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="İptal edilmiş bir sipariş ödendi olarak işaretlenemez.")

            # TODO: Ödeme yöntemi gibi bilgiler de veritabanına eklenebilir.
            updated_order_query = """
                UPDATE siparisler SET durum = :yeni_durum WHERE id = :id
                RETURNING id, masa, durum, sepet, istek, zaman
            """
            updated_order = await db.fetch_one(updated_order_query, {"yeni_durum": Durum.ODENDI.value, "id": siparis_id})

        if not updated_order: # Normalde bu olmamalı, yukarıda kontrol edildi
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sipariş güncellenirken bir sorun oluştu veya bulunamadı.")

        order_dict = dict(updated_order)
        try:
            order_dict["sepet"] = json.loads(order_dict.get("sepet", "[]"))
        except json.JSONDecodeError:
            order_dict["sepet"] = []
            logger.warning(f"Sipariş {siparis_id} sepet JSON parse hatası (mark_order_as_paid_endpoint).")

        notif_data = {
            "id": order_dict["id"], "masa": order_dict["masa"], "durum": order_dict["durum"],
            "sepet": order_dict["sepet"], "istek": order_dict["istek"],
            "zaman": datetime.now(TR_TZ).isoformat(), # Ödeme zamanı olarak güncel zaman
            "odeme_yontemi": odeme_bilgisi.odeme_yontemi if odeme_bilgisi and odeme_bilgisi.odeme_yontemi else None
        }
        notification = {"type": "durum", "data": notif_data}
        await broadcast_message(aktif_mutfak_websocketleri, notification, "Mutfak/Masa")
        await broadcast_message(aktif_admin_websocketleri, notification, "Admin")
        await broadcast_message(aktif_kasa_websocketleri, notification, "Kasa")

        await update_table_status(order_dict["masa"], f"Sipariş {siparis_id} ödendi (Kullanıcı: {current_user.kullanici_adi})")
        logger.info(f"Sipariş {siparis_id} (Masa: {order_dict['masa']}) başarıyla 'ödendi' olarak işaretlendi.")
        return {"message": f"Sipariş {siparis_id} başarıyla ödendi olarak işaretlendi.", "data": order_dict}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"❌ Kasa: Sipariş {siparis_id} ödendi olarak işaretlenirken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Sipariş durumu güncellenirken bir sunucu hatası oluştu.")

@app.get("/kasa/odemeler", tags=["Kasa İşlemleri"]) # Admin, Kasiyer
async def get_payable_orders_endpoint(
    durum: Optional[str] = Query(None, description=f"Filtrelenecek sipariş durumu (örn: {Durum.HAZIR.value}). Boş bırakılırsa 'hazir' ve 'bekliyor' listelenir."),
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN, KullaniciRol.KASIYER]))
):
    logger.info(f"💰 Kasa: Ödeme bekleyen siparişler listeleniyor (Kullanıcı: {current_user.kullanici_adi}, Filtre: {durum}).")
    try:
        base_query_str = "SELECT id, masa, istek, sepet, zaman, durum FROM siparisler WHERE "
        values = {}

        valid_statuses_for_filter = [s.value for s in Durum if s not in [Durum.IPTAL, Durum.ODENDI]]
        if durum:
            if durum not in valid_statuses_for_filter:
                 raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Geçersiz durum filtresi. Kullanılabilecekler: {', '.join(valid_statuses_for_filter)}")
            query = base_query_str + "durum = :durum ORDER BY zaman ASC"
            values = {"durum": durum}
        else:
            # Varsayılan olarak 'hazir', 'bekliyor', 'hazirlaniyor' (yani ödenmemiş ve iptal olmayan)
            allowed_payable_statuses = "','".join([Durum.HAZIR.value, Durum.BEKLIYOR.value, Durum.HAZIRLANIYOR.value])
            query = base_query_str + f"durum IN ('{allowed_payable_statuses}') ORDER BY zaman ASC"

        orders_raw = await db.fetch_all(query, values)
        orders_data = []
        for row in orders_raw:
            order_dict = dict(row)
            try:
                order_dict['sepet'] = json.loads(order_dict.get('sepet', '[]'))
            except json.JSONDecodeError:
                order_dict['sepet'] = []
            orders_data.append(order_dict)
        return {"orders": orders_data}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"❌ Kasa: Ödeme bekleyen siparişler alınırken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Siparişler alınırken bir hata oluştu.")

@app.get("/kasa/masa/{masa_id}/hesap", tags=["Kasa İşlemleri"]) # Admin, Kasiyer
async def get_table_bill_endpoint(
    masa_id: str = Path(..., description="Hesabı istenen masa numarası/adı."),
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN, KullaniciRol.KASIYER]))
):
    logger.info(f"💰 Kasa: Masa {masa_id} için hesap isteniyor (Kullanıcı: {current_user.kullanici_adi}).")
    try:
        allowed_bill_statuses = "','".join([Durum.BEKLIYOR.value, Durum.HAZIRLANIYOR.value, Durum.HAZIR.value])
        query = f"""
            SELECT id, masa, istek, sepet, zaman, durum, yanit
            FROM siparisler
            WHERE masa = :masa_id AND durum IN ('{allowed_bill_statuses}')
            ORDER BY zaman ASC
        """
        orders_raw = await db.fetch_all(query, {"masa_id": masa_id})

        orders_data = []
        toplam_tutar = 0.0
        if not orders_raw:
            logger.info(f"Masa {masa_id} için aktif ödenmemiş sipariş bulunamadı.")

        for row in orders_raw:
            order_dict = dict(row)
            try:
                sepet_items = json.loads(order_dict.get('sepet', '[]'))
                order_dict['sepet'] = sepet_items
                for item in sepet_items:
                    adet = item.get('adet', 0)
                    fiyat = item.get('fiyat', 0.0)
                    if isinstance(adet, (int, float)) and isinstance(fiyat, (int, float)):
                        toplam_tutar += adet * fiyat
                    else:
                        logger.warning(f"Masa hesabı: Ürün '{item.get('urun', 'Bilinmeyen')}' için adet ({adet}) veya fiyat ({fiyat}) geçersiz. Sipariş ID: {order_dict.get('id')}")
            except json.JSONDecodeError:
                order_dict['sepet'] = []
                logger.warning(f"Masa hesabı: Sepet JSON parse hatası. Sipariş ID: {order_dict.get('id')}")
            except Exception as e_item:
                logger.error(f"Masa hesabı: Sepet öğesi işlenirken hata: {e_item}. Sipariş ID: {order_dict.get('id')}", exc_info=True)
            orders_data.append(order_dict)

        return {"masa_id": masa_id, "siparisler": orders_data, "toplam_tutar": round(toplam_tutar, 2)}
    except Exception as e:
        logger.error(f"❌ Kasa: Masa {masa_id} hesabı alınırken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Masa hesabı alınırken bir hata oluştu.")

# Admin Şifre Değiştirme Endpoint'i (Artık DB üzerinden yönetilecek, bu endpoint kaldırılabilir veya güncellenebilir)
@app.post("/admin/sifre-degistir-eski", deprecated=True, include_in_schema=False) # Sadece Admin
async def change_admin_password_endpoint_eski(
    creds: AdminCredentialsUpdate,
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN])) # Sadece Admin
):
    logger.warning(f"ℹ️ ESKİ Admin şifre/kullanıcı adı değiştirme endpoint'i çağrıldı (Kullanıcı: {creds.yeniKullaniciAdi}). Bu işlem artık doğrudan veritabanı üzerinden yapılmalıdır.")
    return {
        "mesaj": "Bu endpoint kullanımdan kaldırılmıştır. Admin kullanıcı adı ve şifresi artık doğrudan veritabanı üzerinden veya yeni kullanıcı yönetimi endpoint'leri (oluşturulacaksa) aracılığıyla yönetilmelidir."
    }

# --- YENİ: Kullanıcı Yönetimi Endpointleri (Admin için) ---
# Bu kısım bir sonraki adımda eklenebilir. Şimdilik sadece Admin rolüne sahip
# kullanıcının var olduğundan ve giriş yapabildiğinden emin oluyoruz.
# Örnekler:
# @app.post("/admin/kullanicilar", response_model=Kullanici, status_code=status.HTTP_201_CREATED, tags=["Kullanıcı Yönetimi"])
# async def create_new_user(...): ...
# @app.get("/admin/kullanicilar", response_model=List[Kullanici], tags=["Kullanıcı Yönetimi"])
# async def list_all_users(...): ...
# @app.put("/admin/kullanicilar/{user_id}", response_model=Kullanici, tags=["Kullanıcı Yönetimi"])
# async def update_existing_user(...): ...
# @app.delete("/admin/kullanicilar/{user_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Kullanıcı Yönetimi"])
# async def delete_existing_user(...): ...


if __name__ == "__main__":
    import uvicorn
    host_ip = os.getenv("HOST", "127.0.0.1")
    port_num = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 FastAPI uygulaması {host_ip}:{port_num} adresinde başlatılıyor (yerel geliştirme modu)...")
    uvicorn.run("main:app", host=host_ip, port=port_num, reload=True)