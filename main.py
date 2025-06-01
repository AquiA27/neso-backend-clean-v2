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
    version="1.4.0", # Sürüm güncellendi
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

db = Database(DATABASE_CONNECTION_STRING) # Ana DB
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
menu_db = Database(MENU_DATABASE_CONNECTION_STRING) # Menü DB

try:
    if not DATABASE_CONNECTION_STRING.startswith("sqlite:///"):
        logger.info(f"PostgreSQL veya benzeri bir veritabanı kullanılıyor. '{settings.DB_DATA_DIR}' dizini SQLite için oluşturulmayacak.")
    elif settings.DB_DATA_DIR != ".": # pragma: no cover
        os.makedirs(settings.DB_DATA_DIR, exist_ok=True)
        logger.info(f"SQLite için '{settings.DB_DATA_DIR}' dizini kontrol edildi/oluşturuldu.")
except OSError as e: # pragma: no cover
    logger.error(f"'{settings.DB_DATA_DIR}' dizini oluşturulurken hata: {e}.")

TR_TZ = dt_timezone(timedelta(hours=3))

# --- Pydantic Modelleri ---

class StokDegerKategoriDetay(BaseModel):
    stok_kategori_ad: str
    kategori_toplam_deger: float
    kalem_sayisi: int

class StokDegerRaporuResponse(BaseModel):
    rapor_tarihi: datetime
    genel_toplam_stok_degeri: float
    kategori_bazli_degerler: List[StokDegerKategoriDetay]
    degeri_hesaplanamayan_kalem_sayisi: int # Son alış fiyatı olmayan veya miktarı sıfır olanlar

#Saatlik yoğunluk
class SaatlikYogunlukDetay(BaseModel):
    saat: int # 0-23 arası saat dilimi
    siparis_sayisi: int
    toplam_gelir: float

class SaatlikYogunlukResponse(BaseModel):
    tarih: VeliDate # Hangi günün analizi olduğu veya analiz periyodunun özeti
    saatlik_veri: List[SaatlikYogunlukDetay]
    # İsteğe bağlı olarak genel toplamlar da eklenebilir
    # genel_siparis_sayisi: int
    # genel_toplam_gelir: float

# Ortalama sepet tutarı

class OrtalamaSepetTutariResponse(BaseModel):
    baslangic_tarihi: VeliDate
    bitis_tarihi: VeliDate
    toplam_gelir: float
    toplam_siparis_sayisi: int
    ortalama_sepet_tutari: float

# Günlük Gelir Detayı
class GunlukIstatistik(BaseModel): # Eski IstatistikBase'i override ediyoruz
    tarih: str
    siparis_sayisi: int
    toplam_gelir: float
    satilan_urun_adedi: int
    nakit_gelir: Optional[float] = 0.0
    kredi_karti_gelir: Optional[float] = 0.0
    diger_odeme_yontemleri_gelir: Optional[float] = 0.0

# Menü Kategori Yönetimi
class MenuKategoriBase(BaseModel):
    isim: str = Field(..., min_length=1, max_length=100)

class MenuKategoriCreate(MenuKategoriBase):
    pass

class MenuKategori(MenuKategoriBase):
    id: int
    class Config:
        from_attributes = True
        

#GunlukIstatistik
class SatisRaporuUrunDetay(BaseModel):
    urun_adi: str
    kategori_adi: Optional[str] = "Bilinmiyor"
    satilan_adet: int
    toplam_gelir: float

class SatisRaporuKategoriDetay(BaseModel):
    kategori_adi: str
    satilan_adet: int
    toplam_gelir: float

class SatisRaporuResponse(BaseModel):
    baslangic_tarihi: VeliDate # 'date' yerine alias 'VeliDate' kullandık
    bitis_tarihi: VeliDate   # 'date' yerine alias 'VeliDate' kullandık
    urun_bazli_satislar: List[SatisRaporuUrunDetay]
    kategori_bazli_satislar: List[SatisRaporuKategoriDetay]
    genel_toplam_gelir: float
    genel_toplam_adet: int       

# Stok Yönetimi için Modeller (Temel)
class StokKategoriBase(BaseModel):
    ad: str = Field(..., min_length=1, max_length=100)

class StokKategoriCreate(StokKategoriBase):
    pass

class StokKategori(StokKategoriBase):
    id: int
    class Config:
        from_attributes = True

class StokKalemiBase(BaseModel):
    ad: str = Field(..., min_length=1, max_length=150)
    stok_kategori_id: int
    birim: str = Field(..., min_length=1, max_length=20) # kg, lt, adet, paket vb.
    min_stok_seviyesi: float = Field(default=0, ge=0)

class StokKalemiCreate(StokKalemiBase):
    mevcut_miktar: float = Field(default=0, ge=0)
    son_alis_fiyati: Optional[float] = Field(default=None, ge=0)

class StokKalemiUpdate(BaseModel):
    ad: Optional[str] = Field(None, min_length=1, max_length=150)
    stok_kategori_id: Optional[int] = None
    birim: Optional[str] = Field(None, min_length=1, max_length=20)
    min_stok_seviyesi: Optional[float] = Field(None, ge=0)

class StokKalemi(StokKalemiBase):
    id: int
    mevcut_miktar: float = 0.0
    son_alis_fiyati: Optional[float] = None
    stok_kategori_ad: Optional[str] = None # Görüntüleme için eklenebilir
    class Config:
        from_attributes = True

# YENİ EKLENEN KISIM: Reçete Sistemi için Pydantic Modelleri
class StokKalemiSimple(BaseModel): # Reçete oluştururken/görüntülerken stok kalemlerini listelemek için
    id: int
    ad: str
    birim: str # Stok kaleminin ana birimi (örn: kg, lt, adet)
    class Config:
        from_attributes = True

class MenuUrunuSimple(BaseModel): # Reçete oluştururken/görüntülerken menü ürünlerini listelemek için
    id: int # Bu ID menu_db'deki menu.id'ye karşılık gelir
    ad: str
    kategori_ad: Optional[str] = None # Menü ürününün kategorisi (menu_db'den join ile alınabilir)
    class Config:
        from_attributes = True

class ReceteBileseniBase(BaseModel):
    stok_kalemi_id: int = Field(..., description="Bu bileşenin hangi stok kalemine ait olduğu (stok_kalemleri.id)")
    miktar: float = Field(..., gt=0, description="Bu stok kaleminden ne kadar kullanılacağı")
    birim: str = Field(..., min_length=1, max_length=30, description="Kullanılan miktarın birimi (örn: gram, ml, adet, çay kaşığı)")

class ReceteBileseniCreate(ReceteBileseniBase):
    pass

class ReceteBileseni(ReceteBileseniBase):
    id: int
    stok_kalemi_ad: Optional[str] = Field(None, description="Stok kaleminin adı (gösterim için, join ile stok_kalemleri tablosundan alınacak)")
    class Config:
        from_attributes = True

class MenuUrunRecetesiBase(BaseModel):
    menu_urun_id: int = Field(..., description="Reçetesi oluşturulan menü ürününün ID'si (menu_db.menu.id)")
    aciklama: Optional[str] = Field(None, max_length=500, description="Reçete için genel açıklamalar veya hazırlama notları")
    porsiyon_birimi: str = Field(default="adet", max_length=50, description="Reçetenin ürettiği porsiyonun birimi (örn: adet, tabak, bardak, kg, porsiyon)")
    porsiyon_miktari: float = Field(default=1.0, gt=0, description="Bu reçete ile kaç birim porsiyon üretildiği")

class MenuUrunRecetesiCreate(MenuUrunRecetesiBase):
    bilesenler: List[ReceteBileseniCreate] = Field(..., min_items=1, description="Reçeteyi oluşturan stok kalemleri ve miktarları")

class MenuUrunRecetesi(MenuUrunRecetesiBase):
    id: int
    menu_urun_ad: Optional[str] = Field(None, description="Menü ürününün adı (gösterim için, menu_db'den alınacak)")
    bilesenler: List[ReceteBileseni] = []
    olusturulma_tarihi: datetime
    guncellenme_tarihi: datetime
    class Config:
        from_attributes = True
# YENİ EKLENEN KISIM SONU (Pydantic Modelleri)

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
        # Ana DB bağlantısı
        if not db.is_connected: # pragma: no cover
            await db.connect()
            logger.info("✅ Ana veritabanı bağlantısı (db) kuruldu.")

        # Menü DB bağlantısı (eğer ana DB'den farklıysa ve bağlı değilse)
        if menu_db != db:
            if not menu_db.is_connected: # pragma: no cover
                await menu_db.connect()
                logger.info("✅ Menü veritabanı bağlantısı (menu_db) kuruldu.")

        logger.info("Veritabanı tabloları başlatılıyor...")
        await init_databases() # Bu fonksiyon tüm init_*_db fonksiyonlarını çağırır

        await update_system_prompt()
        logger.info(f"🚀 FastAPI uygulaması başlatıldı. Sistem mesajı güncellendi.")
    except Exception as e_startup: # pragma: no cover
        logger.critical(f"❌ Uygulama başlangıcında KRİTİK HATA: {e_startup}", exc_info=True)
        if menu_db != db and menu_db.is_connected: await menu_db.disconnect()
        if db.is_connected: await db.disconnect()
        raise SystemExit(f"Uygulama başlatılamadı: {e_startup}")

@app.on_event("shutdown")
async def shutdown_event(): # pragma: no cover
    logger.info("🚪 Uygulama kapatılıyor...")
    try:
        if menu_db != db and menu_db.is_connected:
            await menu_db.disconnect()
            logger.info("✅ Menü veritabanı bağlantısı (menu_db) kapatıldı.")
        if db.is_connected:
            await db.disconnect()
            logger.info("✅ Ana veritabanı bağlantısı (db) kapatıldı.")
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
        except RuntimeError: # pragma: no cover
            disconnected_ws.add(ws)
            logger.warning(f"⚠️ {ws_type_name} WS bağlantısı zaten kopuk (RuntimeError), listeden kaldırılıyor: {ws.client}")
        except Exception as e_send: # pragma: no cover
            disconnected_ws.add(ws)
            logger.warning(f"⚠️ {ws_type_name} WS gönderme sırasında BEKLENMEDİK hata ({ws.client}): {e_send}")
    for ws in disconnected_ws:
        connections.discard(ws)
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results): # pragma: no cover
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
            except json.JSONDecodeError: # pragma: no cover
                logger.warning(f"⚠️ {endpoint_name} WS: Geçersiz JSON formatında mesaj alındı: {data} from {client_info}")
            except Exception as e_inner: # pragma: no cover
                logger.error(f"❌ {endpoint_name} WS mesaj işleme hatası ({client_info}): {e_inner} - Mesaj: {data}", exc_info=True)
    except WebSocketDisconnect as e: # pragma: no cover
        if e.code == 1000 or e.code == 1001:
            logger.info(f"🔌 {endpoint_name} WS normal şekilde kapandı (Kod {e.code}): {client_info}")
        elif e.code == 1012:
            logger.warning(f"🔌 {endpoint_name} WS beklenmedik şekilde kapandı (Kod {e.code} - Sunucu Yeniden Başlıyor Olabilir): {client_info}")
        else:
            logger.warning(f"🔌 {endpoint_name} WS beklenmedik şekilde kapandı (Kod {e.code}): {client_info}")
    except Exception as e_outer: # pragma: no cover
        logger.error(f"❌ {endpoint_name} WS beklenmedik genel hata ({client_info}): {e_outer}", exc_info=True)
    finally:
        if websocket in connections:
            connections.discard(websocket)
        logger.info(f"📉 {endpoint_name} WS kaldırıldı: {client_info} (Kalan: {len(connections)})")

@app.websocket("/ws/admin")
async def websocket_admin_endpoint(websocket: WebSocket): # pragma: no cover
    await websocket_lifecycle(websocket, aktif_admin_websocketleri, "Admin")

@app.websocket("/ws/mutfak")
async def websocket_mutfak_endpoint(websocket: WebSocket): # pragma: no cover
    await websocket_lifecycle(websocket, aktif_mutfak_websocketleri, "Mutfak/Masa")

@app.websocket("/ws/kasa")
async def websocket_kasa_endpoint(websocket: WebSocket): # pragma: no cover
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
    except Exception as e: # pragma: no cover
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
    except Exception as e: # pragma: no cover
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
    if not user_in_db.aktif_mi: # pragma: no cover
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

class AylikIstatistik(BaseModel):
    ay: str
    siparis_sayisi: int
    toplam_gelir: float
    satilan_urun_adedi: int


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

@app.get("/istatistik/saatlik-yogunluk", response_model=SaatlikYogunlukResponse, tags=["İstatistikler"])
async def get_saatlik_yogunluk(
    tarih_str: Optional[str] = Query(None, description="Analiz yapılacak tarih (YYYY-MM-DD formatında). Boş bırakılırsa bugünün verileri alınır."),
    # Gelecekte bir tarih aralığı da desteklenebilir:
    # baslangic_tarihi_str: Optional[str] = Query(None, description="Başlangıç tarihi..."),
    # bitis_tarihi_str: Optional[str] = Query(None, description="Bitiş tarihi..."),
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"Admin '{current_user.kullanici_adi}' saatlik yoğunluk istatistiklerini istedi (Tarih: {tarih_str or 'Bugün'}).")

    target_date: VeliDate
    if tarih_str:
        try:
            target_date = datetime.strptime(tarih_str, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Geçersiz tarih formatı. YYYY-MM-DD kullanın.")
    else:
        target_date = datetime.now(TR_TZ).date()

    # Belirtilen günün başlangıç ve bitişini TR_TZ ile ayarla
    gun_baslangic_dt = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=TR_TZ)
    gun_bitis_dt = datetime.combine(target_date, datetime.max.time()).replace(tzinfo=TR_TZ)
    # veya gun_bitis_dt = gun_baslangic_dt + timedelta(days=1) - eğer zaman < bitis ise

    query = """
        SELECT sepet, zaman
        FROM siparisler
        WHERE durum = 'odendi' AND zaman >= :baslangic AND zaman < :bitis_exclusive
    """
    # gun_bitis_dt'yi bir sonraki günün başlangıcı olarak ayarlayıp < kullanmak daha güvenli olabilir
    bitis_exclusive_dt = gun_baslangic_dt + timedelta(days=1)

    odenen_siparisler = await db.fetch_all(
        query,
        {"baslangic": gun_baslangic_dt, "bitis_exclusive": bitis_exclusive_dt}
    )

    saatlik_analiz: Dict[int, Dict[str, Union[int, float]]] = {saat: {"siparis_sayisi": 0, "toplam_gelir": 0.0} for saat in range(24)}

    for siparis in odenen_siparisler:
        try:
            # Sipariş zamanını TR_TZ'ye göre al (eğer veritabanından UTC veya farklı bir TZ geliyorsa)
            siparis_zamani_tr = siparis["zaman"]
            if siparis_zamani_tr.tzinfo is None: # Eğer naive ise TR_TZ varsay
                 siparis_zamani_tr = siparis_zamani_tr.replace(tzinfo=TR_TZ)
            else: # Aware ise TR_TZ'ye çevir
                 siparis_zamani_tr = siparis_zamani_tr.astimezone(TR_TZ)
            
            saat = siparis_zamani_tr.hour
            
            saatlik_analiz[saat]["siparis_sayisi"] += 1
            
            sepet_items = json.loads(siparis["sepet"] or "[]")
            siparis_tutari = 0.0
            for item in sepet_items:
                adet = item.get("adet", 0)
                fiyat = item.get("fiyat", 0.0)
                siparis_tutari += adet * fiyat
            saatlik_analiz[saat]["toplam_gelir"] += siparis_tutari
            
        except json.JSONDecodeError:
            logger.warning(f"Saatlik analiz: Sepet parse hatası, Sipariş zamanı: {siparis['zaman']}, Sepet: {siparis['sepet']}")
            continue
        except Exception as e_loop:
            logger.error(f"Saatlik analiz döngüsünde beklenmedik hata: {e_loop}", exc_info=True)
            continue

    sonuc_listesi: List[SaatlikYogunlukDetay] = []
    for saat, veri in saatlik_analiz.items():
        sonuc_listesi.append(SaatlikYogunlukDetay(
            saat=saat,
            siparis_sayisi=veri["siparis_sayisi"],
            toplam_gelir=round(veri["toplam_gelir"], 2)
        ))

    # Saate göre sırala
    sonuc_listesi.sort(key=lambda x: x.saat)

    return SaatlikYogunlukResponse(
        tarih=target_date,
        saatlik_veri=sonuc_listesi
    )

@app.get("/istatistik/ortalama-sepet-tutari", response_model=OrtalamaSepetTutariResponse, tags=["İstatistikler"])
async def get_ortalama_sepet_tutari(
    baslangic_tarihi_str: str = Query(..., description="Başlangıç tarihi (YYYY-MM-DD formatında)"),
    bitis_tarihi_str: str = Query(..., description="Bitiş tarihi (YYYY-MM-DD formatında)"),
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(
        f"Admin '{current_user.kullanici_adi}' ortalama sepet tutarı istedi. "
        f"Tarih Aralığı: {baslangic_tarihi_str} - {bitis_tarihi_str}"
    )

    try:
        start_date = datetime.strptime(baslangic_tarihi_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(bitis_tarihi_str, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Geçersiz tarih formatı. Lütfen YYYY-MM-DD formatını kullanın."
        )

    if start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Başlangıç tarihi, bitiş tarihinden sonra olamaz."
        )

    # Tarih aralığını TR_TZ ile datetime objelerine çevir
    period_start_dt = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=TR_TZ)
    # Bitiş tarihinin sonunu dahil etmek için bir sonraki günün başlangıcını kullan
    period_end_dt_exclusive = datetime.combine(end_date + timedelta(days=1), datetime.min.time()).replace(tzinfo=TR_TZ)

    query = """
        SELECT sepet FROM siparisler
        WHERE durum = 'odendi' AND zaman >= :baslangic AND zaman < :bitis
    """
    odenen_siparisler_raw = await db.fetch_all(
        query,
        {"baslangic": period_start_dt, "bitis": period_end_dt_exclusive}
    )

    if not odenen_siparisler_raw:
        return OrtalamaSepetTutariResponse(
            baslangic_tarihi=start_date,
            bitis_tarihi=end_date,
            toplam_gelir=0.0,
            toplam_siparis_sayisi=0,
            ortalama_sepet_tutari=0.0
        )

    genel_toplam_gelir = 0.0
    toplam_siparis_sayisi = len(odenen_siparisler_raw)

    for siparis_raw in odenen_siparisler_raw:
        try:
            sepet_items = json.loads(siparis_raw["sepet"] or "[]")
            for item in sepet_items:
                adet = item.get("adet", 0)
                fiyat = item.get("fiyat", 0.0)
                if isinstance(adet, (int, float)) and isinstance(fiyat, (int, float)) and adet > 0 and fiyat >= 0:
                    genel_toplam_gelir += adet * fiyat
        except json.JSONDecodeError:
            logger.warning(f"Ortalama sepet tutarı hesaplanırken sepet parse hatası: Sepet: {siparis_raw['sepet']}")
            continue # Bu siparişin gelirini hesaplamaya katma
        except Exception as e_loop:
            logger.error(f"Ortalama sepet tutarı hesaplama döngüsünde beklenmedik hata: {e_loop}", exc_info=True)
            continue


    ortalama_tutar = (genel_toplam_gelir / toplam_siparis_sayisi) if toplam_siparis_sayisi > 0 else 0.0

    return OrtalamaSepetTutariResponse(
        baslangic_tarihi=start_date,
        bitis_tarihi=end_date,
        toplam_gelir=round(genel_toplam_gelir, 2),
        toplam_siparis_sayisi=toplam_siparis_sayisi,
        ortalama_sepet_tutari=round(ortalama_tutar, 2)
    )

@app.get("/admin/stok/deger-raporu", response_model=StokDegerRaporuResponse, tags=["Stok Yönetimi", "Admin İşlemleri"])
async def get_stok_deger_raporu(
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"Admin '{current_user.kullanici_adi}' stok değer raporunu istedi.")

    query = """
        SELECT sk.ad as stok_kalem_adi, sk.mevcut_miktar, sk.son_alis_fiyati, skat.ad as stok_kategori_ad
        FROM stok_kalemleri sk
        JOIN stok_kategorileri skat ON sk.stok_kategori_id = skat.id
    """
    tum_stok_kalemleri = await db.fetch_all(query)

    genel_toplam_stok_degeri: float = 0.0
    kategori_gecici_veriler: Dict[str, Dict[str, Any]] = {} # Kategori adı -> {"toplam_deger": x, "kalem_sayisi": y}
    degeri_hesaplanamayan_kalem_sayisi: int = 0

    for kalem in tum_stok_kalemleri:
        kategori_adi = kalem["stok_kategori_ad"]
        if kategori_adi not in kategori_gecici_veriler:
            kategori_gecici_veriler[kategori_adi] = {"toplam_deger": 0.0, "kalem_sayisi": 0}

        if kalem["son_alis_fiyati"] is not None and kalem["son_alis_fiyati"] > 0 and \
           kalem["mevcut_miktar"] is not None and kalem["mevcut_miktar"] > 0:
            
            kalem_degeri = kalem["mevcut_miktar"] * kalem["son_alis_fiyati"]
            genel_toplam_stok_degeri += kalem_degeri
            kategori_gecici_veriler[kategori_adi]["toplam_deger"] += kalem_degeri
            kategori_gecici_veriler[kategori_adi]["kalem_sayisi"] += 1
        else:
            degeri_hesaplanamayan_kalem_sayisi += 1
            # Değeri hesaplanamayan ama kategoride kalem olarak saymak isterseniz:
            # kategori_gecici_veriler[kategori_adi]["kalem_sayisi"] += 1 
            # (Yukarıdaki satır yoruma alınırsa sadece değeri olanlar kategori kalem sayısına dahil olur)


    kategori_bazli_liste: List[StokDegerKategoriDetay] = []
    for kat_ad, veri in kategori_gecici_veriler.items():
        kategori_bazli_liste.append(StokDegerKategoriDetay(
            stok_kategori_ad=kat_ad,
            kategori_toplam_deger=round(veri["toplam_deger"], 2),
            kalem_sayisi=veri["kalem_sayisi"]
        ))
    
    # Kategori adına göre sırala (isteğe bağlı)
    kategori_bazli_liste.sort(key=lambda x: x.stok_kategori_ad)

    return StokDegerRaporuResponse(
        rapor_tarihi=datetime.now(TR_TZ),
        genel_toplam_stok_degeri=round(genel_toplam_stok_degeri, 2),
        kategori_bazli_degerler=kategori_bazli_liste,
        degeri_hesaplanamayan_kalem_sayisi=degeri_hesaplanamayan_kalem_sayisi
    )

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
            except ValueError: # pragma: no cover
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Geçersiz tarih formatı. YYYY-MM-DD kullanın.")
        else:
            gun_baslangic_dt = datetime.now(TR_TZ).replace(hour=0, minute=0, second=0, microsecond=0)

        gun_bitis_dt = gun_baslangic_dt + timedelta(days=1)

        query = """
            SELECT sepet, durum, odeme_yontemi FROM siparisler
            WHERE zaman >= :baslangic AND zaman < :bitis AND durum = 'odendi'
        """
        odenen_siparisler = await db.fetch_all(query, {"baslangic": gun_baslangic_dt, "bitis": gun_bitis_dt})

        siparis_sayisi = len(odenen_siparisler)
        toplam_gelir = 0.0
        satilan_urun_adedi = 0
        nakit_gelir = 0.0
        kredi_karti_gelir = 0.0
        diger_odeme_yontemleri_gelir = 0.0

        for siparis in odenen_siparisler:
            try:
                sepet_items = json.loads(siparis["sepet"] or "[]")
                siparis_tutari_bu_iterasyonda = 0
                for item in sepet_items:
                    adet = item.get("adet", 0)
                    fiyat = item.get("fiyat", 0.0)
                    siparis_tutari_bu_iterasyonda += adet * fiyat
                    satilan_urun_adedi += adet

                toplam_gelir += siparis_tutari_bu_iterasyonda
                odeme_yontemi_str = str(siparis["odeme_yontemi"]).lower() if siparis["odeme_yontemi"] else "bilinmiyor"

                if "nakit" in odeme_yontemi_str:
                    nakit_gelir += siparis_tutari_bu_iterasyonda
                elif "kredi kartı" in odeme_yontemi_str or "kart" in odeme_yontemi_str or "credit card" in odeme_yontemi_str:
                    kredi_karti_gelir += siparis_tutari_bu_iterasyonda
                else:
                    diger_odeme_yontemleri_gelir += siparis_tutari_bu_iterasyonda

            except json.JSONDecodeError: # pragma: no cover
                logger.warning(f"Günlük istatistik: Sepet parse hatası, Sipariş durumu: {siparis['durum']}, Sepet: {siparis['sepet']}")
                continue

        return GunlukIstatistik(
            tarih=gun_baslangic_dt.strftime("%Y-%m-%d"),
            siparis_sayisi=siparis_sayisi,
            toplam_gelir=round(toplam_gelir, 2),
            satilan_urun_adedi=satilan_urun_adedi,
            nakit_gelir=round(nakit_gelir, 2),
            kredi_karti_gelir=round(kredi_karti_gelir, 2),
            diger_odeme_yontemleri_gelir=round(diger_odeme_yontemleri_gelir, 2)
        )
    except Exception as e: # pragma: no cover
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
        if not (1 <= target_ay <= 12): # pragma: no cover
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Geçersiz ay değeri. 1-12 arasında olmalıdır.")
        ay_baslangic_dt = datetime(target_yil, target_ay, 1, tzinfo=TR_TZ)
        if target_ay == 12: # pragma: no cover
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
            except json.JSONDecodeError: # pragma: no cover
                logger.warning(f"Aylık istatistik: Sepet parse hatası, Sipariş durumu: {siparis['durum']}, Sepet: {siparis['sepet']}")
                continue
        return AylikIstatistik(
            ay=ay_baslangic_dt.strftime("%Y-%m"),
            siparis_sayisi=siparis_sayisi,
            toplam_gelir=round(toplam_gelir, 2),
            satilan_urun_adedi=satilan_urun_adedi
        )
    except Exception as e: # pragma: no cover
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
            if siparis_zamani.tzinfo is None: # pragma: no cover
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
            except json.JSONDecodeError: # pragma: no cover
                logger.warning(f"Yıllık kırılım: Sepet parse hatası, Sipariş zamanı: {siparis['zaman']}, Sepet: {siparis['sepet']}")
                continue
        response_data = {
            key: YillikAylikKirilimDetay(**value)
            for key, value in aylik_kirilim_data.items()
        }
        return YillikAylikKirilimResponse(aylik_kirilim=response_data)
    except Exception as e: # pragma: no cover
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
            except json.JSONDecodeError: # pragma: no cover
                logger.warning(f"En çok satılan: Sepet parse hatası, Sepet: {siparis['sepet']}")
                continue
        en_cok_satilanlar = [
            EnCokSatilanUrun(urun=item[0], adet=item[1])
            for item in urun_sayaclari.most_common(limit)
        ]
        return en_cok_satilanlar
    except Exception as e: # pragma: no cover
        logger.error(f"❌ En çok satılan ürünler alınırken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="En çok satılan ürünler alınırken bir sorun oluştu.")

@app.get("/admin/aktif-masa-tutarlari", response_model=List[AktifMasaOzet], tags=["Admin İşlemleri"])
async def get_aktif_masa_tutarlari(
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"Admin '{current_user.kullanici_adi}' aktif masa tutarlarını istedi.")
    try:
        odenmemis_durumlar = [Durum.BEKLIYOR.value, Durum.HAZIRLANIYOR.value, Durum.HAZIR.value]
        query_str = "SELECT masa, sepet FROM siparisler WHERE durum = ANY(:statuses_list)" # PostgreSQL ANY kullanımı
        values = {"statuses_list": odenmemis_durumlar} # Liste olarak gönder
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
            except json.JSONDecodeError: # pragma: no cover
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
    except Exception as e: # pragma: no cover
        logger.error(f"❌ Aktif masa tutarları alınırken hata: {e}", exc_info=True)
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
            if not order_info: # pragma: no cover
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sipariş bulunamadı.")
            siparis_masasi = order_info["masa"]
            updated_raw = await db.fetch_one(
                "UPDATE siparisler SET durum = :durum WHERE id = :id RETURNING id, masa, durum, sepet, istek, zaman, odeme_yontemi",
                {"durum": data.durum.value, "id": id}
            )
        if not updated_raw: # pragma: no cover
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sipariş bulunamadı veya güncellenemedi.")
        updated_order = dict(updated_raw)
        try:
            updated_order["sepet"] = json.loads(updated_order.get("sepet", "[]"))
        except json.JSONDecodeError: # pragma: no cover
            updated_order["sepet"] = []
            logger.warning(f"Sipariş {id} sepet JSON parse hatası (patch_order_endpoint).")
        if isinstance(updated_order.get('zaman'), datetime):
             updated_order['zaman'] = updated_order['zaman'].isoformat()
        notif_data = {**updated_order, "zaman": datetime.now(TR_TZ).isoformat()} # Zamanı güncelle
        notification = {"type": "durum", "data": notif_data}
        await broadcast_message(aktif_mutfak_websocketleri, notification, "Mutfak/Masa")
        await broadcast_message(aktif_admin_websocketleri, notification, "Admin")
        await broadcast_message(aktif_kasa_websocketleri, notification, "Kasa")
        await update_table_status(siparis_masasi, f"Sipariş {id} durumu güncellendi -> {updated_order['durum']} (by {current_user.kullanici_adi})")
        return {"message": f"Sipariş {id} güncellendi.", "data": updated_order}
    except HTTPException: # pragma: no cover
        raise
    except Exception as e: # pragma: no cover
        logger.error(f"❌ PATCH /siparis/{id} hatası: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Sipariş durumu güncellenirken hata oluştu.")

@app.delete("/siparis/{id}", tags=["Siparişler"])
async def delete_order_by_admin_endpoint(
    id: int = Path(..., description="İptal edilecek (silinecek) siparişin ID'si"),
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"🗑️ ADMIN DELETE (as cancel) /siparis/{id} ile iptal isteği (Kullanıcı: {current_user.kullanici_adi})")
    row = await db.fetch_one("SELECT zaman, masa, durum, odeme_yontemi FROM siparisler WHERE id = :id", {"id": id})
    if not row: # pragma: no cover
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sipariş bulunamadı.")
    if row["durum"] == Durum.IPTAL.value: # pragma: no cover
        return {"message": f"Sipariş {id} zaten iptal edilmiş."} # 200 OK döner
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
    except Exception as e: # pragma: no cover
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
    if not order_details: # pragma: no cover
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="İptal edilecek sipariş bulunamadı veya bu masaya ait değil.")
    if order_details["durum"] == "iptal": # pragma: no cover
        return {"message": "Bu sipariş zaten iptal edilmiş."}
    if order_details["durum"] not in [Durum.BEKLIYOR.value, Durum.HAZIRLANIYOR.value]: # pragma: no cover
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Siparişinizin durumu ({order_details['durum']}) iptal işlemi için uygun değil.")

    olusturma_zamani = order_details["zaman"]
    if isinstance(olusturma_zamani, str): # pragma: no cover
        try:
            olusturma_zamani_dt = datetime.fromisoformat(olusturma_zamani)
        except ValueError:
            olusturma_zamani_dt = datetime.strptime(olusturma_zamani, "%Y-%m-%d %H:%M:%S").replace(tzinfo=TR_TZ) # Fallback for older format
    else: # datetime object
        olusturma_zamani_dt = olusturma_zamani

    if olusturma_zamani_dt.tzinfo is None: # pragma: no cover
        olusturma_zamani_dt = olusturma_zamani_dt.replace(tzinfo=TR_TZ) # Assume TR_TZ if naive
    else:
        olusturma_zamani_dt = olusturma_zamani_dt.astimezone(TR_TZ) # Convert to TR_TZ if aware but different

    if datetime.now(TR_TZ) - olusturma_zamani_dt > timedelta(minutes=2): # pragma: no cover
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
    except Exception as e: # pragma: no cover
        logger.error(f"❌ Müşteri sipariş iptali sırasında (Sipariş ID: {siparis_id}, Masa: {masa_no}) hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Siparişiniz iptal edilirken bir sunucu hatası oluştu.")

@alru_cache(maxsize=1)
async def get_menu_price_dict() -> Dict[str, float]:
    logger.info(">>> get_menu_price_dict ÇAĞRILIYOR...")
    try:
        if not menu_db.is_connected: await menu_db.connect() # pragma: no cover
        prices_raw = await menu_db.fetch_all("SELECT ad, fiyat FROM menu")
        price_dict = {row['ad'].lower().strip(): float(row['fiyat']) for row in prices_raw}
        logger.info(f"Fiyat sözlüğü {len(price_dict)} ürün için oluşturuldu/alındı.")
        return price_dict
    except Exception as e: # pragma: no cover
        logger.error(f"❌ Fiyat sözlüğü oluşturma/alma hatası: {e}", exc_info=True)
        return {}

@alru_cache(maxsize=1)
async def get_menu_stock_dict() -> Dict[str, int]:
    logger.info(">>> get_menu_stock_dict ÇAĞRILIYOR...")
    try:
        if not menu_db.is_connected: await menu_db.connect() # pragma: no cover
        stocks_raw = await menu_db.fetch_all("SELECT ad, stok_durumu FROM menu")
        if not stocks_raw: return {} # pragma: no cover
        stock_dict = {}
        for row in stocks_raw:
            try: stock_dict[str(row['ad']).lower().strip()] = int(row['stok_durumu'])
            except Exception as e_loop: logger.error(f"Stok sözlüğü oluştururken satır işleme hatası: {e_loop}", exc_info=True) # pragma: no cover
        logger.info(f">>> get_menu_stock_dict: Oluşturulan stock_dict ({len(stock_dict)} öğe).")
        return stock_dict
    except Exception as e_main: # pragma: no cover
        logger.error(f"❌ Stok sözlüğü oluşturma/alma sırasında genel hata: {e_main}", exc_info=True)
        return {}

@alru_cache(maxsize=1)
async def get_menu_for_prompt_cached() -> str:
    logger.info(">>> GET_MENU_FOR_PROMPT_CACHED ÇAĞRILIYOR (Fiyatlar Dahil Edilecek)...")
    try:
        if not menu_db.is_connected: # pragma: no cover
            await menu_db.connect()
        query = """
            SELECT k.isim as kategori_isim, m.ad as urun_ad, m.fiyat as urun_fiyat
            FROM menu m
            JOIN kategoriler k ON m.kategori_id = k.id
            WHERE m.stok_durumu = 1
            ORDER BY k.isim, m.ad
        """
        urunler_raw = await menu_db.fetch_all(query)
        if not urunler_raw: # pragma: no cover
            return "Üzgünüz, şu anda menümüzde aktif ürün bulunmamaktadır."
        kategorili_menu: Dict[str, List[str]] = {}
        for row in urunler_raw:
            try:
                urun_adi = row['urun_ad']
                urun_fiyati_str = f"{float(row['urun_fiyat']):.2f} TL"
                kategori_ismi = row['kategori_isim']
                kategorili_menu.setdefault(kategori_ismi, []).append(f"{urun_adi} ({urun_fiyati_str})")
            except Exception as e_row: # pragma: no cover
                logger.error(f"get_menu_for_prompt_cached (fiyatlı): Satır işlenirken hata: {e_row} - Satır: {row}", exc_info=True)
        if not kategorili_menu: # pragma: no cover
            return "Üzgünüz, menü bilgisi şu anda düzgün bir şekilde formatlanamıyor."
        menu_aciklama_list = [
            f"- {kategori}: {', '.join(urun_listesi_detayli)}"
            for kategori, urun_listesi_detayli in kategorili_menu.items() if urun_listesi_detayli
        ]
        if not menu_aciklama_list: # pragma: no cover
            return "Üzgünüz, menüde listelenecek ürün bulunamadı."
        logger.info(f"Menü (fiyatlar dahil) prompt için başarıyla oluşturuldu ({len(kategorili_menu)} kategori).")
        return "\n".join(menu_aciklama_list)
    except Exception as e: # pragma: no cover
        logger.error(f"❌ Menü (fiyatlar dahil) prompt oluşturma hatası: {e}", exc_info=True)
        return "Teknik bir sorun nedeniyle menü bilgisine ve fiyatlara şu anda ulaşılamıyor. Lütfen daha sonra tekrar deneyin veya personelden yardım isteyin."

SISTEM_MESAJI_ICERIK_TEMPLATE = (
    "Sen Fıstık Kafe için **Neso** adında, son derece zeki, neşeli, konuşkan, müşteriyle empati kurabilen, hafif esprili ve satış yapmayı seven ama asla bunaltmayan bir sipariş asistanısın. "
    "Görevin, müşterilerin taleplerini doğru anlamak, onlara Fıstık Kafe'nin MENÜSÜNDEKİ lezzetleri coşkuyla tanıtmak ve **SADECE VE SADECE** sana aşağıda '# GÜNCEL STOKTAKİ ÜRÜNLER, FİYATLARI VE KATEGORİLERİ' başlığı altında verilen güncel MENÜ LİSTESİNDEKİ ürünleri (isimleri, fiyatları, kategorileri ve varsa özellikleriyle) kullanarak siparişlerini JSON formatında hazırlamaktır. Bu MENÜ LİSTESİ dışındaki hiçbir ürünü önerme, kabul etme, hakkında yorum yapma veya varmış gibi davranma. **KAFEDE KESİNLİKLE ANA YEMEK (pizza, kebap, dürüm vb.) SERVİSİ BULUNMAMAKTADIR.** Amacın, Fıstık Kafe deneyimini bu sana verilen MENÜ çerçevesinde unutulmaz kılmaktır.\n\n"

    "# TEMEL ÇALIŞMA PRENSİBİ VE BAĞLAM YÖNETİMİ\n"
    "1.  **Bağlam Bilgisi (`previous_context_summary`):** Sana bir önceki AI etkileşiminin JSON özeti (`previous_context_summary`) verilebilir. Bu özet, bir önceki AI yanıtındaki `sepet`, `toplam_tutar`, `konusma_metni` ve `onerilen_urun` gibi bilgileri içerir. Kullanıcının yeni mesajını **HER ZAMAN BU ÖZETİ DİKKATE ALARAK** yorumla. Bu, konuşmanın doğal akışını ve tutarlılığını sağlamak için KRİTİKTİR.\n"
    "    * **Önceki Öneriyi Kabul/Red:** Eğer `previous_context_summary` içinde bir `onerilen_urun` varsa (bu öneri fiyatını da içerir) ve kullanıcı 'evet', 'olsun', 'tamamdır' gibi bir onay veriyorsa, o ürünü (1 adet) MENÜDEKİ doğru fiyat ve kategoriyle JSON sepetine ekle. Eğer 'hayır', 'istemiyorum' gibi bir red cevabı verirse, kibarca başka bir şey isteyip istemediğini sor (DÜZ METİN).\n"
    "    * **Önceki Sepete Referans:** Eğer `previous_context_summary` içinde bir `sepet` varsa ve kullanıcı 'ondan bir tane daha', 'şunu çıkar', 'bir de [başka ürün]' gibi mevcut sepete atıfta bulunan bir ifade kullanıyorsa, `previous_context_summary`'deki `sepet` ve `konusma_metni`'ni kullanarak hangi üründen bahsettiğini ANLAMAYA ÇALIŞ. Eğer netse, `previous_context_summary`'deki sepeti güncelleyerek YENİ JSON oluştur. Net değilse, DÜZ METİN ile hangi ürünü kastettiğini sor (örn: 'Tabii, hangi üründen bir tane daha ekleyelim? Masanızdaki siparişte X ve Y var.').\n"
    "    * **Önceki Soruya Cevap:** Eğer `previous_context_summary`'deki `konusma_metni` bir soru içeriyorsa (örn: 'Türk Kahveniz şekerli mi olsun, şekersiz mi?'), kullanıcının yeni mesajını bu soruya bir cevap olarak değerlendir ve gerekiyorsa `musteri_notu`'na işle.\n"
    "    * **YENİ KURAL (JSON ZORUNLULUĞU):** Eğer kullanıcı MENÜDEN net bir sipariş verirse (örn: '2 limonata', '1 Türk Kahvesi') veya `previous_context_summary` içindeki bir öneriyi kabul ederek adet belirtirse (örn: önerilen 'Limonata' için '2 tane alayım'), **BU DURUMDA ASLA DÜZ METİN DÖNME**. Doğrudan aşağıdaki '# JSON YANIT FORMATI'na uygun bir JSON yanıtı ver ve `aksiyon_durumu` alanını `\"siparis_guncellendi\"` olarak ayarla. Örneğin, kullanıcı '2 limonata' derse ve Limonata menüde (diyelim ki) 25.00 TL ise, beklenen JSON (fiyat ve toplam_tutar sayısal (float) olmalı):\n"
    "      ```json\n"
    "      {{\n"
    "        \"sepet\": [\n"
    "          {{\n"
    "            \"urun\": \"Limonata\",\n"
    "            \"adet\": 2,\n"
    "            \"fiyat\": 25.00,      // SAYI (FLOAT) OLARAK, MENÜDEN ALINACAK\n"
    "            \"kategori\": \"Soğuk İçecekler\",\n"
    "            \"musteri_notu\": \"\"\n"
    "          }}\n"
    "        ],\n"
    "        \"toplam_tutar\": 50.00,  // SAYI (FLOAT) OLARAK, HESAPLANACAK\n"
    "        \"musteri_notu\": \"\",\n"
    "        \"konusma_metni\": \"Harika tercih! 2 adet Limonata masanıza eklendi. Masanızın güncel tutarı 50.00 TL oldu. Başka bir lezzet de eklemek ister misiniz? 🍋\",\n"
    "        \"onerilen_urun\": null,\n"
    "        \"aksiyon_durumu\": \"siparis_guncellendi\"\n"
    "      }}\n"
    "      ```\n"
    "2.  **Yanıt Formatı:** Amacın, kullanıcıdan sana verilen MENÜYE göre net bir sipariş almak veya MENÜ hakkında sorularını coşkulu bir şekilde yanıtlamaktır. Yanıtlarını HER ZAMAN aşağıdaki '# JSON YANIT FORMATI' veya '# DÜZ METİN YANIT KURALLARI'na göre ver.\n\n"

    "# GÜNCEL STOKTAKİ ÜRÜNLER, FİYATLARI VE KATEGORİLERİ (TEK GEÇERLİ KAYNAK BUDUR!)\n"
    "Fıstık Kafe sadece içecek ve hafif atıştırmalıklar sunar. İşte tam liste:\n"
    "{menu_prompt_data}\n"
    "# KESİN KURAL (MENÜ SADAKATİ):\n"
    "1.  Yukarıdaki MENÜ güncel ve doğrudur. İşleyebileceğin TÜM ürünler, kategoriler, fiyatlar ve özellikler (varsa) BU LİSTEYLE SINIRLIDIR.\n"
    "2.  Ürün isimlerini, fiyatlarını (SAYI olarak) ve kategorilerini JSON'a yazarken **TAM OLARAK BU LİSTEDE GÖRDÜĞÜN GİBİ KULLAN**.\n"
    "3.  Bu listede olmayan hiçbir şeyi siparişe ekleme, önerme, hakkında yorum yapma veya varmış gibi davranma.\n"
    "4.  Kullanıcı bu listede olmayan bir şey sorarsa, '# ÖNEMLİ KURALLAR' bölümündeki 'Menü Dışı Talepler' kuralına göre yanıt ver.\n"
    "5.  **ASLA MENÜ DIŞI BİR ÜRÜN UYDURMA, VARSAYIM YAPMA VEYA MENÜDEKİ BİR ÜRÜNÜ İSTENEN FARKLI BİR ÜRÜN YERİNE KOYMA.** HER ZAMAN KULLANICIDAN NET BİLGİ AL.\n"
    "6.  **BİR ÜRÜN İÇİN ALTERNATİF SUNARKEN DAHİ, SUNACAĞIN ALTERNATİFLER MUTLAKA YUKARIDAKİ LİSTEDE BULUNAN ÜRÜNLER OLMALIDIR. BU LİSTEDE OLMAYAN HİÇBİR ŞEYİ ALTERNATİF OLARAK DAHİ ÖNERME.**\n\n"

    "# JSON YANIT FORMATI (SADECE SİPARİŞ ALINDIĞINDA VEYA MEVCUT SİPARİŞ GÜNCELLENDİĞİNDE KULLANILACAK!)\n"
    "**KURAL: Aşağıdaki durumlar GERÇEKLEŞTİĞİNDE, yanıtın SADECE ve KESİNLİKLE bu JSON formatında olmalıdır. Başka hiçbir metin ekleme:**\n"
    "1. Kullanıcı MENÜDEN net bir ürün ve adet belirtirse (örn: '2 limonata', 'bir Türk kahvesi').\n"
    "2. Kullanıcı MENÜDEN bir önceki AI önerisini kabul ederse.\n"
    "3. Kullanıcı mevcut sepetine MENÜDEN ürün ekler, çıkarır veya adedini değiştirirse.\n"
    "4. Kullanıcı bir ürün için varyasyon belirtirse (örn: 'şekerli olsun').\n"
    "Eğer bu durumlardan biri geçerliyse, aşağıdaki JSON formatını KULLAN:\n"
    "{{\n"
    "  \"sepet\": [\n"
    "    {{\n"
    "      \"urun\": \"MENÜDEKİ TAM ÜRÜN ADI. Listede olmayan bir ürünü ASLA buraya yazma.\",\n"
    "      \"adet\": ADET_SAYISI (integer, pozitif olmalı),\n"
    "      \"fiyat\": MENUDEKI_URUNUN_BIRIM_FIYATI (float), // ÖNEMLİ: Burası SAYI (float) olmalı, string değil! MENÜDEN ALINACAK.\n"
    "      \"kategori\": \"MENÜDEKİ DOĞRU KATEGORİ_ADI.\",\n"
    "      \"musteri_notu\": \"Müşterinin BU ÜRÜN İÇİN özel isteği (örn: 'az şekerli', 'bol buzlu', 'yanında limonla') veya ürün varyasyonu (örn: 'orta şekerli') veya boş string ('').\"\n"
    "    }}\n"
    "    // Sepette birden fazla ürün olabilir...\n"
    "  ],\n"
    "  \"toplam_tutar\": SEPETTEKI_TUM_URUNLERIN_HESAPLANMIS_TOPLAM_TUTARI (float), // ÖNEMLİ: Burası SAYI (float) olmalı, string değil! (adet * birim_fiyat)ların toplamı.\n"
    "  \"musteri_notu\": \"SİPARİŞİN GENELİ İÇİN müşteri notu (örn: 'hepsi ayrı paketlensin', 'doğum günü için') veya boş string ('').\",\n"
    "  \"konusma_metni\": \"Müşteriye söylenecek, durumu özetleyen, Neso'nun enerjik ve samimi karakterine uygun bir metin. Örn: 'Harika! [Ürün Adı] masanıza eklendi. Masanızın güncel tutarı [Toplam Tutar] TL oldu. Başka bir arzunuz var mı?'\",\n"
    "  \"onerilen_urun\": \"Eğer bu etkileşimde MENÜDEN bir ürün öneriyorsan VE kullanıcı henüz bu öneriyi kabul etmediyse, önerdiğin ürünün TAM ADINI ve MENÜDEKİ BİRİM FİYATINI buraya yaz (örn: 'Fıstık Rüyası (75.00 TL)'). Aksi halde null bırak.\",\n"
    "  \"aksiyon_durumu\": \"'siparis_guncellendi'\" // BU ALAN HER ZAMAN BU ŞEKİLDE OLMALI EĞER JSON YANITI VERİYORSAN!\n"
    "}}\n\n"

    "# DÜZ METİN YANIT KURALLARI (JSON YERİNE KULLANILACAK DURUMLAR)\n"
    "AŞAĞIDAKİ durumlardan biri geçerliyse, YUKARIDAKİ JSON FORMATINI KULLANMA. SADECE müşteriye söylenecek `konusma_metni`'ni Neso karakterine uygun, doğal, canlı ve samimi bir dille düz metin olarak yanıtla:\n"
    "1.  **İlk Karşılama ve Genel Selamlamalar:** Müşteri sohbete yeni başladığında ('merhaba', 'selam').\n"
    "    Örnek: \"Merhaba! Ben Neso, Fıstık Kafe'nin neşe dolu asistanı! Bugün sizi burada görmek harika! Menümüzden size hangi lezzetleri önermemi istersiniz? 😉\"\n"
    "2.  **Genel MENÜ Soruları veya Fiyat Sorma:** Müşteri MENÜ, MENÜDEKİ ürünler hakkında genel bir soru soruyorsa (örn: 'MENÜDE hangi Pastalar var?', 'Sıcak İçecekleriniz nelerdir?', 'Fıstık Rüyası nasıl bir tatlı?', 'Türk Kahvesi ne kadar?'). Cevabında MENÜDEKİ ürünleri, İSTENİRSE fiyatlarını ve (varsa) özelliklerini kullan. Fiyatı sadece müşteri sorarsa veya sen bir ürün önerirken belirt.\n"
    "3.  **MENÜDEN Öneri İstekleri (Henüz Ürün Seçilmemişse):** Müşteri bir öneri istiyorsa ama HENÜZ bir ürün seçmemişse. Bu durumda SADECE MENÜDEKİ ürünlerin özelliklerini kullanarak coşkulu bir şekilde 1-2 ürün öner. Önerini yaparken ürünün TAM ADINI ve FİYATINI da belirt. Örnek: \"Soğuk ve sütsüz bir içecek mi arıyorsunuz? Size ferahlatıcı Limonata (XX.XX TL) veya serinletici Kola (YY.YY TL) öneririm! Hangisini denemek istersiniz? 🥤\"\n"
    "4.  **Belirsiz veya Eksik Bilgiyi MENÜDEN Netleştirme İhtiyacı:** Müşterinin isteği belirsizse (örn: 'bir kahve') ve MENÜDEN netleştirme gerekiyorsa (örn: 'Menümüzde Türk Kahvesi ve Filtre Kahve mevcut, hangisini tercih edersiniz? Fiyatlarını öğrenmek ister misiniz?').\n"
    "5.  **Menü Dışı Talepler veya Anlaşılamayan İstekler:** '# ÖNEMLİ KURALLAR' bölümündeki 'Menü Dışı Talepler' kuralına göre yanıt ver.\n"
    "6.  **Sipariş Dışı Kısa Sohbetler:** Konuyu nazikçe MENÜYE veya siparişe getir.\n\n"

    "# ÖNEMLİ KURALLAR (HER ZAMAN UYULACAK!)\n\n"
    "## 1. Menü Dışı Talepler ve Anlamsız Sorular:\n"
    "   - Müşteri SANA VERİLEN MENÜDE olmayan bir ürün (özellikle kebap, pizza gibi ana yemekler VEYA menüde olmayan spesifik bir içecek/tatlı çeşidi) veya konuyla tamamen alakasız, anlamsız bir soru sorarsa, ürünün/konunun MENÜDE olmadığını veya yardımcı olamayacağını KISA, NET ve KİBARCA Neso üslubuyla belirt. ASLA o ürün hakkında yorum yapma, VARSAYIMDA BULUNARAK BENZER BİR ÜRÜN EKLEME veya varmış gibi davranma. Sonrasında HEMEN konuyu Fıstık Kafe'nin MENÜSÜNE veya sipariş işlemine geri getirerek **SADECE MENÜDE BULUNAN ÜRÜNLERDEN** bir alternatif öner. DÜZ METİN yanıt ver.\n"
    "     Örnek Yanıt (Kullanıcı 'Papatya çayı var mı?' derse ve menüde Adaçayı ve Kuşburnu varsa): 'Papatya çayımız maalesef şu anda menümüzde bulunmuyor. Ama dilerseniz menümüzdeki diğer harika bitki çaylarımızdan Adaçayı (fiyatını isterseniz söyleyebilirim) veya Kuşburnu Çayı (fiyatını isterseniz söyleyebilirim) deneyebilirsiniz. Hangisini istersiniz? Ya da farklı bir sıcak içecek mi düşünmüştünüz? 🍵'\n\n"
    "## 2. Ürün Varyasyonları ve Özel İstekler:\n"
    "   - Bazı ürünler için müşteriye seçenek sunman gerekebilir (örn: Türk Kahvesi için 'şekerli mi, orta mı, şekersiz mi?'). Bu durumda DÜZ METİN ile soruyu sor. Müşteri yanıtladığında, bu bilgiyi ilgili ürünün JSON içindeki `musteri_notu` alanına işle ve JSON yanıtı ile siparişi güncelle.\n"
    "   - Müşteri kendiliğinden 'az şekerli olsun', 'yanında limonla' gibi bir istekte bulunursa, bunu da ilgili ürünün JSON `musteri_notu`'na ekle ve JSON yanıtı ile siparişi güncelle.\n\n"
    "## 3. Fiyat, Kategori ve Ürün Özellikleri Bilgisi:\n"
    "   - Sepete eklediğin veya hakkında bilgi verdiğin her ürün için isim, fiyat ve kategori bilgisini **KESİNLİKLE VE SADECE** yukarıdaki **'# GÜNCEL STOKTAKİ ÜRÜNLER, FİYATLARI VE KATEGORİLERİ'** listesinden al. Fiyatları (SAYI olarak) ve kategorileri ASLA TAHMİN ETME. Toplam tutarı hesaplarken birim fiyatları bu listeden al. **Birim fiyatları müşteriye sadece sorulduğunda veya bir ürün önerirken belirt.**\n\n"
    "## 4. Ürün Adı Eşleştirme ve Netleştirme:\n"
    "   - Kullanıcı tam ürün adını söylemese bile (örn: 'sahlepli bir şey', 'fıstıklı olan tatlıdan'), yalnızca SANA VERİLEN MENÜ LİSTESİNDEKİ ürün adları, kategorileri ve (varsa) açıklamalarıyla %100'e yakın ve KESİN bir eşleşme bulabiliyorsan bu ürünü dikkate al.\n"
    "   - **ÇOK ÖNEMLİ:** Eğer kullanıcı menüde olmayan bir ürün isterse (örn: 'papatya çayı') VE aynı zamanda menüde olan bir ürün de isterse (örn: '2 sahlep ve papatya çayı'), aşağıdaki adımları izle:\n"
    "     a. Önce DÜZ METİN bir yanıt ver. Bu yanıtta, menüde olmayan ürün için ('Papatya çayımız şu anda menümüzde bulunmuyor maalesef.') bilgi ver.\n"
    "     b. Ardından, **SADECE MENÜDEKİ** mevcut benzer kategoriden (eğer varsa ve ürün içeriyorsa) alternatifler sun (örn: 'Ancak dilerseniz menümüzdeki diğer bitki çaylarımızdan Adaçayı veya Ihlamur deneyebilirsiniz. Fiyatlarını isterseniz belirtebilirim.' -> BU ÖRNEKTE ADAÇAYI VE IHLAMURUN MENÜDE OLDUĞU VARSAYILMIŞTIR. EĞER MENÜDE BU ÜRÜNLER YOKSA, ONLARI ÖNERME! Sadece menüde olanları öner.).\n"
    "     c. Aynı zamanda, menüde olan ve istenen diğer ürünleri (örn: '2 adet Sahlep') siparişinize ekleyebileceğini belirt.\n"
    "     d. Kullanıcıya ne yapmak istediğini sor (örn: 'Sahleplerinizi masanıza ekleyelim mi? Yanında menümüzdeki başka bir çayımızdan denemek ister misiniz?').\n"
    "     e. Kullanıcıdan net bir onay (örn: 'Evet, sahlepleri alayım, adaçayı da olsun') aldıktan SONRA ilgili ürünleri içeren JSON sepetini oluştur.\n"
    "   - Eğer eşleşmeden %100 emin değilsen veya kullanıcının isteği MENÜDEKİ birden fazla ürüne benziyorsa, ASLA varsayım yapma. Bunun yerine, DÜZ METİN ile soru sorarak MENÜDEN netleştir ve kullanıcıya MENÜDEKİ seçenekleri (isimlerini) hatırlat (örn: 'Fıstıklı olarak menümüzde Fıstık Rüyası ve Fıstıklı Dondurma mevcut, hangisini arzu edersiniz? Fiyatlarını isterseniz söyleyebilirim.').\n\n"
    "## 5. `aksiyon_durumu` JSON Alanının Kullanımı:\n"
    "   - Eğer bir JSON yanıtı üretiyorsan (yani bir sipariş alınıyor veya güncelleniyorsa), JSON objesinin İÇİNDE **MUTLAKA** `\"aksiyon_durumu\": \"siparis_guncellendi\"` satırı bulunmalıdır. Bu, sistemin siparişi kaydetmesi için gereklidir. Başka bir değer KULLANMA.\n"
    "   - DÜZ METİN yanıt verdiğin durumlarda (bilgi verme, soru sorma, hata yönetimi) JSON dönmediğin için bu alan kullanılmaz.\n\n"

    "### TEMEL PRENSİP: MENÜYE TAM BAĞLILIK!\n"
    "HER NE KOŞULDA OLURSA OLSUN, tüm işlemlerin SADECE '# GÜNCEL STOKTAKİ ÜRÜNLER, FİYATLARI VE KATEGORİLERİ' bölümünde sana sunulan ürünlerle sınırlıdır. Bu listenin dışına çıkmak, menüde olmayan bir üründen bahsetmek veya varmış gibi davranmak KESİNLİKLE YASAKTIR. Müşteriyi HER ZAMAN menüdeki seçeneklere yönlendir.\n\n"
    "Neso olarak görevin, Fıstık Kafe müşterilerine keyifli, enerjik ve lezzet dolu bir deneyim sunarken, SADECE MENÜDEKİ ürünlerle doğru ve eksiksiz siparişler almak ve gerektiğinde MENÜ hakkında doğru bilgi vermektir. Şimdi bu KESİN KURALLARA ve yukarıdaki MENÜYE göre kullanıcının talebini işle ve uygun JSON veya DÜZ METİN çıktısını üret!"
)

SYSTEM_PROMPT: Optional[Dict[str, str]] = None

async def update_system_prompt():
    global SYSTEM_PROMPT
    logger.info("🔄 Sistem mesajı (menü bilgisi) güncelleniyor...")
    menu_data_for_prompt = "Menü bilgisi geçici olarak yüklenemedi." # Fallback
    try:
        # Cache'leri temizle
        if hasattr(get_menu_for_prompt_cached, 'cache_clear'): get_menu_for_prompt_cached.cache_clear()
        if hasattr(get_menu_price_dict, 'cache_clear'): get_menu_price_dict.cache_clear()
        if hasattr(get_menu_stock_dict, 'cache_clear'): get_menu_stock_dict.cache_clear()
        logger.info("İlgili menü cache'leri temizlendi (update_system_prompt).")

        menu_data_for_prompt = await get_menu_for_prompt_cached()
        current_system_content = SISTEM_MESAJI_ICERIK_TEMPLATE.format(menu_prompt_data=menu_data_for_prompt)
        SYSTEM_PROMPT = {"role": "system", "content": current_system_content}
        logger.info(f"✅ Sistem mesajı başarıyla güncellendi.")
    except KeyError as ke: # pragma: no cover
        logger.error(f"❌ Sistem mesajı güncellenirken KeyError oluştu: {ke}. Şablonda eksik/yanlış anahtar olabilir.", exc_info=True)
        try:
            current_system_content_fallback = SISTEM_MESAJI_ICERIK_TEMPLATE.replace("{menu_prompt_data}", "Menü bilgisi yüklenirken hata oluştu (fallback).")
            SYSTEM_PROMPT = {"role": "system", "content": current_system_content_fallback}
            logger.warning(f"Fallback sistem mesajı (KeyError sonrası) kullanılıyor.")
        except Exception as fallback_e:
            logger.error(f"❌ Fallback sistem mesajı oluşturulurken de hata oluştu: {fallback_e}", exc_info=True)
            SYSTEM_PROMPT = {"role": "system", "content": "Ben Neso, Fıstık Kafe sipariş asistanıyım. Size nasıl yardımcı olabilirim? (Sistem mesajı yüklenemedi.)"}
    except Exception as e: # pragma: no cover
        logger.error(f"❌ Sistem mesajı güncellenirken BEKLENMEDİK BİR HATA oluştu: {e}", exc_info=True)
        if SYSTEM_PROMPT is None: # Eğer ilk başlatmada hata olursa ve SYSTEM_PROMPT hiç set edilmemişse
            try:
                current_system_content_fallback = SISTEM_MESAJI_ICERIK_TEMPLATE.replace("{menu_prompt_data}", "Menü bilgisi yüklenirken genel hata oluştu (fallback).")
                SYSTEM_PROMPT = {"role": "system", "content": current_system_content_fallback}
                logger.warning(f"Fallback sistem mesajı (BEKLENMEDİK HATA sonrası) kullanılıyor.")
            except Exception as fallback_e:
                 logger.error(f"❌ Fallback sistem mesajı oluşturulurken de (genel hata sonrası) hata oluştu: {fallback_e}", exc_info=True)
                 SYSTEM_PROMPT = {"role": "system", "content": "Ben Neso, Fıstık Kafe sipariş asistanıyım. Size nasıl yardımcı olabilirim? (Sistem mesajı yüklenemedi.)"}

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
            await db.execute("CREATE INDEX IF NOT EXISTS idx_siparisler_masa_zaman ON siparisler(masa, zaman DESC)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_siparisler_durum ON siparisler(durum)")

            await db.execute("""
                CREATE TABLE IF NOT EXISTS masa_durumlar (
                    id SERIAL PRIMARY KEY,
                    masa_id TEXT UNIQUE NOT NULL,
                    son_erisim TIMESTAMP WITH TIME ZONE NOT NULL,
                    aktif BOOLEAN DEFAULT TRUE,
                    son_islem TEXT
                )""")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_masa_durumlar_erisim ON masa_durumlar(son_erisim DESC)")

            await db.execute("""
                CREATE TABLE IF NOT EXISTS kullanicilar (
                    id SERIAL PRIMARY KEY,
                    kullanici_adi TEXT UNIQUE NOT NULL,
                    sifre_hash TEXT NOT NULL,
                    rol TEXT NOT NULL CHECK(rol IN ('admin', 'kasiyer', 'barista', 'mutfak_personeli')),
                    aktif_mi BOOLEAN DEFAULT TRUE,
                    olusturulma_tarihi TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )""")
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
    except Exception as e: # pragma: no cover
        logger.critical(f"❌ Ana veritabanı tabloları başlatılırken kritik hata: {e}", exc_info=True)
        raise

async def init_menu_db():
    logger.info(f"Menü veritabanı tabloları kontrol ediliyor/oluşturuluyor...")
    try:
        async with menu_db.transaction():
            await menu_db.execute("""
                CREATE TABLE IF NOT EXISTS kategoriler (
                    id SERIAL PRIMARY KEY,
                    isim TEXT UNIQUE NOT NULL
                )""")
            await menu_db.execute("CREATE INDEX IF NOT EXISTS idx_kategoriler_isim ON kategoriler(isim)")

            await menu_db.execute("""
                CREATE TABLE IF NOT EXISTS menu (
                    id SERIAL PRIMARY KEY,
                    ad TEXT NOT NULL,
                    fiyat REAL NOT NULL CHECK(fiyat >= 0),
                    kategori_id INTEGER NOT NULL REFERENCES kategoriler(id) ON DELETE CASCADE,
                    stok_durumu INTEGER DEFAULT 1,
                    UNIQUE(ad, kategori_id)
                )""")
            await menu_db.execute("CREATE INDEX IF NOT EXISTS idx_menu_kategori_id ON menu(kategori_id)")
            await menu_db.execute("CREATE INDEX IF NOT EXISTS idx_menu_ad ON menu(ad)")
            await menu_db.execute("CREATE INDEX IF NOT EXISTS idx_menu_stok_durumu ON menu(stok_durumu)")
        logger.info(f"✅ Menü veritabanı tabloları başarıyla doğrulandı/oluşturuldu.")
    except Exception as e: # pragma: no cover
        logger.critical(f"❌ Menü veritabanı tabloları başlatılırken kritik hata: {e}", exc_info=True)
        raise

async def init_stok_db():
    logger.info(f"Stok veritabanı tabloları kontrol ediliyor/oluşturuluyor...")
    try:
        async with db.transaction():
            await db.execute("""
                CREATE TABLE IF NOT EXISTS stok_kategorileri (
                    id SERIAL PRIMARY KEY,
                    ad TEXT UNIQUE NOT NULL
                )""")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_stok_kategorileri_ad ON stok_kategorileri(ad)")

            await db.execute("""
                CREATE TABLE IF NOT EXISTS stok_kalemleri (
                    id SERIAL PRIMARY KEY,
                    ad TEXT NOT NULL,
                    stok_kategori_id INTEGER NOT NULL REFERENCES stok_kategorileri(id) ON DELETE RESTRICT,
                    birim TEXT NOT NULL,
                    mevcut_miktar REAL DEFAULT 0 CHECK(mevcut_miktar >= 0),
                    min_stok_seviyesi REAL DEFAULT 0 CHECK(min_stok_seviyesi >= 0),
                    son_alis_fiyati REAL CHECK(son_alis_fiyati IS NULL OR son_alis_fiyati >= 0),
                    olusturulma_tarihi TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    guncellenme_tarihi TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ad, stok_kategori_id)
                )""")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_stok_kalemleri_kategori_id ON stok_kalemleri(stok_kategori_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_stok_kalemleri_ad ON stok_kalemleri(ad)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_stok_kalemleri_min_stok ON stok_kalemleri(mevcut_miktar, min_stok_seviyesi)")
        logger.info(f"✅ Stok veritabanı tabloları başarıyla doğrulandı/oluşturuldu.")
    except Exception as e: # pragma: no cover
        logger.critical(f"❌ Stok veritabanı tabloları başlatılırken kritik hata: {e}", exc_info=True)
        raise

# YENİ EKLENEN KISIM: Reçete Veritabanı Tabloları
async def init_recete_db():
    logger.info(f"Reçete veritabanı tabloları (ana db) kontrol ediliyor/oluşturuluyor...")
    try:
        async with db.transaction():
            await db.execute("""
                CREATE TABLE IF NOT EXISTS menu_urun_receteleri (
                    id SERIAL PRIMARY KEY,
                    menu_urun_id INTEGER NOT NULL,
                    aciklama TEXT,
                    porsiyon_birimi TEXT DEFAULT 'adet',
                    porsiyon_miktari REAL DEFAULT 1.0 CHECK(porsiyon_miktari > 0),
                    olusturulma_tarihi TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    guncellenme_tarihi TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (menu_urun_id)
                )""")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_menu_urun_receteleri_menu_urun_id ON menu_urun_receteleri(menu_urun_id)")

            await db.execute("""
                CREATE TABLE IF NOT EXISTS recete_bilesenleri (
                    id SERIAL PRIMARY KEY,
                    recete_id INTEGER NOT NULL REFERENCES menu_urun_receteleri(id) ON DELETE CASCADE,
                    stok_kalemi_id INTEGER NOT NULL REFERENCES stok_kalemleri(id) ON DELETE RESTRICT,
                    miktar REAL NOT NULL CHECK(miktar > 0),
                    birim TEXT NOT NULL,
                    olusturulma_tarihi TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    guncellenme_tarihi TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (recete_id, stok_kalemi_id)
                )""")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_recete_bilesenleri_recete_id ON recete_bilesenleri(recete_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_recete_bilesenleri_stok_kalemi_id ON recete_bilesenleri(stok_kalemi_id)")
        logger.info(f"✅ Reçete veritabanı tabloları (menu_urun_receteleri, recete_bilesenleri) ana db üzerinde başarıyla doğrulandı/oluşturuldu.")
    except Exception as e: # pragma: no cover
        logger.critical(f"❌ Reçete veritabanı tabloları (ana db) başlatılırken KRİTİK HATA: {e}", exc_info=True)
        raise
# YENİ EKLENEN KISIM SONU

async def init_databases():
    await init_db()
    await init_menu_db()
    await init_stok_db()
    await init_recete_db() # GÜNCELLENDİ: init_recete_db çağrısı eklendi

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

    cached_price_dict = await get_menu_price_dict()
    cached_stock_dict = await get_menu_stock_dict() # Stok kontrolü için

    processed_sepet = []
    for item in sepet:
        urun_adi_lower = item.urun.lower().strip()
        # Stok kontrolü
        stok_kontrol_degeri = cached_stock_dict.get(urun_adi_lower) # stok_durumu (1 veya 0)
        if stok_kontrol_degeri is None or stok_kontrol_degeri == 0: # Eğer ürün cache'de yoksa veya stok_durumu 0 ise
            logger.warning(f"Sipariş ekleme hatası: Ürün '{item.urun}' menüde/stokta bulunamadı veya stokta yok (cache).")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"'{item.urun}' adlı ürün stokta yok veya menüde bulunmuyor.")

        item_dict = item.model_dump()
        cached_fiyat = cached_price_dict.get(urun_adi_lower, item.fiyat)
        if cached_fiyat != item.fiyat: # pragma: no cover
            logger.warning(f"Fiyat uyuşmazlığı: Ürün '{item.urun}', Frontend Fiyatı: {item.fiyat}, Cache Fiyatı: {cached_fiyat}. Cache fiyatı kullanılacak.")
        item_dict['fiyat'] = cached_fiyat
        processed_sepet.append(item_dict)

    if not processed_sepet: # pragma: no cover
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Sepette geçerli ürün yok.")

    istek_ozet = ", ".join([f"{p_item['adet']}x {p_item['urun']}" for p_item in processed_sepet])
    try:
        async with db.transaction():
            siparis_id = await db.fetch_val("""
                INSERT INTO siparisler (masa, istek, yanit, sepet, zaman, durum, odeme_yontemi)
                VALUES (:masa, :istek, :yanit, :sepet, :zaman, 'bekliyor', NULL)
                RETURNING id
            """, { "masa": masa, "istek": istek or istek_ozet, "yanit": yanit, "sepet": json.dumps(processed_sepet, ensure_ascii=False), "zaman": db_zaman_kayit })
            if siparis_id is None: raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Sipariş kaydedilemedi.") # pragma: no cover

        siparis_bilgisi_ws = { "type": "siparis", "data": {"id": siparis_id, "masa": masa, "istek": istek or istek_ozet, "sepet": processed_sepet, "zaman": yanit_zaman_iso_str, "durum": "bekliyor", "odeme_yontemi": None}}
        await broadcast_message(aktif_mutfak_websocketleri, siparis_bilgisi_ws, "Mutfak/Masa")
        await broadcast_message(aktif_admin_websocketleri, siparis_bilgisi_ws, "Admin")
        await broadcast_message(aktif_kasa_websocketleri, siparis_bilgisi_ws, "Kasa")
        await update_table_status(masa, f"Sipariş verdi ({len(processed_sepet)} çeşit ürün)")
        logger.info(f"✅ Sipariş (ID: {siparis_id}) Masa: {masa} kaydedildi.")
        return { "mesaj": "Siparişiniz başarıyla alındı ve mutfağa iletildi.", "siparisId": siparis_id, "zaman": yanit_zaman_iso_str }
    except HTTPException as http_exc: # pragma: no cover
        raise http_exc
    except Exception as e: # pragma: no cover
        logger.error(f"❌ Sipariş ekleme hatası (Masa: {masa}): {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Sipariş işlenirken sunucu hatası.")

@app.post("/siparis-guncelle", tags=["Siparişler"]) # Bu endpoint yerine /siparis/{id} PATCH kullanılmalı
async def update_order_status_endpoint(
    data: SiparisGuncelleData,
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN, KullaniciRol.MUTFAK_PERSONELI, KullaniciRol.BARISTA]))
): # pragma: no cover
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
            except json.JSONDecodeError: # pragma: no cover
                order_dict['sepet'] = []
            if isinstance(order_dict.get('zaman'), datetime):
                 order_dict['zaman'] = order_dict['zaman'].isoformat()
            orders_data.append(order_dict)
        return {"orders": orders_data}
    except Exception as e: # pragma: no cover
        logger.error(f"❌ Tüm siparişler alınırken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Siparişler veritabanından alınırken bir sorun oluştu.")

@app.get("/admin/clear-menu-caches", tags=["Admin İşlemleri"])
async def clear_all_caches_endpoint(current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))):
    logger.info(f"Admin '{current_user.kullanici_adi}' tarafından manuel cache temizleme isteği alındı.")
    await update_system_prompt() # Bu fonksiyon cache'leri de temizliyor
    return {"message": "Menü, fiyat ve stok cache'leri başarıyla temizlendi. Sistem promptu güncellendi."}

@app.get("/menu", tags=["Menü"])
async def get_full_menu_endpoint():
    logger.info("Tam menü isteniyor (/menu)...")
    try:
        if not menu_db.is_connected: await menu_db.connect() # pragma: no cover
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
    except Exception as e: # pragma: no cover
        logger.error(f"❌ Tam menü alınırken veritabanı hatası: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Menü bilgileri alınırken bir sorun oluştu.")

@app.post("/menu/ekle", status_code=status.HTTP_201_CREATED, tags=["Menü Yönetimi"])
async def add_menu_item_endpoint(
    item_data: MenuEkleData,
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"📝 Menüye yeni ürün ekleme isteği (Kullanıcı: {current_user.kullanici_adi}): {item_data.ad} ({item_data.kategori})")
    try:
        if not menu_db.is_connected: await menu_db.connect() # pragma: no cover
        async with menu_db.transaction():
            await menu_db.execute("INSERT INTO kategoriler (isim) VALUES (:isim) ON CONFLICT (isim) DO NOTHING", {"isim": item_data.kategori})
            category_id_row = await menu_db.fetch_one("SELECT id FROM kategoriler WHERE LOWER(isim) = LOWER(:isim)", {"isim": item_data.kategori})
            if not category_id_row: raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Kategori oluşturulurken veya bulunurken bir sorun oluştu.") # pragma: no cover
            category_id = category_id_row['id']

            try:
                item_id = await menu_db.fetch_val("""
                    INSERT INTO menu (ad, fiyat, kategori_id, stok_durumu) VALUES (:ad, :fiyat, :kategori_id, 1) RETURNING id
                """, {"ad": item_data.ad, "fiyat": item_data.fiyat, "kategori_id": category_id})
            except Exception as e_db: # pragma: no cover
                 if "duplicate key value violates unique constraint" in str(e_db).lower() or "UNIQUE constraint failed" in str(e_db).lower():
                    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"'{item_data.ad}' adlı ürün bu kategoride zaten mevcut.")
                 logger.error(f"DB Hatası /menu/ekle: {e_db}", exc_info=True)
                 raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Veritabanı hatası: {str(e_db)}")

        await update_system_prompt()
        return {"mesaj": f"'{item_data.ad}' ürünü menüye başarıyla eklendi.", "itemId": item_id}
    except HTTPException as http_exc: # pragma: no cover
        raise http_exc
    except Exception as e: # pragma: no cover
        logger.error(f"❌ Menüye ürün eklenirken beklenmedik genel hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Menüye ürün eklenirken sunucuda bir hata oluştu.")

@app.delete("/menu/sil", tags=["Menü Yönetimi"])
async def delete_menu_item_endpoint(
    urun_adi: str = Query(..., min_length=1, description="Silinecek ürünün tam adı."),
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"🗑️ Menüden ürün silme isteği (Kullanıcı: {current_user.kullanici_adi}): {urun_adi}")
    try:
        if not menu_db.is_connected: await menu_db.connect() # pragma: no cover
        async with menu_db.transaction():
            item_to_delete = await menu_db.fetch_one("SELECT id FROM menu WHERE LOWER(ad) = LOWER(:ad)", {"ad": urun_adi})
            if not item_to_delete: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{urun_adi}' adlı ürün menüde bulunamadı.") # pragma: no cover

            await menu_db.execute("DELETE FROM menu WHERE id = :id", {"id": item_to_delete['id']}) # ID ile silmek daha güvenli

        await update_system_prompt()
        return {"mesaj": f"'{urun_adi}' ürünü menüden başarıyla silindi."}
    except HTTPException as http_exc: # pragma: no cover
        raise http_exc
    except Exception as e: # pragma: no cover
        logger.error(f"❌ Menüden ürün silinirken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Menüden ürün silinirken bir sunucu hatası oluştu.")

@app.get("/admin/menu/kategoriler", response_model=List[MenuKategori], tags=["Menü Yönetimi"])
async def list_menu_kategoriler(
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"Admin '{current_user.kullanici_adi}' menü kategorilerini listeliyor.")
    if not menu_db.is_connected: await menu_db.connect() # pragma: no cover
    query = "SELECT id, isim FROM kategoriler ORDER BY isim"
    kategoriler_raw = await menu_db.fetch_all(query)
    return [MenuKategori(**row) for row in kategoriler_raw]

@app.delete("/admin/menu/kategoriler/{kategori_id}", status_code=status.HTTP_200_OK, tags=["Menü Yönetimi"])
async def delete_menu_kategori(
    kategori_id: int = Path(..., description="Silinecek menü kategorisinin ID'si"),
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.warning(f"❗ Admin '{current_user.kullanici_adi}' MENÜ KATEGORİSİ silme isteği: ID {kategori_id}.")
    try:
        if not menu_db.is_connected: await menu_db.connect() # pragma: no cover
        async with menu_db.transaction():
            kategori_check = await menu_db.fetch_one("SELECT isim FROM kategoriler WHERE id = :id", {"id": kategori_id})
            if not kategori_check: # pragma: no cover
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"ID: {kategori_id} ile eşleşen menü kategorisi bulunamadı.")

            await menu_db.execute("DELETE FROM kategoriler WHERE id = :id", {"id": kategori_id}) # ON DELETE CASCADE menu tablosunu etkiler

        await update_system_prompt()
        logger.info(f"✅ Menü kategorisi '{kategori_check['isim']}' (ID: {kategori_id}) ve bağlı tüm ürünler başarıyla silindi.")
        return {"mesaj": f"'{kategori_check['isim']}' adlı menü kategorisi ve bu kategoriye ait tüm ürünler başarıyla silindi."}
    except HTTPException as http_exc: # pragma: no cover
        raise http_exc
    except Exception as e: # pragma: no cover
        logger.error(f"❌ Menü kategorisi (ID: {kategori_id}) silinirken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Menü kategorisi silinirken bir sunucu hatası oluştu.")

@app.post("/admin/stok/kategoriler", response_model=StokKategori, status_code=status.HTTP_201_CREATED, tags=["Stok Yönetimi"])
async def create_stok_kategori(
    stok_kategori_data: StokKategoriCreate,
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"Admin '{current_user.kullanici_adi}' yeni stok kategorisi oluşturuyor: {stok_kategori_data.ad}")
    try:
        query_check = "SELECT id FROM stok_kategorileri WHERE LOWER(ad) = LOWER(:ad)"
        existing_cat = await db.fetch_one(query_check, {"ad": stok_kategori_data.ad})
        if existing_cat: # pragma: no cover
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"'{stok_kategori_data.ad}' adlı stok kategorisi zaten mevcut.")

        query_insert = "INSERT INTO stok_kategorileri (ad) VALUES (:ad) RETURNING id, ad"
        created_cat_row = await db.fetch_one(query_insert, {"ad": stok_kategori_data.ad})
        if not created_cat_row: # pragma: no cover
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Stok kategorisi oluşturulamadı.")
        logger.info(f"Stok kategorisi '{created_cat_row['ad']}' (ID: {created_cat_row['id']}) oluşturuldu.")
        return StokKategori(**created_cat_row)
    except HTTPException: # pragma: no cover
        raise
    except Exception as e: # pragma: no cover
        logger.error(f"Stok kategorisi '{stok_kategori_data.ad}' oluşturulurken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Stok kategorisi oluşturulurken bir hata oluştu.")

@app.get("/admin/stok/kategoriler", response_model=List[StokKategori], tags=["Stok Yönetimi"])
async def list_stok_kategoriler(
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"Admin '{current_user.kullanici_adi}' stok kategorilerini listeliyor.")
    query = "SELECT id, ad FROM stok_kategorileri ORDER BY ad"
    rows = await db.fetch_all(query)
    return [StokKategori(**row) for row in rows]

@app.put("/admin/stok/kategoriler/{stok_kategori_id}", response_model=StokKategori, tags=["Stok Yönetimi"])
async def update_stok_kategori(
    stok_kategori_id: int,
    stok_kategori_data: StokKategoriCreate,
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"Admin '{current_user.kullanici_adi}' stok kategorisi ID {stok_kategori_id} güncelliyor: Yeni ad '{stok_kategori_data.ad}'")
    try:
        query_check_id = "SELECT id FROM stok_kategorileri WHERE id = :id"
        target_cat = await db.fetch_one(query_check_id, {"id": stok_kategori_id})
        if not target_cat: # pragma: no cover
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"ID: {stok_kategori_id} ile stok kategorisi bulunamadı.")

        query_check_name = "SELECT id FROM stok_kategorileri WHERE LOWER(ad) = LOWER(:ad) AND id != :id_param"
        existing_cat_with_name = await db.fetch_one(query_check_name, {"ad": stok_kategori_data.ad, "id_param": stok_kategori_id})
        if existing_cat_with_name: # pragma: no cover
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"'{stok_kategori_data.ad}' adlı stok kategorisi zaten başka bir kayıtta mevcut.")

        query_update = "UPDATE stok_kategorileri SET ad = :ad WHERE id = :id RETURNING id, ad"
        updated_row = await db.fetch_one(query_update, {"ad": stok_kategori_data.ad, "id": stok_kategori_id})
        if not updated_row: # pragma: no cover
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Stok kategorisi güncellenemedi.")
        logger.info(f"Stok kategorisi ID {stok_kategori_id} güncellendi. Yeni ad: {updated_row['ad']}")
        return StokKategori(**updated_row)
    except HTTPException: # pragma: no cover
        raise
    except Exception as e: # pragma: no cover
        logger.error(f"Stok kategorisi ID {stok_kategori_id} güncellenirken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Stok kategorisi güncellenirken bir hata oluştu.")


@app.delete("/admin/stok/kategoriler/{stok_kategori_id}", status_code=status.HTTP_200_OK, tags=["Stok Yönetimi"])
async def delete_stok_kategori(
    stok_kategori_id: int,
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"Admin '{current_user.kullanici_adi}' stok kategorisi ID {stok_kategori_id} siliyor.")
    try:
        query_check_items = "SELECT COUNT(*) as item_count FROM stok_kalemleri WHERE stok_kategori_id = :kategori_id"
        item_count_row = await db.fetch_one(query_check_items, {"kategori_id": stok_kategori_id})
        if item_count_row and item_count_row["item_count"] > 0: # pragma: no cover
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Bu stok kategorisi ({item_count_row['item_count']} kalem) tarafından kullanıldığı için silinemez. Önce kalemleri başka kategoriye taşıyın veya silin.")

        query_delete = "DELETE FROM stok_kategorileri WHERE id = :id RETURNING ad"
        deleted_cat_name_row = await db.fetch_one(query_delete, {"id": stok_kategori_id})
        if not deleted_cat_name_row: # pragma: no cover
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"ID: {stok_kategori_id} ile stok kategorisi bulunamadı.")

        logger.info(f"Stok kategorisi '{deleted_cat_name_row['ad']}' (ID: {stok_kategori_id}) başarıyla silindi.")
        return {"mesaj": f"Stok kategorisi '{deleted_cat_name_row['ad']}' başarıyla silindi."}
    except HTTPException: # pragma: no cover
        raise
    except Exception as e: # pragma: no cover
        if "foreign key constraint" in str(e).lower():
             raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Bu stok kategorisi hala stok kalemleri tarafından kullanıldığı için silinemez.")
        logger.error(f"Stok kategorisi ID {stok_kategori_id} silinirken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Stok kategorisi silinirken bir hata oluştu.")

@app.post("/admin/stok/kalemler", response_model=StokKalemi, status_code=status.HTTP_201_CREATED, tags=["Stok Yönetimi"])
async def create_stok_kalemi(
    stok_kalemi_data: StokKalemiCreate,
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"Admin '{current_user.kullanici_adi}' yeni stok kalemi ekliyor: {stok_kalemi_data.ad}")
    try:
        cat_check = await db.fetch_one("SELECT id FROM stok_kategorileri WHERE id = :cat_id", {"cat_id": stok_kalemi_data.stok_kategori_id})
        if not cat_check: # pragma: no cover
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"ID: {stok_kalemi_data.stok_kategori_id} ile stok kategorisi bulunamadı.")

        query_insert = """
            INSERT INTO stok_kalemleri (ad, stok_kategori_id, birim, mevcut_miktar, min_stok_seviyesi, son_alis_fiyati, guncellenme_tarihi)
            VALUES (:ad, :stok_kategori_id, :birim, :mevcut_miktar, :min_stok_seviyesi, :son_alis_fiyati, :guncellenme_tarihi)
            RETURNING id, ad, stok_kategori_id, birim, mevcut_miktar, min_stok_seviyesi, son_alis_fiyati
        """
        now_ts = datetime.now(TR_TZ)
        values = stok_kalemi_data.model_dump()
        values["guncellenme_tarihi"] = now_ts

        created_item_row = await db.fetch_one(query_insert, values)
        if not created_item_row: # pragma: no cover
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Stok kalemi oluşturulamadı.")

        logger.info(f"Stok kalemi '{created_item_row['ad']}' (ID: {created_item_row['id']}) başarıyla oluşturuldu.")
        # stok_kategori_ad Pydantic modelinde Optional olduğu için sorun olmaz.
        # İstenirse burada join ile çekilip eklenebilir ama create response için genelde gerekmez.
        return StokKalemi(**created_item_row)
    except HTTPException: # pragma: no cover
        raise
    except Exception as e: # pragma: no cover
        if "duplicate key value violates unique constraint" in str(e).lower() or "UNIQUE constraint failed" in str(e).lower():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"'{stok_kalemi_data.ad}' adlı stok kalemi bu kategoride zaten mevcut.")
        logger.error(f"Stok kalemi '{stok_kalemi_data.ad}' oluşturulurken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Stok kalemi oluşturulurken bir hata oluştu.")

@app.get("/admin/stok/kalemler", response_model=List[StokKalemi], tags=["Stok Yönetimi"])
async def list_stok_kalemleri(
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN])),
    kategori_id: Optional[int] = Query(None, description="Belirli bir stok kategorisindeki kalemleri filtrele"),
    dusuk_stok: Optional[bool] = Query(None, description="Sadece minimum stok seviyesinin altındaki kalemleri göster")
):
    logger.info(f"Admin '{current_user.kullanici_adi}' stok kalemlerini listeliyor. Kategori ID: {kategori_id}, Düşük Stok: {dusuk_stok}")

    query_base = """
        SELECT sk.id, sk.ad, sk.stok_kategori_id, sk.birim, sk.mevcut_miktar, sk.min_stok_seviyesi, sk.son_alis_fiyati, s_kat.ad as stok_kategori_ad
        FROM stok_kalemleri sk
        JOIN stok_kategorileri s_kat ON sk.stok_kategori_id = s_kat.id
    """
    conditions = []
    values = {}

    if kategori_id is not None:
        conditions.append("sk.stok_kategori_id = :kategori_id")
        values["kategori_id"] = kategori_id

    if dusuk_stok is True:
        conditions.append("sk.mevcut_miktar < sk.min_stok_seviyesi")

    if conditions:
        query_base += " WHERE " + " AND ".join(conditions)

    query_base += " ORDER BY s_kat.ad, sk.ad"

    rows = await db.fetch_all(query_base, values)
    return [StokKalemi(**row) for row in rows]

@app.get("/admin/stok/kalemler/{stok_kalemi_id}", response_model=StokKalemi, tags=["Stok Yönetimi"])
async def get_stok_kalemi_detay(
    stok_kalemi_id: int,
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"Admin '{current_user.kullanici_adi}' stok kalemi ID {stok_kalemi_id} detayını istiyor.")
    query = """
        SELECT sk.id, sk.ad, sk.stok_kategori_id, sk.birim, sk.mevcut_miktar, sk.min_stok_seviyesi, sk.son_alis_fiyati, s_kat.ad as stok_kategori_ad
        FROM stok_kalemleri sk
        JOIN stok_kategorileri s_kat ON sk.stok_kategori_id = s_kat.id
        WHERE sk.id = :id
    """
    row = await db.fetch_one(query, {"id": stok_kalemi_id})
    if not row: # pragma: no cover
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Stok kalemi bulunamadı.")
    return StokKalemi(**row)

@app.put("/admin/stok/kalemler/{stok_kalemi_id}", response_model=StokKalemi, tags=["Stok Yönetimi"])
async def update_stok_kalemi(
    stok_kalemi_id: int,
    stok_kalemi_data: StokKalemiUpdate,
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"Admin '{current_user.kullanici_adi}' stok kalemi ID {stok_kalemi_id} güncelliyor: {stok_kalemi_data.model_dump_json(exclude_unset=True)}")

    try:
        async with db.transaction():
            existing_item_query = """
                SELECT sk.id, sk.ad, sk.stok_kategori_id, sk.birim, sk.mevcut_miktar, sk.min_stok_seviyesi, sk.son_alis_fiyati, s_kat.ad as stok_kategori_ad
                FROM stok_kalemleri sk
                JOIN stok_kategorileri s_kat ON sk.stok_kategori_id = s_kat.id
                WHERE sk.id = :id
            """
            existing_item_record = await db.fetch_one(existing_item_query, {"id": stok_kalemi_id})
            if not existing_item_record: # pragma: no cover
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Güncellenecek stok kalemi bulunamadı.")

            existing_item = StokKalemi.model_validate(existing_item_record)
            update_dict = stok_kalemi_data.model_dump(exclude_unset=True)

            if not update_dict: # pragma: no cover
                logger.info(f"Stok kalemi ID {stok_kalemi_id} için güncellenecek bir alan belirtilmedi.")
                return existing_item

            if "stok_kategori_id" in update_dict and update_dict["stok_kategori_id"] != existing_item.stok_kategori_id: # pragma: no cover
                cat_check = await db.fetch_one("SELECT id FROM stok_kategorileri WHERE id = :cat_id", {"cat_id": update_dict["stok_kategori_id"]})
                if not cat_check:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"ID: {update_dict['stok_kategori_id']} ile yeni stok kategorisi bulunamadı.")

            check_ad = update_dict.get("ad", existing_item.ad)
            check_cat_id = update_dict.get("stok_kategori_id", existing_item.stok_kategori_id)

            if "ad" in update_dict or "stok_kategori_id" in update_dict: # pragma: no cover
                unique_check = await db.fetch_one(
                    "SELECT id FROM stok_kalemleri WHERE LOWER(ad) = LOWER(:ad) AND stok_kategori_id = :cat_id AND id != :item_id",
                    {"ad": check_ad, "cat_id": check_cat_id, "item_id": stok_kalemi_id}
                )
                if unique_check:
                    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"'{check_ad}' adlı stok kalemi '{check_cat_id}' ID'li kategoride zaten mevcut.")

            update_dict["guncellenme_tarihi"] = datetime.now(TR_TZ)
            set_clauses = [f"{key} = :{key}" for key in update_dict.keys()]
            query_update_stmt = f"UPDATE stok_kalemleri SET {', '.join(set_clauses)} WHERE id = :stok_kalemi_id_param RETURNING id" # PostgreSQL RETURNING

            updated_item_id_row = await db.fetch_one(query_update_stmt, {**update_dict, "stok_kalemi_id_param": stok_kalemi_id})

            if not updated_item_id_row or not updated_item_id_row['id']: # pragma: no cover
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Stok kalemi güncellenemedi (UPDATE sorgusu sonuç döndürmedi).")

        # Güncellenmiş tam veriyi çek
        final_updated_row_record = await db.fetch_one(existing_item_query, {"id": updated_item_id_row['id']}) # existing_item_query join'li olduğu için tekrar kullanıldı
        if not final_updated_row_record: # pragma: no cover
            logger.error(f"Stok kalemi ID {stok_kalemi_id} güncellendi ancak hemen ardından detayları çekilemedi.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Stok kalemi güncellendi ancak sonuç verisi alınamadı.")

        logger.info(f"Stok kalemi ID {stok_kalemi_id} başarıyla güncellendi.")
        return StokKalemi.model_validate(final_updated_row_record)

    except HTTPException: # pragma: no cover
        raise
    except Exception as e: # pragma: no cover
        if "duplicate key value violates unique constraint" in str(e).lower() or \
           "UNIQUE constraint failed: stok_kalemleri.ad, stok_kalemleri.stok_kategori_id" in str(e) or \
           "UNIQUE constraint failed: stok_kalemleri.ad" in str(e):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Bu stok kalemi adı ve kategori kombinasyonu zaten mevcut veya başka bir unique kısıtlama ihlal edildi.")
        logger.error(f"Stok kalemi ID {stok_kalemi_id} güncellenirken beklenmedik hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Stok kalemi güncellenirken bir hata oluştu: {type(e).__name__}")

@app.delete("/admin/stok/kalemler/{stok_kalemi_id}", status_code=status.HTTP_200_OK, tags=["Stok Yönetimi"])
async def delete_stok_kalemi(
    stok_kalemi_id: int,
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"Admin '{current_user.kullanici_adi}' stok kalemi ID {stok_kalemi_id} siliyor.")
    try:
        # Reçete bileşenlerinde kullanılıyor mu kontrol et (FOREIGN KEY RESTRICT nedeniyle)
        check_bilesen = await db.fetch_one(
            "SELECT COUNT(*) as count FROM recete_bilesenleri WHERE stok_kalemi_id = :id",
            {"id": stok_kalemi_id}
        )
        if check_bilesen and check_bilesen['count'] > 0: # pragma: no cover
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Bu stok kalemi ({check_bilesen['count']} reçetede) kullanıldığı için silinemez. Önce ilgili reçetelerden çıkarın."
            )

        deleted_row = await db.fetch_one("DELETE FROM stok_kalemleri WHERE id = :id RETURNING ad", {"id": stok_kalemi_id})
        if not deleted_row: # pragma: no cover
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Silinecek stok kalemi bulunamadı.")
        logger.info(f"Stok kalemi '{deleted_row['ad']}' (ID: {stok_kalemi_id}) başarıyla silindi.")
        return {"mesaj": f"Stok kalemi '{deleted_row['ad']}' başarıyla silindi."}
    except HTTPException: # pragma: no cover
        raise
    except Exception as e: # pragma: no cover
        if "foreign key constraint" in str(e).lower() and "recete_bilesenleri_stok_kalemi_id_fkey" in str(e).lower():
             raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Bu stok kalemi hala reçetelerde kullanıldığı için silinemez.")
        logger.error(f"Stok kalemi ID {stok_kalemi_id} silinirken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Stok kalemi silinirken bir hata oluştu.")

@app.post("/yanitla", tags=["Yapay Zeka"])
async def handle_message_endpoint(request: Request, data: dict = Body(...)):
    user_message = data.get("text", "").strip()
    table_id = data.get("masa", "bilinmiyor")
    previous_ai_state_from_frontend = data.get("onceki_ai_durumu", None)

    session_id = request.session.get("session_id")
    if not session_id: # pragma: no cover
        session_id = secrets.token_hex(16)
        request.session["session_id"] = session_id
        request.session["chat_history"] = []

    chat_history = request.session.get("chat_history", [])

    logger.info(f"💬 AI Yanıt isteği: Masa '{table_id}', Session ID: '{session_id}', Kullanıcı Mesajı: '{user_message}'")
    if previous_ai_state_from_frontend: # pragma: no cover
        logger.info(f"🧠 Frontend'den alınan önceki AI durumu: {json.dumps(previous_ai_state_from_frontend, ensure_ascii=False, indent=2)}")

    if not user_message: # pragma: no cover
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Mesaj boş olamaz.")

    if SYSTEM_PROMPT is None: # pragma: no cover
        await update_system_prompt()
        if SYSTEM_PROMPT is None:
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="AI asistanı şu anda hazır değil (sistem mesajı eksik).")

    try:
        messages_for_openai = [SYSTEM_PROMPT]

        if previous_ai_state_from_frontend: # pragma: no cover
            context_for_ai_prompt = "Bir önceki etkileşimden önemli bilgiler (müşterinin bir sonraki yanıtı bu bağlamda olabilir):\n"
            current_sepet_items = previous_ai_state_from_frontend.get("sepet", [])
            if current_sepet_items:
                sepet_str_list = [f"- {item.get('adet',0)} x {item.get('urun','Bilinmeyen')} ({item.get('fiyat',0.0):.2f} TL)" for item in current_sepet_items]
                context_for_ai_prompt += f"Mevcut Sepet:\n" + "\n".join(sepet_str_list) + "\n"
                context_for_ai_prompt += f"Mevcut Sepet Toplam Tutar: {previous_ai_state_from_frontend.get('toplam_tutar', 0.0):.2f} TL\n"
            if previous_ai_state_from_frontend.get("onerilen_urun"):
                context_for_ai_prompt += f"Bir Önceki Önerilen Ürün: {previous_ai_state_from_frontend['onerilen_urun']}\n"
            if previous_ai_state_from_frontend.get("konusma_metni"):
                context_for_ai_prompt += f"Bir Önceki AI Konuşma Metni: \"{previous_ai_state_from_frontend['konusma_metni']}\"\n"
            if context_for_ai_prompt.strip() != "Bir önceki etkileşimden önemli bilgiler (müşterinin bir sonraki yanıtı bu bağlamda olabilir):":
                messages_for_openai.append({"role": "system", "name": "previous_context_summary", "content": context_for_ai_prompt.strip()})
                logger.info(f"🤖 AI'a gönderilen ek bağlam özeti: {context_for_ai_prompt.strip()}")

        messages_for_openai.extend(chat_history)
        messages_for_openai.append({"role": "user", "content": user_message})

        logger.debug(f"OpenAI'ye gönderilecek tam mesaj listesi:\n{json.dumps(messages_for_openai, ensure_ascii=False, indent=2)}")

        response = openai_client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages_for_openai,
            temperature=0.2,
            max_tokens=600,
        )
        ai_reply_content = response.choices[0].message.content
        ai_reply = ai_reply_content.strip() if ai_reply_content else "Üzgünüm, şu anda bir yanıt üretemiyorum." # pragma: no cover

        parsed_ai_json = None
        if ai_reply.startswith("{") and ai_reply.endswith("}"):
            try:
                parsed_ai_json = json.loads(ai_reply)
                logger.info(f"AI JSON formatında yanıt verdi (parse başarılı): {json.dumps(parsed_ai_json, ensure_ascii=False, indent=2)}")
            except json.JSONDecodeError: # pragma: no cover
                logger.warning(f"AI JSON gibi görünen ama geçersiz bir yanıt verdi, düz metin olarak işlenecek: {ai_reply[:300]}...")
        else:
             logger.info(f"AI düz metin formatında yanıt verdi: {ai_reply[:300]}...")

        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": ai_reply})
        request.session["chat_history"] = chat_history[-10:]

        return {"reply": ai_reply, "sessionId": session_id}

    except OpenAIError as e: # pragma: no cover
        logger.error(f"❌ OpenAI API hatası: {type(e).__name__} - {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"AI servisinden yanıt alınırken bir sorun oluştu: {type(e).__name__}")
    except Exception as e: # pragma: no cover
        logger.error(f"❌ /yanitla endpoint genel hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Mesajınız işlenirken beklenmedik bir sunucu hatası oluştu.")

SUPPORTED_LANGUAGES = {"tr-TR", "en-US", "en-GB", "fr-FR", "de-DE"}
@app.post("/sesli-yanit", tags=["Yapay Zeka"])
async def generate_speech_endpoint(data: SesliYanitData):
    if not tts_client: raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Sesli yanıt servisi şu anda kullanılamıyor.") # pragma: no cover
    if data.language not in SUPPORTED_LANGUAGES: raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Desteklenmeyen dil: {data.language}.") # pragma: no cover

    cleaned_text = temizle_emoji(data.text)
    try: # JSON içinden konuşma metnini ayıkla
        if cleaned_text.strip().startswith("{") and cleaned_text.strip().endswith("}"): # pragma: no cover
            parsed_json = json.loads(cleaned_text)
            if "konusma_metni" in parsed_json and isinstance(parsed_json["konusma_metni"], str):
                cleaned_text = parsed_json["konusma_metni"]
                logger.info(f"Sesli yanıt için JSON'dan 'konusma_metni' çıkarıldı: {cleaned_text[:100]}...")
            else:
                logger.warning("Sesli yanıt için gelen JSON'da 'konusma_metni' bulunamadı veya string değil, ham metin kullanılacak.")
    except json.JSONDecodeError: # pragma: no cover
        pass # JSON değilse, olduğu gibi kullan

    if not cleaned_text.strip(): raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Sese dönüştürülecek geçerli bir metin bulunamadı.") # pragma: no cover

    try:
        synthesis_input = texttospeech.SynthesisInput(text=cleaned_text)
        # Studio ve WaveNet sesleri daha kalitelidir ancak daha pahalı olabilir.
        # Örnek HD (Chirp) ses modeli (daha yeni ve kaliteli):
        voice_name = "tr-TR-Chirp3-HD-Laomedeia" if data.language == "tr-TR" else None
        # Eski Studio veya WaveNet örneği:
        # voice_name = "tr-TR-Studio-B" if data.language == "tr-TR" else None # ("tr-TR-Wavenet-B" de olabilir)

        voice_params = texttospeech.VoiceSelectionParams(
            language_code=data.language,
            name=voice_name, # Belirli bir ses adı belirtilirse kullanılır
            ssml_gender=(texttospeech.SsmlVoiceGender.FEMALE if data.language == "tr-TR" and voice_name else texttospeech.SsmlVoiceGender.NEUTRAL)
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.1 # Biraz daha hızlı konuşma
        )
        response_tts = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice_params, audio_config=audio_config
        )
        return Response(content=response_tts.audio_content, media_type="audio/mpeg")
    except google_exceptions.GoogleAPIError as e_google: # pragma: no cover
        detail_msg = f"Google TTS servisinden ses üretilirken bir hata oluştu: {getattr(e_google, 'message', str(e_google))}"
        status_code_tts = status.HTTP_503_SERVICE_UNAVAILABLE
        if "API key not valid" in str(e_google) or "permission" in str(e_google).lower() or "RESOURCE_EXHAUSTED" in str(e_google):
            detail_msg = "Google TTS servisi için kimlik/kota sorunu veya kaynak yetersiz."
        elif "Requested voice not found" in str(e_google) or "Invalid DefaultVoice" in str(e_google):
            detail_msg = f"İstenen ses modeli ({voice_name}) bulunamadı veya geçersiz."; status_code_tts = status.HTTP_400_BAD_REQUEST
        logger.error(f"❌ Google TTS API hatası: {e_google}", exc_info=True)
        raise HTTPException(status_code=status_code_tts, detail=detail_msg)
    except Exception as e: # pragma: no cover
        logger.error(f"❌ Sesli yanıt endpoint'inde beklenmedik hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Sesli yanıt oluşturulurken beklenmedik bir sunucu hatası oluştu.")

@app.post("/kasa/siparis/{siparis_id}/odendi", tags=["Kasa İşlemleri"])
async def mark_order_as_paid_endpoint(
    siparis_id: int = Path(..., description="Ödendi olarak işaretlenecek siparişin ID'si"),
    odeme_bilgisi: KasaOdemeData = Body(...),
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN, KullaniciRol.KASIYER]))
):
    logger.info(f"💰 Kasa: Sipariş {siparis_id} ödendi olarak işaretleniyor (Kullanıcı: {current_user.kullanici_adi}). Ödeme: {odeme_bilgisi.odeme_yontemi}")
    simdiki_zaman_obj = datetime.now(TR_TZ) # Stok güncelleme zamanı için

    try:
        async with db.transaction():
            # Siparişin varlığını ve mevcut durumunu kontrol et
            order_check = await db.fetch_one(
                "SELECT id, masa, durum, sepet FROM siparisler WHERE id = :id", # Sepeti de alıyoruz stok düşme için
                {"id": siparis_id}
            )
            if not order_check: # pragma: no cover
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sipariş bulunamadı.")
            if order_check["durum"] == Durum.ODENDI.value: # pragma: no cover
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Sipariş zaten ödendi.")
            if order_check["durum"] == Durum.IPTAL.value: # pragma: no cover
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="İptal edilmiş sipariş ödenemez.")

            # Sipariş durumunu "odendi" olarak güncelle
            updated_order_raw = await db.fetch_one(
                """UPDATE siparisler
                   SET durum = :yeni_durum, odeme_yontemi = :odeme_yontemi
                   WHERE id = :id
                   RETURNING id, masa, durum, sepet, istek, zaman, odeme_yontemi""",
                {"yeni_durum": Durum.ODENDI.value, "odeme_yontemi": odeme_bilgisi.odeme_yontemi, "id": siparis_id}
            )
            
            if not updated_order_raw: # pragma: no cover
                logger.error(f"Sipariş {siparis_id} 'odendi' olarak güncellenemedi (DB update sonuç döndürmedi).")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Sipariş durumu güncellenirken bir veritabanı sorunu oluştu.")
            
            logger.info(f"Sipariş {siparis_id} durumu '{Durum.ODENDI.value}' olarak güncellendi. Ödeme yöntemi: {odeme_bilgisi.odeme_yontemi}. Şimdi stoklar düşülecek.")

            # STOK DÜŞÜRME MANTIĞI BAŞLANGICI
            # updated_order_raw["sepet"] ödeme anındaki sepeti içerir (RETURNING ile alındığı için)
            try:
                sepet_items_for_stock_deduction = json.loads(updated_order_raw["sepet"] or "[]")
            except json.JSONDecodeError: # pragma: no cover
                logger.error(f"Sipariş {siparis_id} için ödeme sonrası sepet JSON parse hatası. Stok düşülemiyor.", exc_info=True)
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Ödenen siparişin sepeti okunamadığı için stok güncellenemedi.")

            if not sepet_items_for_stock_deduction:
                logger.info(f"Sipariş {siparis_id} (ödendi) sepeti boş. Stok düşme işlemi yapılmayacak.")
            else:
                for item_in_sepet_data in sepet_items_for_stock_deduction:
                    menu_item_name = item_in_sepet_data.get('urun')
                    ordered_quantity = item_in_sepet_data.get('adet')

                    if not menu_item_name or not isinstance(ordered_quantity, int) or ordered_quantity <= 0:
                        logger.warning(f"Sipariş {siparis_id} (ödendi) içindeki bir sepet öğesi geçersiz (Ürün: {menu_item_name}, Adet: {ordered_quantity}). Bu öğe için stok düşme atlanıyor.")
                        continue
                    
                    menu_item_name_lower = menu_item_name.lower().strip()

                    is_menu_db_separate = menu_db != db
                    if is_menu_db_separate and not menu_db.is_connected: # pragma: no cover
                        try:
                            await menu_db.connect()
                        except Exception as e_connect_menu: # pragma: no cover
                            logger.error(f"Ödeme sırasında menu_db'ye bağlanırken hata (Sip.ID: {siparis_id}): {e_connect_menu}", exc_info=True)
                            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Menü veritabanına ulaşılamadı (ödeme).")
                    
                    menu_item_record_from_menu_db = await menu_db.fetch_one(
                        "SELECT id FROM menu WHERE LOWER(ad) = :ad_lower",
                        {"ad_lower": menu_item_name_lower}
                    )

                    if not menu_item_record_from_menu_db:
                        logger.warning(f"'{menu_item_name}' için menüde kayıt bulunamadı (ödeme). Stok düşme atlanıyor. Sipariş ID: {siparis_id}")
                        continue
                    
                    menu_urun_id_from_db = menu_item_record_from_menu_db['id']

                    recipe_main_info = await db.fetch_one(
                        "SELECT id FROM menu_urun_receteleri WHERE menu_urun_id = :menu_urun_id",
                        {"menu_urun_id": menu_urun_id_from_db}
                    )

                    if not recipe_main_info:
                        logger.info(f"'{menu_item_name}' (Menü ID: {menu_urun_id_from_db}) için reçete yok (ödeme). Stok düşme atlanıyor. Sipariş ID: {siparis_id}")
                        continue
                    
                    recete_id_from_db = recipe_main_info['id']
                    recipe_components = await db.fetch_all(
                        "SELECT stok_kalemi_id, miktar FROM recete_bilesenleri WHERE recete_id = :recete_id",
                        {"recete_id": recete_id_from_db}
                    )

                    if not recipe_components:
                        logger.info(f"'{menu_item_name}' (Reçete ID: {recete_id_from_db}) için bileşen yok (ödeme). Stok düşme atlanıyor. Sipariş ID: {siparis_id}")
                        continue
                    
                    for component in recipe_components:
                        stok_kalemi_id_to_deduct = component['stok_kalemi_id']
                        quantity_per_recipe_unit = component['miktar']
                        total_quantity_to_deduct_for_stock_item = ordered_quantity * quantity_per_recipe_unit

                        if total_quantity_to_deduct_for_stock_item <= 0:
                            logger.warning(f"Stok ID {stok_kalemi_id_to_deduct} için hesaplanan düşülecek miktar ({total_quantity_to_deduct_for_stock_item}) geçersiz (ödeme). Atlanıyor. Sipariş ID: {siparis_id}")
                            continue
                        
                        update_stock_query = """
                            UPDATE stok_kalemleri SET mevcut_miktar = mevcut_miktar - :miktar_dus, guncellenme_tarihi = :guncellenme_tarihi
                            WHERE id = :stok_kalemi_id RETURNING ad, mevcut_miktar;"""
                        
                        try:
                            updated_stock_item = await db.fetch_one(
                                 query=update_stock_query,
                                 values={
                                     "miktar_dus": total_quantity_to_deduct_for_stock_item,
                                     "stok_kalemi_id": stok_kalemi_id_to_deduct,
                                     "guncellenme_tarihi": simdiki_zaman_obj 
                                 }
                            )
                            if updated_stock_item:
                                 logger.info(f"  ➡️ Ödeme Sonrası Stok Güncellendi: Stok '{updated_stock_item['ad']}' (ID: {stok_kalemi_id_to_deduct}), Düşülen: {total_quantity_to_deduct_for_stock_item}, Yeni Miktar: {updated_stock_item['mevcut_miktar']}. Sipariş ID: {siparis_id}")
                            else: # pragma: no cover
                                 logger.error(f"  ⚠️ Ödeme Sonrası Stok Güncellenemedi (RETURNING yok): Stok ID {stok_kalemi_id_to_deduct}. Sipariş ID: {siparis_id}")
                                 raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Stok kalemi ID {stok_kalemi_id_to_deduct} güncellenirken sorun (ödeme).")
                        except Exception as e_stock_update_payment: # pragma: no cover
                            logger.error(f"  ❌ Ödeme Sonrası Stok ID {stok_kalemi_id_to_deduct} güncellenirken KRİTİK HATA: {e_stock_update_payment}. Sipariş ID: {siparis_id}", exc_info=True)
                            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Ödeme sırasında stok güncelleme hatası.") from e_stock_update_payment
            # STOK DÜŞÜRME MANTIĞI SONU

            # WebSocket ve response hazırlığı (bu kısım sizin kodunuzda zaten vardı, buraya taşıdım)
            updated_order = dict(updated_order_raw)
            try:
                updated_order["sepet"] = json.loads(updated_order.get("sepet", "[]"))
            except json.JSONDecodeError: # pragma: no cover
                 updated_order["sepet"] = [] 
            if isinstance(updated_order.get('zaman'), datetime): # pragma: no cover
                updated_order['zaman'] = updated_order['zaman'].isoformat()

            notif_data = {**updated_order, "zaman": datetime.now(TR_TZ).isoformat()} 
            notification = {"type": "durum", "data": notif_data}
            await broadcast_message(aktif_mutfak_websocketleri, notification, "Mutfak/Masa")
            await broadcast_message(aktif_admin_websocketleri, notification, "Admin")
            await broadcast_message(aktif_kasa_websocketleri, notification, "Kasa")
            
            await update_table_status(updated_order["masa"], f"Sipariş {siparis_id} ödendi (by {current_user.kullanici_adi}, Yöntem: {updated_order['odeme_yontemi']})")
            
            logger.info(f"✅ Sipariş {siparis_id} ödeme ve stok düşme işlemleri başarıyla tamamlandı.")
            return {"message": f"Sipariş {siparis_id} ödendi ve stoklar güncellendi.", "data": updated_order}

    except HTTPException as http_exc: # pragma: no cover
        logger.warning(f"Kasa ödeme işlemi sırasında (Sipariş {siparis_id}) beklenen bir hata: {http_exc.detail}")
        raise http_exc
    except Exception as e: # pragma: no cover
        logger.error(f"❌ Kasa: Sipariş {siparis_id} ödendi olarak işaretlenirken veya stok düşülürken beklenmedik GENEL HATA: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Sipariş durumu güncellenirken veya stok düşülürken sunucuda bir hata oluştu.")

@app.post("/admin/receteler", response_model=MenuUrunRecetesi, status_code=status.HTTP_201_CREATED, tags=["Reçete Yönetimi"])
async def create_menu_urun_recetesi(
    recete_data: MenuUrunRecetesiCreate,
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"Admin '{current_user.kullanici_adi}' yeni menü ürünü reçetesi ekliyor: Menu ID {recete_data.menu_urun_id}")
    async with db.transaction():
        # 1. Ana reçete kaydını oluştur
        query_recete = """
            INSERT INTO menu_urun_receteleri (menu_urun_id, aciklama, porsiyon_birimi, porsiyon_miktari, guncellenme_tarihi)
            VALUES (:menu_urun_id, :aciklama, :porsiyon_birimi, :porsiyon_miktari, :guncellenme_tarihi)
            RETURNING id, menu_urun_id, aciklama, porsiyon_birimi, porsiyon_miktari, olusturulma_tarihi, guncellenme_tarihi;
        """
        now_ts = datetime.now(TR_TZ)
        try:
            created_recete_row = await db.fetch_one(query_recete, {
                "menu_urun_id": recete_data.menu_urun_id,
                "aciklama": recete_data.aciklama,
                "porsiyon_birimi": recete_data.porsiyon_birimi,
                "porsiyon_miktari": recete_data.porsiyon_miktari,
                "guncellenme_tarihi": now_ts
            })
            if not created_recete_row: # pragma: no cover
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Reçete oluşturulamadı.")
        except Exception as e_recete: # pragma: no cover
             if "unique constraint" in str(e_recete).lower() and "menu_urun_receteleri_menu_urun_id_key" in str(e_recete).lower():
                 raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Menü ürünü ID {recete_data.menu_urun_id} için zaten bir reçete mevcut.")
             logger.error(f"Reçete DB kaydı hatası: {e_recete}", exc_info=True)
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Veritabanı hatası (reçete): {str(e_recete)}")


        recete_id = created_recete_row["id"]
        created_bilesenler_db = []

        # 2. Reçete bileşenlerini kaydet
        for bilesen_data in recete_data.bilesenler:
            query_bilesen = """
                INSERT INTO recete_bilesenleri (recete_id, stok_kalemi_id, miktar, birim, guncellenme_tarihi)
                VALUES (:recete_id, :stok_kalemi_id, :miktar, :birim, :guncellenme_tarihi)
                RETURNING id, stok_kalemi_id, miktar, birim;
            """
            try:
                # Stok kalemi var mı kontrol et (opsiyonel, FK constraint'i zaten var ama öncesinde de kontrol iyi olabilir)
                stok_kalemi_check = await db.fetch_one("SELECT ad FROM stok_kalemleri WHERE id = :id", {"id": bilesen_data.stok_kalemi_id})
                if not stok_kalemi_check: # pragma: no cover
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"ID: {bilesen_data.stok_kalemi_id} ile stok kalemi bulunamadı.")

                bilesen_row = await db.fetch_one(query_bilesen, {
                    "recete_id": recete_id,
                    "stok_kalemi_id": bilesen_data.stok_kalemi_id,
                    "miktar": bilesen_data.miktar,
                    "birim": bilesen_data.birim,
                    "guncellenme_tarihi": now_ts
                })
                if not bilesen_row: # pragma: no cover
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Reçete bileşeni {bilesen_data.stok_kalemi_id} kaydedilemedi.")
                
                # stok_kalemi_ad'ı da ekleyerek response için hazırla
                bilesen_dict = dict(bilesen_row)
                bilesen_dict["stok_kalemi_ad"] = stok_kalemi_check["ad"]
                created_bilesenler_db.append(ReceteBileseni(**bilesen_dict))

            except Exception as e_bilesen: # pragma: no cover
                if "foreign key constraint" in str(e_bilesen).lower() and "stok_kalemleri" in str(e_bilesen).lower():
                     raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"ID: {bilesen_data.stok_kalemi_id} ile stok kalemi bulunamadı (FK hatası).")
                if "unique constraint" in str(e_bilesen).lower() and "recete_bilesenleri_recete_id_stok_kalemi_id_key" in str(e_bilesen).lower():
                    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Stok kalemi ID {bilesen_data.stok_kalemi_id} bu reçetede zaten mevcut.")
                logger.error(f"Reçete bileşeni DB kaydı hatası: {e_bilesen}", exc_info=True)
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Veritabanı hatası (bileşen): {str(e_bilesen)}")

        # menu_urun_ad'ı menu_db'den çek
        menu_urun_ad = "Bilinmeyen Ürün" # Default
        if menu_db.is_connected or await menu_db.connect():
            menu_urun_info = await menu_db.fetch_one("SELECT ad FROM menu WHERE id = :id", {"id": recete_data.menu_urun_id})
            if menu_urun_info:
                menu_urun_ad = menu_urun_info["ad"]
            # Eğer menü_db ayrı ise ve sürekli bağlı kalmıyorsa, burada disconnect edilebilir.
            # if menu_db != db: await menu_db.disconnect() # Opsiyonel

        final_recete_data = dict(created_recete_row)
        final_recete_data["bilesenler"] = created_bilesenler_db
        final_recete_data["menu_urun_ad"] = menu_urun_ad

        logger.info(f"Reçete ID {recete_id} başarıyla oluşturuldu.")
        # WebSocket ile admin paneline bildirim gönderilebilir.
        # await broadcast_message(aktif_admin_websocketleri, {"type": "recete_guncellendi", "data": {"action": "create", "id": recete_id}}, "Admin")
        return MenuUrunRecetesi(**final_recete_data)

@app.get("/admin/receteler", response_model=List[MenuUrunRecetesi], tags=["Reçete Yönetimi"])
async def list_menu_urun_receteleri_admin(
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"Admin '{current_user.kullanici_adi}' tüm menü ürün reçetelerini listeliyor.")
    # menu_urun_ad'ı da çekmek için JOIN veya subquery gerekebilir.
    # Eğer menu_db ve db farklı instance'lar ise, bu JOIN/subquery çalışmaz.
    # Bu durumda, önce ana reçeteler çekilir, sonra her biri için menu_db'den ürün adı alınır.
    
    query_receteler = """
        SELECT id, menu_urun_id, aciklama, porsiyon_birimi, porsiyon_miktari, olusturulma_tarihi, guncellenme_tarihi
        FROM menu_urun_receteleri
        ORDER BY id DESC;
    """
    receteler_raw = await db.fetch_all(query_receteler)
    
    response_list = []
    menu_item_names_cache = {} # menu_db'ye tekrar tekrar sorgu atmamak için basit bir cache

    if not menu_db.is_connected: # menu_db'ye sorgu atılacaksa bağlı olduğundan emin ol
        await menu_db.connect() # pragma: no cover

    for recete_row_data in receteler_raw:
        recete_dict = dict(recete_row_data)
        menu_urun_id = recete_dict["menu_urun_id"]

        # Menü ürün adını cache'den veya db'den al
        if menu_urun_id in menu_item_names_cache: # pragma: no cover
            recete_dict["menu_urun_ad"] = menu_item_names_cache[menu_urun_id]
        else:
            menu_item_info = await menu_db.fetch_one("SELECT ad FROM menu WHERE id = :id", {"id": menu_urun_id})
            if menu_item_info: # pragma: no cover
                menu_item_names_cache[menu_urun_id] = menu_item_info["ad"]
                recete_dict["menu_urun_ad"] = menu_item_info["ad"]
            else: # pragma: no cover
                recete_dict["menu_urun_ad"] = f"ID:{menu_urun_id} (Menüde Bulunamadı)"
                menu_item_names_cache[menu_urun_id] = recete_dict["menu_urun_ad"]


        # Bileşenleri çek
        bilesenler_query = """
            SELECT rb.id, rb.stok_kalemi_id, sk.ad as stok_kalemi_ad, rb.miktar, rb.birim
            FROM recete_bilesenleri rb
            JOIN stok_kalemleri sk ON rb.stok_kalemi_id = sk.id
            WHERE rb.recete_id = :recete_id
            ORDER BY sk.ad;
        """
        bilesenler_raw = await db.fetch_all(bilesenler_query, {"recete_id": recete_dict["id"]})
        recete_dict["bilesenler"] = [ReceteBileseni(**b) for b in bilesenler_raw]
        
        response_list.append(MenuUrunRecetesi(**recete_dict))

    # if menu_db != db and menu_db.is_connected: # Eğer menu_db ayrı ise ve sürekli bağlı kalmıyorsa
    # await menu_db.disconnect() # pragma: no cover

    return response_list

@app.get("/istatistik/satis-raporu", response_model=SatisRaporuResponse, tags=["İstatistikler"])
async def get_satis_raporu(
    baslangic_tarihi_str: str = Query(..., description="Başlangıç tarihi (YYYY-MM-DD formatında)"),
    bitis_tarihi_str: str = Query(..., description="Bitiş tarihi (YYYY-MM-DD formatında)"),
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(
        f"Admin '{current_user.kullanici_adi}' satış raporu istedi. Tarih Aralığı: {baslangic_tarihi_str} - {bitis_tarihi_str}"
    )
    try:
        try:
            # Tarih string'lerini date objelerine çevir
            # VeliDate alias'ınız varsa onu kullanın, yoksa direkt date
            baslangic_tarihi = datetime.strptime(baslangic_tarihi_str, "%Y-%m-%d").date()
            bitis_tarihi = datetime.strptime(bitis_tarihi_str, "%Y-%m-%d").date()
        except ValueError: # pragma: no cover
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Geçersiz tarih formatı. Lütfen YYYY-MM-DD formatını kullanın."
            )

        if baslangic_tarihi > bitis_tarihi: # pragma: no cover
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Başlangıç tarihi, bitiş tarihinden sonra olamaz."
            )

        # Veritabanı sorgusu için tarihleri datetime objelerine çevir (günün başlangıcı ve sonu)
        # TR_TZ'nin tanımlı olduğunu varsayıyorum
        gun_baslangic_dt = datetime.combine(baslangic_tarihi, datetime.min.time()).replace(tzinfo=TR_TZ)
        gun_bitis_dt = datetime.combine(bitis_tarihi, datetime.max.time()).replace(tzinfo=TR_TZ)

        query = """
            SELECT sepet FROM siparisler
            WHERE zaman >= :baslangic AND zaman <= :bitis AND durum = 'odendi' 
        """ # zaman <= :bitis kullandık çünkü gun_bitis_dt günün sonu

        odenen_siparisler = await db.fetch_all(query, {"baslangic": gun_baslangic_dt, "bitis": gun_bitis_dt})

        urun_satis_verileri: Dict[Tuple[str, str], Dict[str, Union[int, float]]] = {} # ((urun_adi, kategori_adi) -> {adet, gelir})
        kategori_satis_verileri: Dict[str, Dict[str, Union[int, float]]] = {} # (kategori_adi -> {adet, gelir})

        genel_toplam_gelir = 0.0
        genel_toplam_adet = 0

        for siparis in odenen_siparisler:
            try:
                sepet_items = json.loads(siparis["sepet"] or "[]")
                for item in sepet_items:
                    urun_adi = item.get("urun", "Bilinmeyen Ürün")
                    kategori_adi = item.get("kategori", "Kategorisiz")
                    adet = item.get("adet", 0)
                    fiyat = item.get("fiyat", 0.0)

                    if not isinstance(adet, (int, float)) or adet <= 0 or not isinstance(fiyat, (int, float)) or fiyat < 0:
                        logger.warning(f"Satış raporu: Geçersiz sepet öğesi verisi atlanıyor: {item}")
                        continue

                    urun_tutari = adet * fiyat

                    # Genel toplamları güncelle
                    genel_toplam_gelir += urun_tutari
                    genel_toplam_adet += adet

                    # Ürün bazlı verileri topla
                    urun_key = (urun_adi, kategori_adi)
                    if urun_key not in urun_satis_verileri:
                        urun_satis_verileri[urun_key] = {"adet": 0, "gelir": 0.0}
                    urun_satis_verileri[urun_key]["adet"] += adet
                    urun_satis_verileri[urun_key]["gelir"] += urun_tutari

                    # Kategori bazlı verileri topla
                    if kategori_adi not in kategori_satis_verileri:
                        kategori_satis_verileri[kategori_adi] = {"adet": 0, "gelir": 0.0}
                    kategori_satis_verileri[kategori_adi]["adet"] += adet
                    kategori_satis_verileri[kategori_adi]["gelir"] += urun_tutari

            except json.JSONDecodeError: # pragma: no cover
                logger.warning(f"Satış raporu: Sepet JSON parse hatası. Sipariş sepeti: {siparis['sepet']}")
                continue

        # Pydantic modellerine dönüştür
        urun_bazli_liste = [
            SatisRaporuUrunDetay(
                urun_adi=key[0], 
                kategori_adi=key[1], 
                satilan_adet=val["adet"], 
                toplam_gelir=round(val["gelir"], 2)
            ) for key, val in urun_satis_verileri.items()
        ]

        kategori_bazli_liste = [
            SatisRaporuKategoriDetay(
                kategori_adi=key, 
                satilan_adet=val["adet"], 
                toplam_gelir=round(val["gelir"], 2)
            ) for key, val in kategori_satis_verileri.items()
        ]

        # Daha iyi okunabilirlik için sıralama (isteğe bağlı)
        urun_bazli_liste.sort(key=lambda x: x.toplam_gelir, reverse=True)
        kategori_bazli_liste.sort(key=lambda x: x.toplam_gelir, reverse=True)

        return SatisRaporuResponse(
            baslangic_tarihi=baslangic_tarihi,
            bitis_tarihi=bitis_tarihi,
            urun_bazli_satislar=urun_bazli_liste,
            kategori_bazli_satislar=kategori_bazli_liste,
            genel_toplam_gelir=round(genel_toplam_gelir, 2),
            genel_toplam_adet=genel_toplam_adet
        )

    except HTTPException as http_err: # pragma: no cover
        raise http_err
    except Exception as e: # pragma: no cover
        logger.error(f"❌ Satış raporu alınırken beklenmedik bir hata oluştu: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Satış raporu oluşturulurken sunucuda bir sorun oluştu."
        )

@app.get("/admin/receteler/{recete_id}", response_model=MenuUrunRecetesi, tags=["Reçete Yönetimi"])
async def get_menu_urun_recetesi_admin(
    recete_id: int,
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"Admin '{current_user.kullanici_adi}' reçete ID {recete_id} detayını istiyor.")
    query_recete = """
        SELECT id, menu_urun_id, aciklama, porsiyon_birimi, porsiyon_miktari, olusturulma_tarihi, guncellenme_tarihi
        FROM menu_urun_receteleri
        WHERE id = :recete_id;
    """
    recete_row = await db.fetch_one(query_recete, {"recete_id": recete_id})
    if not recete_row: # pragma: no cover
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Reçete bulunamadı.")

    recete_dict = dict(recete_row)
    
    if not menu_db.is_connected: await menu_db.connect() # pragma: no cover
    menu_item_info = await menu_db.fetch_one("SELECT ad FROM menu WHERE id = :id", {"id": recete_dict["menu_urun_id"]})
    recete_dict["menu_urun_ad"] = menu_item_info["ad"] if menu_item_info else f"ID:{recete_dict['menu_urun_id']} (Bulunamadı)"
    # if menu_db != db and menu_db.is_connected: await menu_db.disconnect() # Opsiyonel

    bilesenler_query = """
        SELECT rb.id, rb.stok_kalemi_id, sk.ad as stok_kalemi_ad, rb.miktar, rb.birim
        FROM recete_bilesenleri rb
        JOIN stok_kalemleri sk ON rb.stok_kalemi_id = sk.id
        WHERE rb.recete_id = :recete_id ORDER BY sk.ad;
    """
    bilesenler_raw = await db.fetch_all(bilesenler_query, {"recete_id": recete_id})
    recete_dict["bilesenler"] = [ReceteBileseni(**b) for b in bilesenler_raw]
    
    return MenuUrunRecetesi(**recete_dict)


@app.put("/admin/receteler/{recete_id}", response_model=MenuUrunRecetesi, tags=["Reçete Yönetimi"])
async def update_menu_urun_recetesi(
    recete_id: int,
    recete_data: MenuUrunRecetesiCreate, # Create modeli PUT için de kullanılabilir, menu_urun_id frontend'den disabled gelir.
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"Admin '{current_user.kullanici_adi}' reçete ID {recete_id} güncelliyor.")
    
    async with db.transaction():
        # 1. Reçete var mı kontrol et (menu_urun_id'si ile birlikte)
        existing_recete = await db.fetch_one("SELECT id, menu_urun_id FROM menu_urun_receteleri WHERE id = :recete_id", {"recete_id": recete_id})
        if not existing_recete: # pragma: no cover
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Güncellenecek reçete bulunamadı.")
        
        # Frontend'den gelen menu_urun_id ile DB'deki menu_urun_id'nin aynı olduğunu doğrula (genelde değiştirilmez)
        if existing_recete["menu_urun_id"] != recete_data.menu_urun_id: # pragma: no cover
            # Bu durum normalde frontend tarafından engellenmeli.
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Reçetenin ait olduğu menü ürünü değiştirilemez.")

        # 2. Ana reçete bilgilerini güncelle
        now_ts = datetime.now(TR_TZ)
        query_update_recete = """
            UPDATE menu_urun_receteleri
            SET aciklama = :aciklama, porsiyon_birimi = :porsiyon_birimi, porsiyon_miktari = :porsiyon_miktari, guncellenme_tarihi = :guncellenme_tarihi
            WHERE id = :recete_id
            RETURNING id, menu_urun_id, aciklama, porsiyon_birimi, porsiyon_miktari, olusturulma_tarihi, guncellenme_tarihi;
        """
        updated_recete_row = await db.fetch_one(query_update_recete, {
            "recete_id": recete_id,
            "aciklama": recete_data.aciklama,
            "porsiyon_birimi": recete_data.porsiyon_birimi,
            "porsiyon_miktari": recete_data.porsiyon_miktari,
            "guncellenme_tarihi": now_ts
        })
        if not updated_recete_row: # pragma: no cover
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Reçete güncellenemedi.")


        # 3. Mevcut bileşenleri sil
        await db.execute("DELETE FROM recete_bilesenleri WHERE recete_id = :recete_id", {"recete_id": recete_id})

        # 4. Yeni bileşenleri ekle
        updated_bilesenler_db = []
        for bilesen_data in recete_data.bilesenler:
            stok_kalemi_check = await db.fetch_one("SELECT ad FROM stok_kalemleri WHERE id = :id", {"id": bilesen_data.stok_kalemi_id})
            if not stok_kalemi_check: # pragma: no cover
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Bileşen için ID: {bilesen_data.stok_kalemi_id} ile stok kalemi bulunamadı.")

            query_bilesen = """
                INSERT INTO recete_bilesenleri (recete_id, stok_kalemi_id, miktar, birim, guncellenme_tarihi)
                VALUES (:recete_id, :stok_kalemi_id, :miktar, :birim, :guncellenme_tarihi)
                RETURNING id, stok_kalemi_id, miktar, birim;
            """
            bilesen_row = await db.fetch_one(query_bilesen, {
                "recete_id": recete_id,
                "stok_kalemi_id": bilesen_data.stok_kalemi_id,
                "miktar": bilesen_data.miktar,
                "birim": bilesen_data.birim,
                "guncellenme_tarihi": now_ts
            })
            if not bilesen_row: # pragma: no cover
                 raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Reçete bileşeni {bilesen_data.stok_kalemi_id} güncellenirken eklenemedi.")
            
            bilesen_dict = dict(bilesen_row)
            bilesen_dict["stok_kalemi_ad"] = stok_kalemi_check["ad"]
            updated_bilesenler_db.append(ReceteBileseni(**bilesen_dict))

        menu_urun_ad = "Bilinmeyen Ürün"
        if menu_db.is_connected or await menu_db.connect(): # pragma: no cover
            menu_urun_info = await menu_db.fetch_one("SELECT ad FROM menu WHERE id = :id", {"id": updated_recete_row["menu_urun_id"]})
            if menu_urun_info: menu_urun_ad = menu_urun_info["ad"]
            # if menu_db != db: await menu_db.disconnect() # Opsiyonel
        
        final_recete_data = dict(updated_recete_row)
        final_recete_data["bilesenler"] = updated_bilesenler_db
        final_recete_data["menu_urun_ad"] = menu_urun_ad
        
        logger.info(f"Reçete ID {recete_id} başarıyla güncellendi.")
        # await broadcast_message(aktif_admin_websocketleri, {"type": "recete_guncellendi", "data": {"action": "update", "id": recete_id}}, "Admin")
        return MenuUrunRecetesi(**final_recete_data)

@app.delete("/admin/receteler/{recete_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Reçete Yönetimi"])
async def delete_menu_urun_recetesi(
    recete_id: int,
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"Admin '{current_user.kullanici_adi}' reçete ID {recete_id} siliyor.")
    async with db.transaction():
        # Reçete var mı kontrol et
        recete_check = await db.fetch_one("SELECT id FROM menu_urun_receteleri WHERE id = :recete_id", {"recete_id": recete_id})
        if not recete_check: # pragma: no cover
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Silinecek reçete bulunamadı.")
        
        # Bileşenler ON DELETE CASCADE ile otomatik silinecek. Sadece ana reçeteyi silmek yeterli.
        await db.execute("DELETE FROM menu_urun_receteleri WHERE id = :recete_id", {"recete_id": recete_id})
    
    logger.info(f"Reçete ID {recete_id} başarıyla silindi.")
    # await broadcast_message(aktif_admin_websocketleri, {"type": "recete_guncellendi", "data": {"action": "delete", "id": recete_id}}, "Admin")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/admin/menu-items-simple", response_model=List[MenuUrunuSimple], tags=["Reçete Yönetimi Yardımcı"])
async def list_menu_items_for_recipe_selection(
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"Admin '{current_user.kullanici_adi}' reçete seçimi için basit menü ürün listesini istiyor.")
    # Henüz reçetesi olmayan menü ürünlerini de listeleyebiliriz veya tümünü. Şimdilik tümünü listeleyelim.
    # Zaten reçetesi olan bir ürüne ikinci bir reçete eklenmesi UNIQUE constraint ile engellenecek.
    query = """
        SELECT m.id, m.ad, k.isim as kategori_ad
        FROM menu m
        JOIN kategoriler k ON m.kategori_id = k.id
        WHERE m.stok_durumu = 1 -- Sadece stokta olan ve aktif ürünler için reçete mantıklı olabilir
        ORDER BY k.isim, m.ad;
    """
    if not menu_db.is_connected: await menu_db.connect() # pragma: no cover
    menu_items_raw = await menu_db.fetch_all(query)
    
    return [MenuUrunuSimple(**row) for row in menu_items_raw]

@app.get("/admin/stock-items-simple", response_model=List[StokKalemiSimple], tags=["Reçete Yönetimi Yardımcı"])
async def list_stock_items_for_recipe_selection(
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"Admin '{current_user.kullanici_adi}' reçete seçimi için basit stok kalemi listesini istiyor.")
    query = "SELECT id, ad, birim FROM stok_kalemleri ORDER BY ad;"
    stock_items_raw = await db.fetch_all(query)
    return [StokKalemiSimple(**row) for row in stock_items_raw]

@app.get("/kasa/odemeler", tags=["Kasa İşlemleri"])
async def get_payable_orders_endpoint(
    durum: Optional[str] = Query(None, description=f"Sipariş durumu filtresi. Seçenekler: {', '.join([d.value for d in [Durum.HAZIR, Durum.BEKLIYOR, Durum.HAZIRLANIYOR]])}"),
    current_user: Kullanici = Depends(role_checker([KullaniciRol.ADMIN, KullaniciRol.KASIYER]))
):
    logger.info(f"💰 Kasa: Ödeme bekleyen siparişler listeleniyor (Kullanıcı: {current_user.kullanici_adi}, Filtre: {durum}).")
    try:
        base_query = "SELECT id, masa, istek, sepet, zaman, durum, odeme_yontemi FROM siparisler WHERE "
        values = {}
        valid_statuses_for_payment = [Durum.HAZIR.value, Durum.BEKLIYOR.value, Durum.HAZIRLANIYOR.value]

        if durum:
            if durum not in valid_statuses_for_payment: # pragma: no cover
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Geçersiz durum. Kullanılabilecekler: {', '.join(valid_statuses_for_payment)}")
            query_str = base_query + "durum = :durum ORDER BY zaman ASC"
            values["durum"] = durum
        else:
            # PostgreSQL'de IN operatörü için tuple veya liste kullanılabilir. Databases kütüphanesi bunu handle eder.
            query_str = base_query + "durum = ANY(:statuses_list) ORDER BY zaman ASC" # PostgreSQL ANY kullanımı
            values["statuses_list"] = valid_statuses_for_payment # Liste olarak gönder

        orders_raw = await db.fetch_all(query=query_str, values=values)
        orders_data = []
        for row in orders_raw:
            order_dict = dict(row)
            order_dict["sepet"] = json.loads(order_dict.get('sepet','[]'))
            if isinstance(order_dict.get('zaman'), datetime):
                 order_dict['zaman'] = order_dict['zaman'].isoformat()
            orders_data.append(order_dict)
        return {"orders": orders_data}
    except Exception as e: # pragma: no cover
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
        # PostgreSQL IN operatörü için tuple
        query_str = "SELECT id, masa, istek, sepet, zaman, durum, yanit, odeme_yontemi FROM siparisler WHERE masa = :masa_id AND durum = ANY(:statuses) ORDER BY zaman ASC"
        values = {"masa_id": masa_id, "statuses": active_payable_statuses} # Liste olarak gönder
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
    except Exception as e: # pragma: no cover
        logger.error(f"❌ Kasa: Masa {masa_id} hesabı alınırken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Masa hesabı alınırken bir hata oluştu.")

@app.post("/admin/kullanicilar", response_model=Kullanici, status_code=status.HTTP_201_CREATED, tags=["Kullanıcı Yönetimi"])
async def create_new_user(
    user_data: KullaniciCreate,
    current_admin: Kullanici = Depends(role_checker([KullaniciRol.ADMIN]))
):
    logger.info(f"Admin '{current_admin.kullanici_adi}' yeni kullanıcı oluşturuyor: {user_data.kullanici_adi}, Rol: {user_data.rol}")
    existing_user = await get_user_from_db(user_data.kullanici_adi)
    if existing_user: # pragma: no cover
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
        if not created_user_row: # pragma: no cover
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Kullanıcı oluşturulurken bir sorun oluştu (DB).")
        logger.info(f"Kullanıcı '{created_user_row['kullanici_adi']}' başarıyla oluşturuldu (ID: {created_user_row['id']}).")
        return Kullanici(**created_user_row)
    except Exception as e: # pragma: no cover
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
    logger.info(f"Admin '{current_admin.kullanici_adi}', kullanıcı ID {user_id} için güncelleme yapıyor: {user_update_data.model_dump_json(exclude_unset=True)}") # exclude_unset=True daha uygun
    target_user_row = await db.fetch_one("SELECT id, kullanici_adi, rol, aktif_mi FROM kullanicilar WHERE id = :user_id", {"user_id": user_id})
    if not target_user_row: # pragma: no cover
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Güncellenecek kullanıcı bulunamadı.")
    target_user = dict(target_user_row) # Kullanıcı verilerini dict olarak al
    update_fields = {} # Güncellenecek alanları tutacak dict

    # Kullanıcı adı güncelleniyorsa ve farklıysa
    if user_update_data.kullanici_adi is not None and user_update_data.kullanici_adi != target_user["kullanici_adi"]: # pragma: no cover
        # Yeni kullanıcı adının başkası tarafından kullanılıp kullanılmadığını kontrol et
        existing_user_with_new_name = await db.fetch_one(
            "SELECT id FROM kullanicilar WHERE kullanici_adi = :k_adi AND id != :u_id",
            {"k_adi": user_update_data.kullanici_adi, "u_id": user_id}
        )
        if existing_user_with_new_name:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Bu kullanıcı adı zaten başka bir kullanıcı tarafından kullanılıyor.")
        update_fields["kullanici_adi"] = user_update_data.kullanici_adi

    if user_update_data.rol is not None: update_fields["rol"] = user_update_data.rol.value
    if user_update_data.aktif_mi is not None: update_fields["aktif_mi"] = user_update_data.aktif_mi
    if user_update_data.sifre is not None: update_fields["sifre_hash"] = get_password_hash(user_update_data.sifre)

    if not update_fields: # pragma: no cover
        logger.info(f"Kullanıcı ID {user_id} için güncellenecek bir alan belirtilmedi.")
        return Kullanici(**target_user) # Mevcut kullanıcı bilgilerini döndür

    set_clause_parts = [f"{key} = :{key}" for key in update_fields.keys()]
    set_clause = ", ".join(set_clause_parts)
    query = f"UPDATE kullanicilar SET {set_clause} WHERE id = :user_id_param RETURNING id, kullanici_adi, rol, aktif_mi"
    values = {**update_fields, "user_id_param": user_id} # :id ile karışmaması için

    try:
        updated_user_row = await db.fetch_one(query, values)
        if not updated_user_row: # pragma: no cover
            # Bu durumun normalde olmaması gerekir eğer ilk fetch_one başarılıysa
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Kullanıcı güncellenirken bulunamadı (UPDATE sonrası).")
        logger.info(f"Kullanıcı ID {user_id} başarıyla güncellendi. Yeni değerler: {dict(updated_user_row)}")
        return Kullanici(**updated_user_row)
    except Exception as e: # pragma: no cover
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
    if current_admin.id == user_id: # pragma: no cover
        logger.warning(f"Admin '{current_admin.kullanici_adi}' kendini silmeye çalıştı.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Admin kendini silemez.")

    user_to_delete = await db.fetch_one("SELECT id FROM kullanicilar WHERE id = :user_id", {"user_id": user_id})
    if not user_to_delete: # pragma: no cover
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Silinecek kullanıcı bulunamadı.")

    try:
        await db.execute("DELETE FROM kullanicilar WHERE id = :user_id", {"user_id": user_id})
        logger.info(f"Kullanıcı ID {user_id} başarıyla silindi.")
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except Exception as e: # pragma: no cover
        logger.error(f"Kullanıcı ID {user_id} silinirken DB hatası: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Kullanıcı silinirken veritabanı hatası: {str(e)}")

if __name__ == "__main__": # pragma: no cover
    import uvicorn
    host_ip = os.getenv("HOST", "127.0.0.1")
    port_num = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 FastAPI uygulaması {host_ip}:{port_num} adresinde başlatılıyor (yerel geliştirme modu)...")
    uvicorn.run("main:app", host=host_ip, port=port_num, reload=True, log_config=LOGGING_CONFIG)