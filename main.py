from fastapi import (
    FastAPI, Request, Path, Body, Query, HTTPException, status, Depends, WebSocket, WebSocketDisconnect, Response
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.middleware.sessions import SessionMiddleware
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Set
from async_lru import alru_cache # YENİ
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
    version="1.2.5", # Versiyonu güncelleyelim
    description="Fıstık Kafe için sipariş backend servisi."
)
security = HTTPBasic()

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

    message_json = json.dumps(message, ensure_ascii=False) # ensure_ascii=False eklendi
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
        if e.code == 1012: # Sunucu yeniden başlatılıyor veya benzeri bir durum
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
            endpoint_name = request.url.path # Daha genel bir fallback
        await update_table_status(str(masa_id), f"{request.method} {endpoint_name}")
    try:
        response = await call_next(request)
        return response
    except HTTPException as http_exc: # Bilerek fırlatılan HTTP hatalarını tekrar fırlat
        raise http_exc
    except Exception as e: # Diğer beklenmedik hataları logla ve genel bir 500 döndür
        logger.exception(f"❌ HTTP Middleware genel hata ({request.url.path}): {e}")
        return Response("Sunucuda bir hata oluştu.", status_code=500, media_type="text/plain")


# Endpoint'ler
@app.get("/ping")
async def ping_endpoint():
    logger.info("📢 /ping endpoint'ine istek geldi!")
    return {"message": "Neso backend pong! Service is running."}

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

@app.get("/aktif-masalar", dependencies=[Depends(check_admin)])
async def get_active_tables_endpoint():
    """
    Şu anda açık olan mutfak/masa asistanı WS bağlantılarının sayısını döner.
    """
    try:
        return {"count": len(aktif_mutfak_websocketleri)}
    except Exception as e:
        logger.error(f"❌ Aktif masalar WS bağlantı sayısı alınamadı: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="WS bağlantı sayısı alınamadı."
        )


# Pydantic Modelleri
class Durum(str, Enum):
    BEKLIYOR = "bekliyor"
    HAZIRLANIYOR = "hazirlaniyor"
    HAZIR = "hazir"
    IPTAL = "iptal"

class SepetItem(BaseModel):
    urun: str = Field(..., min_length=1, description="Sipariş edilen ürünün adı.")
    adet: int = Field(..., gt=0, description="Sipariş edilen ürünün adedi.")
    fiyat: float = Field(..., ge=0, description="Ürünün birim fiyatı.") # ge=0 fiyat 0 olabilir
    kategori: Optional[str] = Field(None, description="Ürünün kategorisi (isteğe bağlı).")

class SiparisEkleData(BaseModel):
    masa: str = Field(..., min_length=1, description="Siparişin verildiği masa numarası/adı.")
    sepet: List[SepetItem] = Field(..., min_items=1, description="Sipariş edilen ürünlerin listesi.")
    istek: Optional[str] = Field(None, description="Müşterinin özel isteği.") # Frontend'den gelen kullanıcı metni
    yanit: Optional[str] = Field(None, description="AI tarafından üretilen yanıt (müşteri isteğine karşılık).") # Frontend'den gelen AI JSON yanıtı

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
@app.patch("/siparis/{id}", dependencies=[Depends(check_admin)])
async def patch_order_endpoint(
    id: int = Path(..., description="Güncellenecek siparişin ID'si"),
    data: SiparisGuncelleData = Body(...)
):
    logger.info(f"🔧 PATCH /siparis/{id} ile durum güncelleme isteği: {data.durum}")
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
        order["sepet"] = json.loads(order.get("sepet", "[]"))

        notif = {
            "type": "durum",
            "data": {
                "id": order["id"],
                "masa": order["masa"],
                "durum": order["durum"],
                "sepet": order["sepet"],
                "istek": order["istek"],
                "zaman": datetime.now().isoformat()
            }
        }
        await broadcast_message(aktif_mutfak_websocketleri, notif, "Mutfak/Masa")
        await broadcast_message(aktif_admin_websocketleri, notif, "Admin")
        await update_table_status(order["masa"], f"Sipariş {id} durumu güncellendi -> {order['durum']}")
        return {"message": f"Sipariş {id} güncellendi.", "data": order}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ PATCH /siparis/{id} hatası: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Sipariş durumu güncellenirken hata oluştu.")

##
# 2) Sipariş İptali (DELETE /siparis/{id})
##
@app.delete("/siparis/{id}", dependencies=[Depends(check_admin)])
async def delete_order_endpoint(
    id: int = Path(..., description="İptal edilecek siparişin ID'si")
):
    logger.info(f"🗑 DELETE /siparis/{id} ile iptal isteği")
    row = await db.fetch_one("SELECT zaman, masa FROM siparisler WHERE id = :id", {"id": id})
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sipariş bulunamadı.")
    olusturma_zamani = datetime.strptime(row["zaman"], "%Y-%m-%d %H:%M:%S")
    if datetime.now() - olusturma_zamani > timedelta(minutes=1):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bu sipariş 1 dakikayı geçtiği için iptal edilemez."
        )
    try:
        async with db.transaction():
            await db.execute("UPDATE siparisler SET durum = 'iptal' WHERE id = :id", {"id": id})
        notif = {
            "type": "durum",
            "data": {
                "id": id,
                "masa": row["masa"],
                "durum": "iptal",
                "zaman": datetime.now().isoformat()
            }
        }
        await broadcast_message(aktif_mutfak_websocketleri, notif, "Mutfak/Masa")
        await broadcast_message(aktif_admin_websocketleri, notif, "Admin")
        await update_table_status(row["masa"], f"Sipariş {id} iptal edildi")
        return {"message": f"Sipariş {id} iptal edildi."}
    except Exception as e:
        logger.error(f"❌ DELETE /siparis/{id} hatası: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Sipariş iptal edilirken hata oluştu.")

@app.post("/siparis-ekle", status_code=status.HTTP_201_CREATED)
async def add_order_endpoint(data: SiparisEkleData):
    masa = data.masa
    sepet = data.sepet # Bu, frontend'in AI JSON'ından parse ettiği ve backend'e gönderdiği sepet listesi
    istek = data.istek # Müşterinin ilk ham isteği
    yanit = data.yanit # AI'ın ürettiği JSON string veya konuşma metni
    zaman_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"📥 Yeni sipariş isteği alındı: Masa {masa}, {len(sepet)} çeşit ürün. AI Ham Yanıtı (DB'ye yazılacak): {yanit[:200]}...")

    cached_price_dict = await get_menu_price_dict()
    cached_stock_dict = await get_menu_stock_dict() # Bu zaten stokta olanları döndürmeli
    logger.info(f"/siparis-ekle: get_menu_stock_dict çağrıldı. Örnek: {list(cached_stock_dict.items())[:3]}")


    processed_sepet = []
    for item in sepet: # Bu 'sepet', frontend'in AI JSON'ından parse ettiği SepetItem listesi olmalı
        urun_adi_lower = item.urun.lower().strip()

        # Stok kontrolünü cached_stock_dict üzerinden yap (1: stokta, 0: stokta yok)
        stok_kontrol_degeri = cached_stock_dict.get(urun_adi_lower)
        if stok_kontrol_degeri is None or stok_kontrol_degeri == 0:
            logger.warning(f"⚠️ Stokta olmayan ürün sipariş edilmeye çalışıldı: '{item.urun}' (Masa: {masa}). Aranan: '{urun_adi_lower}'. Bulunan Stok: {stok_kontrol_degeri}. Stok Dict (ilk 5): {list(cached_stock_dict.items())[:5]}")
            # Normalde AI prompt'u zaten stokta olmayan ürün için JSON üretmemeli. Bu ek bir kontrol.
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"'{item.urun}' adlı ürün şu anda stokta bulunmamaktadır.")

        item_dict = item.model_dump()
        # Fiyatı cache'den al, eğer frontend'den gelen fiyat farklıysa logla ama cache'dekini kullan
        cached_fiyat = cached_price_dict.get(urun_adi_lower, item.fiyat)
        if cached_fiyat != item.fiyat:
            logger.warning(f"Fiyat uyuşmazlığı: Ürün '{item.urun}', Frontend Fiyatı: {item.fiyat}, Cache Fiyatı: {cached_fiyat}. Cache fiyatı kullanılacak.")
        item_dict['fiyat'] = cached_fiyat
        if item_dict['fiyat'] == 0 and item.fiyat == 0 : # Fiyat 0 ise uyarı ver
            logger.warning(f"⚠️ '{item.urun}' için fiyat bilgisi 0 olarak ayarlandı. Lütfen menüyü kontrol edin.")
        processed_sepet.append(item_dict)

    if not processed_sepet:
        logger.warning(f"⚠️ Sipariş verilemedi, sepetteki tüm ürünler stok dışı veya işlenemedi. (Masa: {masa})")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Sepetinizdeki ürünlerin hiçbiri şu anda mevcut değil veya işlenemedi.")

    istek_ozet = ", ".join([f"{p_item['adet']}x {p_item['urun']}" for p_item in processed_sepet])
    try:
        async with db.transaction():
            siparis_id = await db.fetch_val("""
                INSERT INTO siparisler (masa, istek, yanit, sepet, zaman, durum)
                VALUES (:masa, :istek, :yanit, :sepet, :zaman, 'bekliyor')
                RETURNING id
            """, {
                "masa": masa,
                "istek": istek or istek_ozet, # Müşterinin ham isteği varsa o, yoksa özeti
                "yanit": yanit, # AI'dan gelen ham JSON yanıtı veya konuşma metni
                "sepet": json.dumps(processed_sepet, ensure_ascii=False), # İşlenmiş ve doğrulanmış sepet
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
        raise http_exc # Bilerek fırlatılanları tekrar fırlat
    except Exception as e:
        logger.error(f"❌ Sipariş ekleme sırasında beklenmedik hata (Masa: {masa}): {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Siparişiniz işlenirken bir sunucu hatası oluştu.")

@app.post("/siparis-guncelle")
async def update_order_status_endpoint(data: SiparisGuncelleData, auth: bool = Depends(check_admin)):
    logger.info(f"🔄 Sipariş durum güncelleme isteği: ID {data.id or 'Son'}, Masa {data.masa}, Durum {data.durum}")
    try:
        async with db.transaction():
            if data.id: # Belirli bir sipariş ID'si varsa
                query = "UPDATE siparisler SET durum = :durum WHERE id = :id RETURNING id, masa, durum, sepet, istek, zaman"
                values = {"durum": data.durum.value, "id": data.id}
            else: # Sipariş ID'si yoksa, masanın son aktif siparişini güncelle
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
                        "sepet": updated_order_dict.get("sepet"), # Parse edilmiş sepet
                        "istek": updated_order_dict.get("istek"),
                        "zaman": datetime.now().isoformat() # Güncelleme zamanı
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
                order_dict['sepet'] = json.loads(order_dict.get('sepet') or '[]') # Sepet JSON string'i parse et
            except json.JSONDecodeError:
                order_dict['sepet'] = [] # Hata durumunda boş liste
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
        async with db.transaction(): # transactions.py yerine doğrudan db objesi üzerinden
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
        raise # Bu hatayı tekrar fırlat, uygulama başlamasın

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
                    stok_durumu INTEGER DEFAULT 1, -- 1: Stokta, 0: Stokta Yok
                    FOREIGN KEY (kategori_id) REFERENCES kategoriler(id) ON DELETE CASCADE,
                    UNIQUE(ad, kategori_id) -- Aynı kategoride aynı ürün adı olamaz
                )""")
            # İndeksler
            await menu_db.execute("CREATE INDEX IF NOT EXISTS idx_menu_kategori_id ON menu(kategori_id)")
            await menu_db.execute("CREATE INDEX IF NOT EXISTS idx_menu_ad ON menu(ad)") # ad sütununa da indeks
        logger.info(f"✅ Menü veritabanı ({MENU_DB_PATH}) başarıyla doğrulandı/oluşturuldu.")
    except Exception as e:
        logger.critical(f"❌ Menü veritabanı başlatılırken kritik hata: {e}", exc_info=True)
        raise

async def init_databases():
    await init_db()
    await init_menu_db()

# Menü Yönetimi
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
        """ # Sadece stokta olanları (stok_durumu = 1) al
        urunler_raw = await menu_db.fetch_all(query)
        logger.info(f">>> get_menu_for_prompt_cached: Veritabanından (stok_durumu=1 olan) Çekilen Ham Menü Verisi (Toplam {len(urunler_raw)} ürün). Örnek (ilk 3): {str(urunler_raw[:3]).encode('utf-8', 'ignore').decode('utf-8', 'ignore')}")

        if not urunler_raw:
            logger.warning(">>> get_menu_for_prompt_cached: Menü prompt için stokta olan HİÇ ÜRÜN BULUNAMADI (sorgu boş döndü).")
            return "Üzgünüz, şu anda menümüzde aktif ürün bulunmamaktadır."

        kategorili_menu: Dict[str, List[str]] = {}
        for row in urunler_raw: # row artık Record objesi
            try:
                # Record objesinden sütunlara erişim
                kategori_ismi = row['kategori_isim'] # veya row.get('kategori_isim')
                urun_adi = row['urun_ad']         # veya row.get('urun_ad')

                if kategori_ismi and urun_adi: # İkisi de doluysa
                    kategorili_menu.setdefault(kategori_ismi, []).append(urun_adi)
                else:
                    logger.warning(f"get_menu_for_prompt_cached: Satırda eksik 'kategori_isim' veya 'urun_ad' bulundu: {dict(row) if hasattr(row, '_mapping') else str(row)}")
            except KeyError as ke: # Sütun adı hatası olursa
                logger.error(f"get_menu_for_prompt_cached: Satır işlenirken KeyError: {ke} - Satır: {dict(row) if hasattr(row, '_mapping') else str(row)}", exc_info=False) # exc_info=False daha kısa log için
            except Exception as e_row: # Diğer beklenmedik hatalar
                logger.error(f"get_menu_for_prompt_cached: Satır işlenirken beklenmedik hata: {e_row} - Satır: {dict(row) if hasattr(row, '_mapping') else str(row)}", exc_info=True)


        if not kategorili_menu: # Eğer döngü sonrası hala boşsa (örn. tüm satırlar hatalıysa)
            logger.warning(">>> get_menu_for_prompt_cached: Kategorili menü oluşturulamadı (urunler_raw dolu olmasına rağmen).")
            return "Üzgünüz, menü bilgisi şu anda düzgün bir şekilde formatlanamıyor."

        menu_aciklama_list = [] # Listeyi burada tanımla
        for kategori, urun_listesi in kategorili_menu.items():
            if urun_listesi: # Kategori altında ürün varsa ekle
                menu_aciklama_list.append(f"- {kategori}: {', '.join(urun_listesi)}")

        if not menu_aciklama_list: # Eğer hiçbir kategoriye ürün eklenemediyse
            logger.warning(">>> get_menu_for_prompt_cached: menu_aciklama_list oluşturulduktan sonra boş kaldı (kategorili_menu dolu olabilir ama listeler boş olabilir).")
            return "Üzgünüz, menüde listelenecek ürün bulunamadı."

        menu_aciklama = "\n".join(menu_aciklama_list)
        logger.info(f"Menü prompt için başarıyla oluşturuldu ({len(kategorili_menu)} kategori). Oluşturulan Menü Metni:\n{menu_aciklama}") # Logda menüyü göster
        return menu_aciklama
    except Exception as e:
        logger.error(f"❌ Menü prompt oluşturma hatası (get_menu_for_prompt_cached GENEL HATA): {e}", exc_info=True)
        return "Teknik bir sorun nedeniyle menü bilgisine şu anda ulaşılamıyor."

@alru_cache(maxsize=1)
async def get_menu_price_dict() -> Dict[str, float]:
    logger.info(">>> get_menu_price_dict ÇAĞRILIYOR...")
    try:
        if not menu_db.is_connected: await menu_db.connect()
        prices_raw = await menu_db.fetch_all("SELECT ad, fiyat FROM menu") # Tüm ürünlerin fiyatları
        price_dict = {row['ad'].lower().strip(): float(row['fiyat']) for row in prices_raw}
        logger.info(f"Fiyat sözlüğü {len(price_dict)} ürün için oluşturuldu/alındı. Örnek: {list(price_dict.items())[:3]}")
        return price_dict
    except Exception as e:
        logger.error(f"❌ Fiyat sözlüğü oluşturma/alma hatası: {e}", exc_info=True)
        return {}

@alru_cache(maxsize=1)
async def get_menu_stock_dict() -> Dict[str, int]:
    logger.info(">>> get_menu_stock_dict ÇAĞRILIYOR...")
    try:
        if not menu_db.is_connected:
            logger.info(">>> get_menu_stock_dict: menu_db BAĞLI DEĞİL, bağlanıyor...")
            await menu_db.connect()

        stocks_raw = await menu_db.fetch_all("SELECT ad, stok_durumu FROM menu") # Tüm ürünlerin stok durumları
        logger.info(f">>> get_menu_stock_dict: Veritabanından Çekilen Ham Stok Verisi (Toplam {len(stocks_raw)} ürün). Örnek (ilk 3): {str(stocks_raw[:3]).encode('utf-8', 'ignore').decode('utf-8', 'ignore')}")

        if not stocks_raw:
            logger.warning(">>> get_menu_stock_dict: Stok bilgisi için veritabanından HİÇ ürün çekilemedi!")
            return {}

        stock_dict = {}
        processed_count = 0
        for row in stocks_raw:
            try:
                urun_adi = str(row['ad']).lower().strip() # Küçük harf ve boşluksuz
                stok = int(row['stok_durumu']) # 1 veya 0 olmalı
                stock_dict[urun_adi] = stok
                processed_count += 1
            except Exception as e_loop:
                logger.error(f"Stok sözlüğü oluştururken satır işleme hatası: {e_loop} - Satır: {dict(row) if hasattr(row, '_mapping') else str(row)}", exc_info=True)

        logger.info(f">>> get_menu_stock_dict: Başarıyla işlenen ürün sayısı: {processed_count}")
        logger.info(f">>> get_menu_stock_dict: Oluşturulan stock_dict ({len(stock_dict)} öğe). Örnek (ilk 3): {list(stock_dict.items())[:3]}")
        return stock_dict
    except Exception as e_main:
        logger.error(f"❌ Stok sözlüğü oluşturma/alma sırasında genel hata: {e_main}", exc_info=True)
        return {}

# Geliştirilmiş SISTEM_MESAJI_ICERIK_TEMPLATE
SISTEM_MESAJI_ICERIK_TEMPLATE = (
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
    "  \"sepet\": [\n"
    "    {{\n"
    "      \"urun\": \"MENÜDEKİ TAM ÜRÜN ADI\",\n"
    "      \"adet\": ADET_SAYISI,\n"
    "      \"fiyat\": BIRIM_FIYAT,\n"
    "      \"kategori\": \"KATEGORI_ADI\"\n"
    "    }}\n"
    "  ],\n"
    "  \"toplam_tutar\": TOPLAM_TUTAR,\n"
    "  \"musteri_notu\": \"EK ÖZELLİKLER (sade, şekerli, vb.) veya ''\",\n"
    "  \"konusma_metni\": \"Kısa, nazik onay mesajı (aynı dilde).\"\n"
    "}}\n"
)

SYSTEM_PROMPT: Optional[Dict[str, str]] = None # Global değişken olarak tanımla

async def update_system_prompt():
    global SYSTEM_PROMPT
    logger.info("🔄 Sistem mesajı (menü bilgisi) güncelleniyor...")
    menu_data_for_prompt = "Menü bilgisi geçici olarak yüklenemedi." # Fallback değeri
    try:
        # Cache'leri temizle
        if hasattr(get_menu_for_prompt_cached, 'cache_clear'): get_menu_for_prompt_cached.cache_clear()
        if hasattr(get_menu_price_dict, 'cache_clear'): get_menu_price_dict.cache_clear()
        if hasattr(get_menu_stock_dict, 'cache_clear'): get_menu_stock_dict.cache_clear()
        logger.info("İlgili menü cache'leri temizlendi (update_system_prompt).")

        menu_data_for_prompt = await get_menu_for_prompt_cached() # Yenilenmiş menüyü al
        logger.info(f"update_system_prompt: get_menu_for_prompt_cached'den dönen menu_data_for_prompt (ilk 200kr): {str(menu_data_for_prompt)[:200]}")

        current_system_content = SISTEM_MESAJI_ICERIK_TEMPLATE.format(menu_prompt_data=menu_data_for_prompt)
        SYSTEM_PROMPT = {"role": "system", "content": current_system_content}
        logger.info(f"✅ Sistem mesajı başarıyla güncellendi. SYSTEM_PROMPT içeriği (ilk 400 karakter): {str(SYSTEM_PROMPT)[:400]}")

    except Exception as e: # Herhangi bir hata durumunda
        logger.error(f"❌ Sistem mesajı güncellenirken BEKLENMEDİK BİR HATA oluştu: {e}", exc_info=True)
        # Hata olsa bile, en azından bir önceki (varsa) veya fallback menü ile devam et
        if SYSTEM_PROMPT is None: # Eğer daha önce hiç set edilmemişse fallback kullan
            current_system_content = SISTEM_MESAJI_ICERIK_TEMPLATE.format(menu_prompt_data=menu_data_for_prompt) # menu_data_for_prompt'un son değeriyle
            SYSTEM_PROMPT = {"role": "system", "content": current_system_content}
            logger.warning(f"Fallback sistem mesajı (BEKLENMEDİK HATA sonrası update_system_prompt içinde) kullanılıyor: {str(SYSTEM_PROMPT)[:300]}")

@app.get("/admin/clear-menu-caches", dependencies=[Depends(check_admin)])
async def clear_all_caches_endpoint():
    logger.info("Manuel cache temizleme isteği alındı (/admin/clear-menu-caches)")
    # Cache temizleme fonksiyonları zaten update_system_prompt içinde çağrılıyor.
    await update_system_prompt() # Bu fonksiyon cache'leri temizleyip prompt'u güncelleyecek
    return {"message": "Menü, fiyat ve stok cache'leri başarıyla temizlendi. Sistem promptu güncellendi."}

@app.get("/menu")
async def get_full_menu_endpoint(): # Admin yetkisi olmadan da erişilebilir olmalı
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
                "urunler": [dict(urun) for urun in urunler_raw] # stok_durumu da dönsün
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
            # Kategori var mı kontrol et, yoksa ekle (COLLATE NOCASE sayesinde büyük/küçük harf duyarsız)
            await menu_db.execute("INSERT OR IGNORE INTO kategoriler (isim) VALUES (:isim)", {"isim": item_data.kategori})
            category_id_row = await menu_db.fetch_one("SELECT id FROM kategoriler WHERE isim = :isim", {"isim": item_data.kategori})
            if not category_id_row: # Teorik olarak olmamalı ama kontrol edelim
                logger.error(f"Menü ekleme: Kategori '{item_data.kategori}' oluşturulamadı veya bulunamadı.")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Kategori oluşturulurken bir sorun oluştu.")
            category_id = category_id_row['id']

            try:
                item_id = await menu_db.fetch_val("""
                    INSERT INTO menu (ad, fiyat, kategori_id, stok_durumu)
                    VALUES (:ad, :fiyat, :kategori_id, 1) -- Yeni eklenen ürün varsayılan olarak stokta (1)
                    RETURNING id
                """, {"ad": item_data.ad, "fiyat": item_data.fiyat, "kategori_id": category_id})
            except sqlite3.IntegrityError as ie: # databases kütüphanesi bu hatayı farklı sarabilir, genel Exception'a da düşebilir
                 # Genellikle databases.exceptions.IntegrityError olarak gelir
                if "UNIQUE constraint failed" in str(ie): # Hata mesajını kontrol et
                    logger.warning(f"Menü ekleme başarısız: '{item_data.ad}' adlı ürün '{item_data.kategori}' kategorisinde zaten mevcut.")
                    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"'{item_data.ad}' adlı ürün bu kategoride zaten mevcut.")
                raise # Başka bir IntegrityError ise tekrar fırlat
        await update_system_prompt() # Menü değişti, prompt'u güncelle
        logger.info(f"✅ '{item_data.ad}' menüye başarıyla eklendi (ID: {item_id}). Sistem mesajı güncellendi.")
        return {"mesaj": f"'{item_data.ad}' ürünü menüye başarıyla eklendi.", "itemId": item_id}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e: # IntegrityError'u da burada yakalayabiliriz
        if "UNIQUE constraint failed" in str(e):
             logger.warning(f"Menü ekleme başarısız (genel exception): '{item_data.ad}' adlı ürün '{item_data.kategori}' kategorisinde zaten mevcut.")
             raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"'{item_data.ad}' adlı ürün bu kategoride zaten mevcut.")
        logger.error(f"❌ Menüye ürün eklenirken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Menüye ürün eklenirken bir sunucu hatası oluştu.")

@app.delete("/menu/sil")
async def delete_menu_item_endpoint(urun_adi: str = Query(..., min_length=1, description="Silinecek ürünün tam adı."), auth: bool = Depends(check_admin)):
    logger.info(f"🗑️ Menüden ürün silme isteği: {urun_adi}")
    try:
        async with menu_db.transaction():
            # Ürünün varlığını kontrol et (COLLATE NOCASE sayesinde büyük/küçük harf duyarsız)
            item_to_delete = await menu_db.fetch_one("SELECT id FROM menu WHERE ad = :ad", {"ad": urun_adi})
            if not item_to_delete:
                logger.warning(f"Silinecek ürün bulunamadı: '{urun_adi}'")
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{urun_adi}' adlı ürün menüde bulunamadı.")

            rows_affected_record = await menu_db.execute("DELETE FROM menu WHERE ad = :ad", {"ad": urun_adi})
            # execute() SQLite için etkilenen satır sayısını döndürmeyebilir, bu yüzden fetch_val kullanmıyoruz.
            # Silme işlemi başarılıysa ve item_to_delete varsa devam et.

        # Eğer silme başarılıysa (yani hata fırlatılmadıysa)
        await update_system_prompt() # Menü değişti, prompt'u güncelle
        logger.info(f"✅ '{urun_adi}' menüden başarıyla silindi. Sistem mesajı güncellendi.")
        return {"mesaj": f"'{urun_adi}' ürünü menüden başarıyla silindi."}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"❌ Menüden ürün silinirken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Menüden ürün silinirken bir sunucu hatası oluştu.")


# AI Yanıt
@app.post("/yanitla")
async def handle_message_endpoint(request: Request, data: dict = Body(...)): # Gelen data Pydantic modeli değil, dict
    user_message = data.get("text", "").strip()
    table_id = data.get("masa", "bilinmiyor") # Masa bilgisi de gelebilir
    session_id = request.session.get("session_id")

    if not session_id:
        session_id = secrets.token_hex(16)
        request.session["session_id"] = session_id
        request.session["chat_history"] = [] # Yeni session için boş sohbet geçmişi
        logger.info(f"Yeni session başlatıldı: {session_id} Masa: {table_id}")

    chat_history = request.session.get("chat_history", []) # Session'dan sohbet geçmişini al
    logger.info(f"💬 AI Yanıt isteği: Masa '{table_id}', Session ID: '{session_id}', Kullanıcı Mesajı: '{user_message}'")

    if not user_message:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Mesaj boş olamaz.")
    if SYSTEM_PROMPT is None: # Sistem prompt'u yüklenmemişse
        logger.error("❌ AI Yanıt: Sistem promptu yüklenmemiş! update_system_prompt düzgün çalışmamış olabilir.")
        # Belki burada update_system_prompt'u tekrar çağırmayı deneyebiliriz? Veya hata döndür.
        # await update_system_prompt() # Deneyelim, ama dikkatli olmalı, recursive loop riski.
        # if SYSTEM_PROMPT is None: # Hala yüklenemediyse hata ver.
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="AI asistanı şu anda hazır değil (sistem mesajı eksik). Lütfen biraz sonra tekrar deneyin.")

    try:
        messages_for_openai = [SYSTEM_PROMPT] + chat_history + [{"role": "user", "content": user_message}]
        logger.debug(f"OpenAI'ye gönderilecek mesajlar (ilk mesaj hariç son 3): {messages_for_openai[-3:]}")

        response = openai_client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages_for_openai, # type: ignore
            temperature=0.3, # Biraz daha deterministik olması için düşürüldü
            max_tokens=450, # Yanıt uzunluğu
            # response_format={ "type": "json_object" } # Eğer AI'dan sadece JSON bekleniyorsa
        )
        ai_reply = response.choices[0].message.content
        if ai_reply is None:
            ai_reply = "Üzgünüm, şu anda bir yanıt üretemiyorum." # Fallback
            logger.warning("OpenAI'den boş yanıt (None) alındı.")
        else:
            ai_reply = ai_reply.strip()

        # Sohbet geçmişini güncelle (kullanıcı ve AI mesajları)
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": ai_reply}) # ai_reply AI'ın ham yanıtı
        request.session["chat_history"] = chat_history[-10:] # Son 5 konuşmayı (10 mesaj) sakla

        logger.info(f"🤖 HAM AI Yanıtı (Masa: {table_id}, Session: {session_id}): {ai_reply}") # HAM YANITI LOGLA
        return {"reply": ai_reply, "sessionId": session_id} # AI'ın ham yanıtını döndür
    except OpenAIError as e:
        logger.error(f"❌ OpenAI API ile iletişim hatası (Masa: {table_id}): {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"AI servisinden yanıt alınırken bir sorun oluştu: {e}")
    except Exception as e:
        logger.error(f"❌ AI yanıt endpoint'inde beklenmedik hata (Masa: {table_id}): {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Mesajınız işlenirken beklenmedik bir sunucu hatası oluştu.")

# İstatistikler
def calculate_statistics(orders_data: List[dict]) -> tuple[int, int, float]: # adet int olmalı
    total_orders_count = len(orders_data)
    total_items_sold = 0 # int olarak başlat
    total_revenue = 0.0

    for order_row in orders_data: # order_row zaten bir dict
        try:
            sepet_items_str = order_row.get('sepet') # .get() burada kullanılabilir
            items = []
            if isinstance(sepet_items_str, str):
                if sepet_items_str.strip(): # Boş string değilse parse et
                    items = json.loads(sepet_items_str)
            elif isinstance(sepet_items_str, list): # Zaten liste ise direkt kullan
                items = sepet_items_str

            if not isinstance(items, list): # Hala liste değilse (örn. null veya başka bir tip)
                logger.warning(f"⚠️ İstatistik: Sepet öğesi beklenen liste formatında değil: {type(items)} - Sipariş ID: {order_row.get('id')}")
                items = [] # Boş liste ile devam et

            for item in items:
                if isinstance(item, dict):
                    adet = item.get("adet", 0)
                    fiyat = item.get("fiyat", 0.0)
                    # Adet ve fiyatın sayısal olup olmadığını kontrol et
                    if isinstance(adet, (int, float)) and isinstance(fiyat, (int, float)):
                        total_items_sold += int(adet) # Adet tam sayı olmalı
                        total_revenue += adet * fiyat
                    else:
                        logger.warning(f"⚠️ İstatistik: Sepet öğesinde geçersiz adet/fiyat: {item} - Sipariş ID: {order_row.get('id')}")
                else:
                    logger.warning(f"⚠️ İstatistik: Sepet öğesi dict değil: {item} - Sipariş ID: {order_row.get('id')}")
        except json.JSONDecodeError:
            logger.warning(f"⚠️ İstatistik: Sepet JSON parse hatası. Sipariş ID: {order_row.get('id')}, Sepet Verisi (ilk 50 krkt): {str(order_row.get('sepet'))[:50]}")
        except KeyError: # Örneğin 'sepet' anahtarı yoksa
            logger.warning(f"⚠️ İstatistik: 'sepet' anahtarı bulunamadı veya başka bir key hatası. Sipariş ID: {order_row.get('id')}")
        except Exception as e: # Diğer beklenmedik hatalar
            logger.error(f"⚠️ İstatistik hesaplama sırasında beklenmedik hata: {e} - Sipariş ID: {order_row.get('id')}", exc_info=True)


    return total_orders_count, total_items_sold, round(total_revenue, 2)

@app.get("/istatistik/en-cok-satilan")
async def get_popular_items_endpoint(limit: int = Query(5, ge=1, le=20), auth: bool = Depends(check_admin)):
    logger.info(f"📊 En çok satılan {limit} ürün istatistiği isteniyor.")
    item_counts: Dict[str, int] = {}
    try:
        # Sadece iptal olmayan siparişleri al
        orders_raw = await db.fetch_all("SELECT sepet FROM siparisler WHERE durum != 'iptal'")
        for row_record in orders_raw: # row_record bir Record objesi
            row_as_dict = dict(row_record) # Record'u dict'e çevir
            try:
                sepet_items_str = row_as_dict.get('sepet') # row_as_dict.get() doğru kullanım
                items = []
                if isinstance(sepet_items_str, str):
                    if sepet_items_str.strip():
                        items = json.loads(sepet_items_str)
                elif isinstance(sepet_items_str, list):
                    items = sepet_items_str

                if not isinstance(items, list):
                    logger.warning(f"Popüler ürünler: Sepet öğesi beklenen liste formatında değil: {type(items)} - Satır: {row_as_dict}")
                    items = []

                for item in items:
                    if isinstance(item, dict):
                        item_name = item.get("urun")
                        quantity = item.get("adet", 0)
                        if item_name and isinstance(quantity, (int, float)) and quantity > 0:
                            item_counts[item_name] = item_counts.get(item_name, 0) + int(quantity) # Adet int olmalı
            except json.JSONDecodeError:
                logger.warning(f"⚠️ Popüler ürünler: Sepet JSON parse hatası. Veri (ilk 50): {str(sepet_items_str)[:50]}")
            except Exception as e_inner:
                logger.error(f"⚠️ Popüler ürünler: Sepet işleme sırasında beklenmedik iç hata: {e_inner} - Satır: {row_as_dict}", exc_info=True)

        sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        logger.info(f"✅ En çok satılan {len(sorted_items)} ürün bulundu.")
        return [{"urun": item, "adet": count} for item, count in sorted_items]
    except Exception as e_outer:
        logger.error(f"❌ Popüler ürünler istatistiği alınırken genel hata: {e_outer}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Popüler ürün istatistikleri alınamadı.")


async def get_stats_for_period(start_date_str: str, end_date_str: Optional[str] = None) -> dict:
    # Zaman formatını YYYY-AA-GG HH:MM:SS olarak varsayalım, eğer sadece tarihse 00:00:00 ekleyelim
    start_datetime_str = f"{start_date_str} 00:00:00"
    query = "SELECT id, sepet, zaman FROM siparisler WHERE durum != 'iptal' AND zaman >= :start_dt"
    values: Dict[str, any] = {"start_dt": start_datetime_str}

    if end_date_str:
        # Bitiş tarihini bir gün sonrasının başlangıcı olarak al (dahil etmek için)
        end_datetime_obj = datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1)
        end_datetime_str = end_datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
        query += " AND zaman < :end_dt" # < kullanarak bitiş gününü dahil etme
        values["end_dt"] = end_datetime_str
    # Eğer end_date_str yoksa, sadece başlangıçtan sonrasını alır (günlük için bu kullanılmaz)

    orders_for_stats_records = await db.fetch_all(query, values)
    orders_list = [dict(record) for record in orders_for_stats_records] # Record'ları dict'e çevir
    total_orders_count, total_items_sold, total_revenue = calculate_statistics(orders_list)
    return {
        "siparis_sayisi": total_orders_count,
        "satilan_urun_adedi": total_items_sold,
        "toplam_gelir": total_revenue,
        "veri_adedi": len(orders_list) # Kaç siparişin işlendiğini de döndür
    }

@app.get("/istatistik/gunluk")
async def get_daily_stats_endpoint(tarih: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="Belirli bir günün istatistiği (YYYY-AA-GG). Boş bırakılırsa bugünün istatistiği."), auth: bool = Depends(check_admin)):
    target_date_str = tarih if tarih else datetime.now().strftime("%Y-%m-%d")
    logger.info(f"📊 Günlük istatistik isteniyor: {target_date_str}")
    try:
        # get_stats_for_period hem başlangıç hem bitişi aynı gün olarak alır
        stats = await get_stats_for_period(target_date_str, target_date_str)
        logger.info(f"✅ Günlük istatistik ({target_date_str}) hesaplandı.")
        return {"tarih": target_date_str, **stats}
    except ValueError: # Tarih formatı hatası
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
        # Ayın son gününü bul
        if target_month == 12:
            end_date = datetime(target_year, 12, 31) # Yılın son günü
        else:
            end_date = datetime(target_year, target_month + 1, 1) - timedelta(days=1)

        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        logger.info(f"📊 Aylık istatistik isteniyor: {target_year}-{target_month:02d} ({start_date_str} - {end_date_str})")
        stats = await get_stats_for_period(start_date_str, end_date_str)
        logger.info(f"✅ Aylık istatistik ({target_year}-{target_month:02d}) hesaplandı.")
        return {"yil": target_year, "ay": target_month, **stats}
    except ValueError as ve: # datetime() için geçersiz yıl/ay
        logger.error(f"❌ Aylık istatistik: Geçersiz yıl/ay değeri: Yıl={yil}, Ay={ay}. Hata: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Geçersiz yıl veya ay değeri. {ve}")
    except Exception as e:
        logger.error(f"❌ Aylık istatistik ({target_year}-{target_month:02d}) alınırken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Aylık istatistikler alınamadı.")

@app.get("/istatistik/yillik-aylik-kirilim")
async def get_yearly_stats_by_month_endpoint(yil: Optional[int] = Query(None, ge=2000, le=datetime.now().year + 1), auth: bool = Depends(check_admin)):
    target_year = yil if yil else datetime.now().year
    logger.info(f"📊 Yıllık ({target_year}) aylık kırılımlı istatistik isteniyor (/istatistik/yillik-aylik-kirilim).")
    try:
        start_of_year_str = f"{target_year}-01-01 00:00:00"
        # Yılın sonunu bir sonraki yılın başı olarak alıp < ile karşılaştıracağız
        end_of_year_exclusive_str = f"{target_year+1}-01-01 00:00:00"

        query = """
            SELECT id, sepet, zaman FROM siparisler
            WHERE durum != 'iptal' AND zaman >= :start AND zaman < :end_exclusive
            ORDER BY zaman ASC
        """
        orders_raw_records = await db.fetch_all(query, {"start": start_of_year_str, "end_exclusive": end_of_year_exclusive_str})

        monthly_stats: Dict[str, Dict[str, any]] = {} # örn: {"2023-01": {"siparis_sayisi": ..., ...}}
        orders_as_dicts = [dict(record) for record in orders_raw_records] # Record'ları dict'e çevir

        for row_dict in orders_as_dicts:
            try:
                order_time_str = row_dict.get('zaman', '')
                order_datetime = None
                # Farklı zaman formatlarını dene (SQLite'tan nasıl geldiğine bağlı)
                possible_formats = ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]
                for fmt in possible_formats:
                    try:
                        order_datetime = datetime.strptime(order_time_str.split('.')[0], fmt.split('.')[0]) # Milisaniyeyi at
                        break
                    except ValueError:
                        continue

                if not order_datetime:
                    logger.warning(f"Yıllık istatistik: Geçersiz zaman formatı: {order_time_str} Sipariş ID: {row_dict.get('id')}")
                    continue

                month_key = order_datetime.strftime("%Y-%m") # "2023-01" formatında

                if month_key not in monthly_stats:
                    monthly_stats[month_key] = {"siparis_sayisi": 0, "satilan_urun_adedi": 0, "toplam_gelir": 0.0}

                sepet_items_str = row_dict.get('sepet')
                items = []
                if isinstance(sepet_items_str, str):
                    if sepet_items_str.strip(): items = json.loads(sepet_items_str)
                elif isinstance(sepet_items_str, list):
                    items = sepet_items_str

                if not isinstance(items, list):
                    logger.warning(f"Yıllık istatistik: Sepet öğesi beklenen liste formatında değil: {type(items)} - Sipariş ID: {row_dict.get('id')}")
                    items = []

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
                logger.warning(f"⚠️ Yıllık istatistik JSON parse hatası. Sipariş ID: {row_dict.get('id')}, Veri (ilk 50): {str(row_dict.get('sepet'))[:50]}")
            except Exception as e_inner:
                logger.error(f"⚠️ Yıllık istatistik (aylık kırılım) iç döngü hatası: {e_inner} - Sipariş ID: {row_dict.get('id')}", exc_info=True)

        logger.info(f"✅ Yıllık ({target_year}) aylık kırılımlı istatistik hesaplandı ({len(monthly_stats)} ay).")
        return {"yil": target_year, "aylik_kirilim": dict(sorted(monthly_stats.items()))} # Aylara göre sıralı döndür
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
        # Tarih formatlarının doğruluğu Query pattern ile sağlanıyor
        start_dt = datetime.strptime(baslangic, "%Y-%m-%d")
        end_dt = datetime.strptime(bitis, "%Y-%m-%d")
        if start_dt > end_dt:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Başlangıç tarihi bitiş tarihinden sonra olamaz.")

        stats = await get_stats_for_period(baslangic, bitis)
        logger.info(f"✅ Filtreli istatistik ({baslangic} - {bitis}) hesaplandı.")
        return {"aralik": f"{baslangic} → {bitis}", **stats}
    except ValueError: # Tarih formatı hatası (pattern'e rağmen olabilir)
        logger.error(f"❌ Filtreli istatistik: Geçersiz tarih formatı. Başlangıç: {baslangic}, Bitiş: {bitis}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Geçersiz tarih formatı. Lütfen YYYY-AA-GG formatını kullanın.")
    except HTTPException as http_exc: # Kendi fırlattığımız HTTP hataları
        raise http_exc
    except Exception as e:
        logger.error(f"❌ Filtreli istatistik ({baslangic} - {bitis}) alınırken hata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Belirtilen aralık için istatistikler alınamadı.")

# Sesli Yanıt
SUPPORTED_LANGUAGES = {"tr-TR", "en-US", "en-GB", "fr-FR", "de-DE"} # Desteklenen diller

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
        cleaned_text = temizle_emoji(data.text) # Emojileri temizle
        if not cleaned_text.strip(): # Temizlenmiş metin boşsa
            logger.warning("⚠️ Sesli yanıt: Boş veya sadece emojiden oluşan metin.")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Sese dönüştürülecek geçerli bir metin bulunamadı.")

        synthesis_input = texttospeech.SynthesisInput(text=cleaned_text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=data.language,
            # name="tr-TR-Standard-A", # Belirli bir ses seçilebilir, yoksa varsayılan kullanılır
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE # Genellikle daha doğal
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0 # Konuşma hızı (0.25 - 4.0)
        )
        response_tts = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        logger.info(f"✅ Sesli yanıt başarıyla oluşturuldu (Dil: {data.language}).")
        return Response(content=response_tts.audio_content, media_type="audio/mpeg")
    except google_exceptions.GoogleAPIError as e: # Google API hatalarını özel olarak yakala
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
    # Bu fonksiyon şifreyi GERÇEKTEN DEĞİŞTİRMEZ. Sadece bir hatırlatma yapar.
    # Gerçek değişiklik için .env dosyasının manuel güncellenmesi ve uygulamanın
    # yeniden başlatılması gerekir.
    logger.warning(f"ℹ️ Admin şifre/kullanıcı adı değiştirme endpoint'i çağrıldı (Kullanıcı: {creds.yeniKullaniciAdi}). Bu işlem için .env dosyasının manuel güncellenmesi gerekmektedir.")
    return {
        "mesaj": "Admin kullanıcı adı ve şifresini değiştirmek için lütfen sunucudaki .env dosyasını güncelleyin ve uygulamayı yeniden başlatın. Bu endpoint sadece bir hatırlatmadır ve aktif bir değişiklik yapmaz."
    }


if __name__ == "__main__":
    import uvicorn
    # Ortam değişkenlerinden host ve port al, yoksa varsayılan kullan
    host_ip = os.getenv("HOST", "127.0.0.1")
    port_num = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 FastAPI uygulaması {host_ip}:{port_num} adresinde başlatılıyor (yerel geliştirme modu)...")
    uvicorn.run("main:app", host=host_ip, port=port_num, reload=True) # reload=True geliştirme için