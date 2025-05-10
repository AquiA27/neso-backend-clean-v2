from fastapi import (
       FastAPI, Request, Body, Query, HTTPException, status, Depends, WebSocket, WebSocketDisconnect
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
   import regex
   import tempfile
   import sqlite3
   import json
   import logging
   import logging.config
   from datetime import datetime, timedelta
   from dotenv import load_dotenv
   from openai import OpenAI, OpenAIError
   from google.cloud import texttospeech
   from google.api_core import exceptions as google_exceptions
   import asyncio
   import secrets
   from enum import Enum

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
           emoji_pattern = regex.compile(r"[\p{Emoji_Presentation}\p{Extended_Pictographic}]+")
           return emoji_pattern.sub('', text)
       except Exception as e:
           logger.error(f"Emoji temizleme hatası: {e}")
           return text

   # API İstemcileri Başlatma
   openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
   logger.info("✅ OpenAI istemcisi başlatıldı.")

   google_creds_path = None
   tts_client = None
   try:
       decoded_creds = base64.b64decode(settings.GOOGLE_APPLICATION_CREDENTIALS_BASE64)
       with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w+b') as tmp_file:
           tmp_file.write(decoded_creds)
           google_creds_path = tmp_file.name
           os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_creds_path
       tts_client = texttospeech.TextToSpeechClient()
       logger.info("✅ Google TTS istemcisi başlatıldı.")
   except Exception as e:
       logger.warning(f"❌ Google TTS istemcisi başlatılamadı: {e}")

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
   logger.info(f"CORS Middleware: {allowed_origins_list}")

   # Veritabanı Bağlantı Havuzu
   DB_NAME = "neso.db"
   MENU_DB_NAME = "neso_menu.db"
   DB_PATH = os.path.join(settings.DB_DATA_DIR, DB_NAME)
   MENU_DB_PATH = os.path.join(settings.DB_DATA_DIR, MENU_DB_NAME)
   os.makedirs(settings.DB_DATA_DIR, exist_ok=True)

   db = Database(f"sqlite:///{DB_PATH}")
   menu_db = Database(f"sqlite:///{MENU_DB_PATH}")

   @app.on_event("startup")
   async def startup():
       await db.connect()
       await menu_db.connect()
       logger.info("✅ Veritabanı bağlantıları kuruldu.")

   @app.on_event("shutdown")
   async def shutdown():
       await db.disconnect()
       await menu_db.disconnect()
       if google_creds_path and os.path.exists(google_creds_path):
           try:
               os.remove(google_creds_path)
               logger.info("✅ Google kimlik bilgisi dosyası silindi.")
           except OSError as e:
               logger.error(f"❌ Google kimlik bilgisi dosyası silinemedi: {e}")
       logger.info("👋 Uygulama kapatıldı.")

   # WebSocket Yönetimi
   aktif_mutfak_websocketleri: Set[WebSocket] = set()
   aktif_admin_websocketleri: Set[WebSocket] = set()

   async def broadcast_message(connections: Set[WebSocket], message: Dict):
       if not connections:
           logger.warning(f"⚠️ Broadcast: Bağlı {message.get('type')} istemcisi yok.")
           return
       message_json = json.dumps(message)
       tasks = []
       for ws in connections:
           try:
               tasks.append(ws.send_text(message_json))
           except RuntimeError:
               connections.discard(ws)
               logger.warning(f"⚠️ WebSocket bağlantısı kopuk, kaldırılıyor: {ws.client}")
       results = await asyncio.gather(*tasks, return_exceptions=True)
       for ws, result in zip(list(connections), results):
           if isinstance(result, Exception):
               connections.discard(ws)
               logger.warning(f"⚠️ WebSocket gönderme hatası, bağlantı kaldırılıyor: {result}")

   async def websocket_lifecycle(websocket: WebSocket, connections: Set[WebSocket], endpoint_name: str):
       await websocket.accept()
       connections.add(websocket)
       client_info = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "Bilinmeyen"
       logger.info(f"🔗 {endpoint_name} WS bağlandı: {client_info}")
       try:
           while True:
               data = await websocket.receive_text()
               try:
                   message = json.loads(data)
                   if message.get("type") == "ping":
                       await websocket.send_text(json.dumps({"type": "pong"}))
                       logger.debug(f"🏓 {endpoint_name} WS: Ping/Pong alındı: {client_info}")
                   elif message.get("type") == "status_update" and endpoint_name == "Admin":
                       logger.info(f"Admin WS: Durum güncelleme: {message.get('data')}")
               except json.JSONDecodeError:
                   logger.warning(f"⚠️ {endpoint_name} WS: Geçersiz JSON: {data}")
       except WebSocketDisconnect as e:
           logger.info(f"🔌 {endpoint_name} WS kapandı: {client_info} (Kod: {e.code})")
       except Exception as e:
           logger.error(f"❌ {endpoint_name} WS beklenmedik hata: {e}")
       finally:
           connections.discard(websocket)
           logger.info(f"📉 {endpoint_name} WS kaldırıldı: {client_info}")

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
           })
       except Exception as e:
           logger.error(f"❌ Masa durumu güncelleme hatası: {e}")

   # Middleware
   @app.middleware("http")
   async def track_active_users(request: Request, call_next):
       masa_id = request.path_params.get("masaId") or request.query_params.get("masa_id")
       if masa_id:
           endpoint_name = request.scope.get("endpoint").__name__ if request.scope.get("endpoint") else request.url.path
           await update_table_status(str(masa_id), f"{request.method} {endpoint_name}")
       try:
           return await call_next(request)
       except Exception as e:
           logger.exception(f"❌ Middleware hatası: {e}")
           raise HTTPException(status_code=500, detail="Sunucu hatası")

   # Endpoint'ler
   @app.get("/aktif-masalar")
   async def get_active_tables_endpoint():
       active_time_limit = datetime.now() - timedelta(minutes=5)
       try:
           tables = await db.fetch_all("""
               SELECT masa_id, son_erisim, aktif, son_islem FROM masa_durumlar
               WHERE son_erisim >= :limit AND aktif = TRUE ORDER BY son_erisim DESC
           """, {"limit": active_time_limit.strftime("%Y-%m-%d %H:%M:%S")})
           return {"tables": [dict(row) for row in tables]}
       except Exception as e:
           logger.error(f"❌ Aktif masalar alınamadı: {e}")
           raise HTTPException(status_code=503, detail="Veritabanı hatası")

   # Admin Doğrulama
   def check_admin(credentials: HTTPBasicCredentials = Depends(security)):
       is_user_ok = secrets.compare_digest(credentials.username.encode('utf-8'), settings.ADMIN_USERNAME.encode('utf-8'))
       is_pass_ok = secrets.compare_digest(credentials.password.encode('utf-8'), settings.ADMIN_PASSWORD.encode('utf-8'))
       if not (is_user_ok and is_pass_ok):
           logger.warning(f"🔒 Başarısız admin girişi: {credentials.username}")
           raise HTTPException(
               status_code=status.HTTP_401_UNAUTHORIZED,
               detail="Geçersiz kimlik bilgileri",
               headers={"WWW-Authenticate": "Basic"},
           )
       return True

   # Pydantic Modelleri
   class Durum(str, Enum):
       BEKLIYOR = "bekliyor"
       HAZIRLANIYOR = "hazirlaniyor"
       HAZIR = "hazir"
       IPTAL = "iptal"

   class SepetItem(BaseModel):
       urun: str = Field(..., min_length=1)
       adet: int = Field(..., gt=0)
       fiyat: float = Field(..., ge=0)
       kategori: Optional[str] = None

   class SiparisEkleData(BaseModel):
       masa: str = Field(..., min_length=1)
       sepet: List[SepetItem] = Field(..., min_items=1)
       istek: Optional[str] = None
       yanit: Optional[str] = None

   class SiparisGuncelleData(BaseModel):
       masa: str = Field(..., min_length=1)
       durum: Durum
       id: Optional[int] = None

   class MenuEkleData(BaseModel):
       ad: str = Field(..., min_length=1)
       fiyat: float = Field(..., gt=0)
       kategori: str = Field(..., min_length=1)

   class AdminCredentialsUpdate(BaseModel):
       yeniKullaniciAdi: str = Field(..., min_length=1)
       yeniSifre: str = Field(..., min_length=8)

   class SesliYanitData(BaseModel):
       text: str = Field(..., min_length=1)
       language: str = Field(default="tr-TR", pattern=r"^[a-z]{2}-[A-Z]{2}$")

   # Sipariş Yönetimi
   @app.post("/siparis-ekle", status_code=status.HTTP_201_CREATED)
   async def add_order_endpoint(data: SiparisEkleData):
       masa = data.masa
       sepet = data.sepet
       istek = data.istek
       yanit = data.yanit
       zaman_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
       logger.info(f"📥 Sipariş: Masa {masa}, {len(sepet)} ürün")

       price_dict = get_menu_price_dict()
       stock_dict = get_menu_stock_dict()
       processed_sepet = []
       for item in sepet:
           urun_adi_lower = item.urun.lower().strip()
           if urun_adi_lower not in stock_dict or stock_dict[urun_adi_lower] == 0:
               logger.warning(f"⚠️ Stokta yok: {urun_adi_lower}")
               raise HTTPException(status_code=400, detail=f"'{item.urun}' stokta yok.")
           item_dict = item.model_dump()
           item_dict['fiyat'] = price_dict.get(urun_adi_lower, 0.0)
           processed_sepet.append(item_dict)

       istek_ozet = ", ".join([f"{item['adet']}x {item['urun']}" for item in processed_sepet])
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
                   "sepet": json.dumps(processed_sepet),
                   "zaman": zaman_str
               })
           siparis_bilgisi = {
               "type": "siparis",
               "data": {"id": siparis_id, "masa": masa, "istek": istek or istek_ozet, "sepet": processed_sepet, "zaman": zaman_str, "durum": "bekliyor"}
           }
           logger.info(f"📢 Broadcast: Yeni sipariş (ID: {siparis_id}) gönderiliyor...")
           await broadcast_message(aktif_mutfak_websocketleri, siparis_bilgisi)
           await broadcast_message(aktif_admin_websocketleri, siparis_bilgisi)
           await update_table_status(masa, f"Sipariş verdi ({len(processed_sepet)} ürün)")
           logger.info(f"✅ Sipariş başarıyla kaydedildi: ID {siparis_id}")
           return {"mesaj": "Sipariş kaydedildi.", "siparisId": siparis_id}
       except Exception as e:
           logger.error(f"❌ Sipariş ekleme hatası: {e}")
           raise HTTPException(status_code=503, detail="Veritabanı hatası")

   @app.post("/siparis-guncelle")
   async def update_order_status_endpoint(data: SiparisGuncelleData, auth: bool = Depends(check_admin)):
       try:
           async with db.transaction():
               if data.id:
                   rows_affected = await db.execute("UPDATE siparisler SET durum = :durum WHERE id = :id",
                                                  {"durum": data.durum, "id": data.id})
               else:
                   rows_affected = await db.execute("""
                       UPDATE siparisler SET durum = :durum WHERE id = (
                           SELECT id FROM siparisler WHERE masa = :masa AND durum NOT IN ('hazir', 'iptal')
                           ORDER BY id DESC LIMIT 1)
                   """, {"durum": data.durum, "masa": data.masa})
           if rows_affected:
               notification = {
                   "type": "durum",
                   "data": {"id": data.id, "masa": data.masa, "durum": data.durum, "zaman": datetime.now().isoformat()}
               }
               await broadcast_message(aktif_mutfak_websocketleri, notification)
               await broadcast_message(aktif_admin_websocketleri, notification)
               await update_table_status(data.masa, f"Sipariş durumu -> {data.durum}")
               return {"message": f"Sipariş '{data.durum}' olarak güncellendi."}
           raise HTTPException(status_code=404, detail="Sipariş bulunamadı.")
       except Exception as e:
           logger.error(f"❌ Sipariş güncelleme hatası: {e}")
           raise HTTPException(status_code=503, detail="Veritabanı hatası")

   @app.get("/siparisler")
   async def get_orders_endpoint(auth: bool = Depends(check_admin)):
       try:
           orders = await db.fetch_all("SELECT id, masa, istek, yanit, sepet, zaman, durum FROM siparisler ORDER BY id DESC")
           orders_data = []
           for row in orders:
               order_dict = dict(row)
               try:
                   order_dict['sepet'] = json.loads(order_dict['sepet'] or '[]')
               except json.JSONDecodeError:
                   order_dict['sepet'] = []
                   logger.warning(f"⚠️ Geçersiz sepet JSON: ID {order_dict['id']}")
               orders_data.append(order_dict)
           return {"orders": orders_data}
       except Exception as e:
           logger.error(f"❌ Siparişler alınamadı: {e}")
           raise HTTPException(status_code=503, detail="Veritabanı hatası")

   # Veritabanı Başlatma
   async def init_db():
       try:
           await db.execute("""
               CREATE TABLE IF NOT EXISTS siparisler (
                   id INTEGER PRIMARY KEY AUTOINCREMENT, masa TEXT NOT NULL, istek TEXT,
                   yanit TEXT, sepet TEXT, zaman TEXT NOT NULL,
                   durum TEXT DEFAULT 'bekliyor' CHECK(durum IN ('bekliyor', 'hazirlaniyor', 'hazir', 'iptal'))
               )""")
           await db.execute("""
               CREATE TABLE IF NOT EXISTS masa_durumlar (
                   id INTEGER PRIMARY KEY AUTOINCREMENT, masa_id TEXT UNIQUE NOT NULL,
                   son_erisim TIMESTAMP NOT NULL, aktif BOOLEAN DEFAULT TRUE, son_islem TEXT
               )""")
           await db.execute("CREATE INDEX IF NOT EXISTS idx_siparisler_masa_zaman ON siparisler(masa, zaman DESC)")
           await db.execute("CREATE INDEX IF NOT EXISTS idx_masa_durumlar_erisim ON masa_durumlar(son_erisim DESC)")
           logger.info(f"✅ Ana veritabanı başlatıldı: {DB_PATH}")
       except Exception as e:
           logger.critical(f"❌ Ana veritabanı başlatılamadı: {e}")
           raise

   async def init_menu_db():
       try:
           await menu_db.execute("CREATE TABLE IF NOT EXISTS kategoriler (id INTEGER PRIMARY KEY AUTOINCREMENT, isim TEXT UNIQUE NOT NULL COLLATE NOCASE)")
           await menu_db.execute("""
               CREATE TABLE IF NOT EXISTS menu (
                   id INTEGER PRIMARY KEY AUTOINCREMENT, ad TEXT NOT NULL COLLATE NOCASE,
                   fiyat REAL NOT NULL CHECK(fiyat >= 0), kategori_id INTEGER NOT NULL,
                   stok_durumu INTEGER DEFAULT 1,
                   FOREIGN KEY (kategori_id) REFERENCES kategoriler(id) ON DELETE CASCADE, UNIQUE(ad, kategori_id)
               )""")
           await menu_db.execute("CREATE INDEX IF NOT EXISTS idx_menu_kategori ON menu(kategori_id)")
           await menu_db.execute("CREATE INDEX IF NOT EXISTS idx_menu_ad ON menu(ad)")
           logger.info(f"✅ Menü veritabanı başlatıldı: {MENU_DB_PATH}")
       except Exception as e:
           logger.critical(f"❌ Menü veritabanı başlatılamadı: {e}")
           raise

   @app.on_event("startup")
   async def init_databases():
       await init_db()
       await init_menu_db()

   # Menü Yönetimi
   @lru_cache(maxsize=1)
   def get_menu_for_prompt() -> str:
       try:
           urunler = menu_db.fetch_all("""
               SELECT k.isim, m.ad FROM menu m
               JOIN kategoriler k ON m.kategori_id = k.id
               WHERE m.stok_durumu = 1 ORDER BY k.isim, m.ad
           """)
           if not urunler:
               return "Menü bilgisi mevcut değil."
           kategorili_menu = {}
           for kategori, urun in urunler:
               kategorili_menu.setdefault(kategori, []).append(urun)
           menu_aciklama = "\n".join([f"- {k}: {', '.join(u)}" for k, u in kategorili_menu.items()])
           return "Mevcut menümüz:\n" + menu_aciklama
       except Exception as e:
           logger.error(f"❌ Menü prompt hatası: {e}")
           return "Menü bilgisi alınamadı."

   @lru_cache(maxsize=1)
   def get_menu_price_dict() -> Dict[str, float]:
       try:
           prices = menu_db.fetch_all("SELECT ad, fiyat FROM menu")
           return {ad.lower().strip(): fiyat for ad, fiyat in prices}
       except Exception as e:
           logger.error(f"❌ Fiyat sözlüğü hatası: {e}")
           return {}

   @lru_cache(maxsize=1)
   def get_menu_stock_dict() -> Dict[str, int]:
       try:
           stocks = menu_db.fetch_all("SELECT ad, stok_durumu FROM menu")
           return {ad.lower().strip(): stok for ad, stok in stocks}
       except Exception as e:
           logger.error(f"❌ Stok sözlüğü hatası: {e}")
           return {}

   SISTEM_MESAJI_ICERIK = (
       "Sen, Gaziantep'teki Fıstık Kafe için Neso sipariş asistanısın. "
       "Müşterilerin taleplerini anlayıp menüdeki ürünlerle eşleştirerek sipariş alıyorsun. "
       "Nazik, yardımsever ve kibar bir Türkçe kullan. Anlamadığında netleştirme soruları sor. "
       "Sipariş tamamlandığında 'Afiyet olsun!' de.\n\n"
       f"{get_menu_for_prompt()}"
   )
   SYSTEM_PROMPT = {"role": "system", "content": SISTEM_MESAJI_ICERIK}

   async def update_system_prompt():
       global SISTEM_MESAJI_ICERIK, SYSTEM_PROMPT
       try:
           get_menu_for_prompt.cache_clear()
           SISTEM_MESAJI_ICERIK = (
               "Sen, Gaziantep'teki Fıstık Kafe için Neso sipariş asistanısın. "
               "Müşterilerin taleplerini anlayıp menüdeki ürünlerle eşleştirerek sipariş alıyorsun. "
               "Nazik, yardımsever ve kibar bir Türkçe kullan. Anlamadığında netleştirme soruları sor. "
               "Sipariş tamamlandığında 'Afiyet olsun!' de.\n\n"
               f"{get_menu_for_prompt()}"
           )
           SYSTEM_PROMPT = {"role": "system", "content": SISTEM_MESAJI_ICERIK}
           logger.info("✅ Sistem prompt güncellendi.")
       except Exception as e:
           logger.error(f"❌ Sistem prompt güncelleme hatası: {e}")

   @app.get("/menu")
   async def get_full_menu_endpoint():
       try:
           full_menu_data = []
           kategoriler = await menu_db.fetch_all("SELECT id, isim FROM kategoriler ORDER BY isim")
           for kat_row in kategoriler:
               urunler = await menu_db.fetch_all("SELECT ad, fiyat, stok_durumu FROM menu WHERE kategori_id = :id ORDER BY ad",
                                               {"id": kat_row['id']})
               full_menu_data.append({"kategori": kat_row['isim'], "urunler": [dict(urun) for urun in urunler]})
           return {"menu": full_menu_data}
       except Exception as e:
           logger.error(f"❌ Menü alınamadı: {e}")
           raise HTTPException(status_code=503, detail="Veritabanı hatası")

   @app.post("/menu/ekle", status_code=status.HTTP_201_CREATED)
   async def add_menu_item_endpoint(item_data: MenuEkleData, auth: bool = Depends(check_admin)):
       try:
           async with menu_db.transaction():
               await menu_db.execute("INSERT OR IGNORE INTO kategoriler (isim) VALUES (:isim)", {"isim": item_data.kategori})
               category_id = await menu_db.fetch_val("SELECT id FROM kategoriler WHERE isim = :isim", {"isim": item_data.kategori})
               item_id = await menu_db.fetch_val("""
                   INSERT INTO menu (ad, fiyat, kategori_id, stok_durumu) VALUES (:ad, :fiyat, :kategori_id, 1)
                   RETURNING id
               """, {"ad": item_data.ad, "fiyat": item_data.fiyat, "kategori_id": category_id})
           await update_system_prompt()
           get_menu_price_dict.cache_clear()
           get_menu_stock_dict.cache_clear()
           return {"mesaj": f"'{item_data.ad}' menüye eklendi.", "itemId": item_id}
       except Exception as e:
           logger.error(f"❌ Menü ekleme hatası: {e}")
           raise HTTPException(status_code=503, detail="Veritabanı hatası")

   @app.delete("/menu/sil")
   async def delete_menu_item_endpoint(urun_adi: str = Query(..., min_length=1), auth: bool = Depends(check_admin)):
       try:
           async with menu_db.transaction():
               rows_affected = await menu_db.execute("DELETE FROM menu WHERE ad = :ad", {"ad": urun_adi})
           if rows_affected:
               await update_system_prompt()
               get_menu_price_dict.cache_clear()
               get_menu_stock_dict.cache_clear()
               return {"mesaj": f"'{urun_adi}' silindi."}
           raise HTTPException(status_code=404, detail=f"'{urun_adi}' bulunamadı.")
       except Exception as e:
           logger.error(f"❌ Menü silme hatası: {e}")
           raise HTTPException(status_code=503, detail="Veritabanı hatası")

   # AI Yanıt
   @app.post("/yanitla")
   async def handle_message_endpoint(data: dict = Body(...)):
       user_message = data.get("text", "").strip()
       table_id = data.get("masa", "bilinmiyor")
       if not user_message:
           raise HTTPException(status_code=400, detail="Mesaj boş olamaz.")
       try:
           messages = [SYSTEM_PROMPT, {"role": "user", "content": user_message}]
           response = openai_client.chat.completions.create(
               model=settings.OPENAI_MODEL,
               messages=messages,
               temperature=0.6,
               max_tokens=300
           )
           ai_reply = response.choices[0].message.content.strip()
           return {"reply": ai_reply}
       except OpenAIError as e:
           logger.error(f"❌ OpenAI hatası: {e}")
           raise HTTPException(status_code=503, detail="AI servis hatası")

   # İstatistikler
   def calculate_statistics(data: List[dict]) -> tuple:
       total_items = 0
       total_revenue = 0.0
       for row in data:
           try:
               items = json.loads(row['sepet'] or '[]')
               for item in items:
                   if isinstance(item, dict):
                       adet = item.get("adet", 0)
                       fiyat = item.get("fiyat", 0.0)
                       if isinstance(adet, (int, float)) and isinstance(fiyat, (int, float)):
                           total_items += adet
                           total_revenue += adet * fiyat
           except json.JSONDecodeError:
               logger.warning(f"⚠️ Sepet JSON hatası: {row['sepet'][:50]}")
       return total_items, total_revenue

   @app.get("/istatistik/en-cok-satilan")
   async def get_popular_items_endpoint(auth: bool = Depends(check_admin)):
       try:
           item_counts = {}
           orders = await db.fetch_all("SELECT sepet FROM siparisler WHERE durum != 'iptal'")
           for row in orders:
               try:
                   items = json.loads(row['sepet'] or '[]')
                   for item in items:
                       if isinstance(item, dict):
                           item_name = item.get("urun")
                           quantity = item.get("adet", 1)
                           if item_name and isinstance(quantity, (int, float)):
                               item_counts[item_name] = item_counts.get(item_name, 0) + quantity
               except json.JSONDecodeError:
                   logger.warning(f"⚠️ Sepet JSON hatası: {row['sepet'][:50]}")
           sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:5]
           return [{"urun": item, "adet": count} for item, count in sorted_items]
       except Exception as e:
           logger.error(f"❌ Popüler ürünler hatası: {e}")
           raise HTTPException(status_code=503, detail="Veritabanı hatası")

   @app.get("/istatistik/gunluk")
   async def get_daily_stats_endpoint(auth: bool = Depends(check_admin)):
       today_str = datetime.now().strftime("%Y-%m-%d")
       try:
           orders = await db.fetch_all("SELECT sepet FROM siparisler WHERE zaman LIKE :today AND durum != 'iptal'",
                                     {"today": f"{today_str}%"})
           total_items, total_revenue = calculate_statistics(orders)
           return {"tarih": today_str, "siparis_sayisi": total_items, "gelir": total_revenue}
       except Exception as e:
           logger.error(f"❌ Günlük istatistik hatası: {e}")
           raise HTTPException(status_code=503, detail="Veritabanı hatası")

   @app.get("/istatistik/aylik")
   async def get_monthly_stats_endpoint(auth: bool = Depends(check_admin)):
       start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
       try:
           orders = await db.fetch_all("SELECT sepet FROM siparisler WHERE zaman >= :start AND durum != 'iptal'",
                                     {"start": start_date})
           total_items, total_revenue = calculate_statistics(orders)
           return {"baslangic": start_date, "siparis_sayisi": total_items, "gelir": total_revenue}
       except Exception as e:
           logger.error(f"❌ Aylık istatistik hatası: {e}")
           raise HTTPException(status_code=503, detail="Veritabanı hatası")

   @app.get("/istatistik/yillik")
   async def get_yearly_stats_endpoint(auth: bool = Depends(check_admin)):
       try:
           monthly_item_counts = {}
           orders = await db.fetch_all("SELECT zaman, sepet FROM siparisler WHERE durum != 'iptal'")
           for row in orders:
               try:
                   month_key = row['zaman'][:7]
                   items = json.loads(row['sepet'] or '[]')
                   month_total = sum(item.get("adet", 1) for item in items if isinstance(item, dict))
                   monthly_item_counts[month_key] = monthly_item_counts.get(month_key, 0) + month_total
               except json.JSONDecodeError:
                   logger.warning(f"⚠️ Yıllık istatistik JSON hatası: {row['sepet'][:50]}")
           return dict(sorted(monthly_item_counts.items()))
       except Exception as e:
           logger.error(f"❌ Yıllık istatistik hatası: {e}")
           raise HTTPException(status_code=503, detail="Veritabanı hatası")

   @app.get("/istatistik/filtreli")
   async def get_filtered_stats_endpoint(
       baslangic: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
       bitis: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
       auth: bool = Depends(check_admin)
   ):
       try:
           end_date_inclusive = (datetime.strptime(bitis, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
           orders = await db.fetch_all("SELECT sepet FROM siparisler WHERE zaman >= :start AND zaman < :end AND durum != 'iptal'",
                                     {"start": baslangic, "end": end_date_inclusive})
           total_items, total_revenue = calculate_statistics(orders)
           return {"aralik": f"{baslangic} → {bitis}", "siparis_sayisi": total_items, "gelir": total_revenue}
       except ValueError:
           logger.error("❌ Geçersiz tarih formatı")
           raise HTTPException(status_code=400, detail="Geçersiz tarih")
       except Exception as e:
           logger.error(f"❌ Filtreli istatistik hatası: {e}")
           raise HTTPException(status_code=503, detail="Veritabanı hatası")

   # Sesli Yanıt
   SUPPORTED_LANGUAGES = {"tr-TR", "en-US", "en-GB", "fr-FR", "de-DE"}

   @app.post("/sesli-yanit")
   async def generate_speech_endpoint(data: SesliYanitData):
       if not tts_client:
           raise HTTPException(status_code=503, detail="TTS servisi kullanılamıyor.")
       if data.language not in SUPPORTED_LANGUAGES:
           raise HTTPException(status_code=400, detail=f"Desteklenmeyen dil: {data.language}. Desteklenen diller: {SUPPORTED_LANGUAGES}")
       try:
           cleaned_text = temizle_emoji(data.text)
           if not cleaned_text.strip():
               raise HTTPException(status_code=400, detail="Geçerli metin yok.")
           synthesis_input = texttospeech.SynthesisInput(text=cleaned_text)
           voice = texttospeech.VoiceSelectionParams(
               language_code=data.language,
               ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
           )
           audio_config = texttospeech.AudioConfig(
               audio_encoding=texttospeech.AudioEncoding.MP3,
               speaking_rate=1.0
           )
           response = tts_client.synthesize_speech(
               input=synthesis_input, voice=voice, audio_config=audio_config
           )
           return Response(content=response.audio_content, media_type="audio/mpeg")
       except google_exceptions.GoogleAPIError as e:
           logger.error(f"❌ Google TTS hatası: {e}")
           raise HTTPException(status_code=503, detail="TTS servisi hatası")

   # Admin Şifre Değiştirme
   @app.post("/admin/sifre-degistir")
   async def change_admin_password_endpoint(creds: AdminCredentialsUpdate, auth: bool = Depends(check_admin)):
       logger.warning(f"ℹ️ Şifre değiştirme isteği: {creds.yeniKullaniciAdi}")
       return {
           "mesaj": "Şifre değiştirme için .env dosyasını güncelleyin ve sunucuyu yeniden başlatın."
       }

   if __name__ == "__main__":
       import uvicorn
       logger.info("🚀 FastAPI başlatılıyor...")
       uvicorn.run("main:app", host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)), reload=True)