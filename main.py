# main.py - Neso Sipariş Asistanı Backend
# Gerekli kütüphaneleri içe aktarma
from fastapi import FastAPI, Request, Body, Query, UploadFile, File, HTTPException, status, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.middleware.sessions import SessionMiddleware # Not: Aktif olarak kullanılmıyor gibi görünüyor.
import os
import base64
import tempfile
import sqlite3
import json
import csv
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
# from fuzzywuzzy import fuzz # Not: Kullanılmadığı için yorum satırı yapıldı.
from openai import OpenAI
from google.cloud import texttospeech
import re # Emoji temizleme için

# 🌍 Ortam Değişkenleri ve Loglama Yapılandırması
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s', # Log formatına fonksiyon adını ekledik
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Yardımcı Fonksiyonlar ---
def temizle_emoji(text):
    """Verilen metinden emojileri temizler."""
    if not isinstance(text, str):
        # Eğer string değilse (None vb.), olduğu gibi döndür
        return text
    try:
        # Kapsamlı emoji deseni (Tek bir string içinde, u' prefix olmadan)
        # Not: Çok satırlı string Python tarafından otomatik birleştirilir.
        emoji_pattern = re.compile("["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001FA70-\U0001FAFF"  # Yeni emojiler
            "\U00002600-\U000026FF"  # Çeşitli semboller
            "\U00002B50"            # Yıldız
            "\U000FE0F"             # Varyasyon seçici (emoji stilini etkileyebilir)
            "]+", flags=re.UNICODE) # re.UNICODE flag'ı Python 3'te varsayılan olabilir ama belirtmekte sakınca yok.

        # Desenle eşleşen tüm emojileri boş string ile değiştir
        return emoji_pattern.sub(r'', text)
    except re.error as e:
        # Eğer regex deseni derlenirken bir hata olursa (beklenmez ama olabilir)
        logger.error(f"Emoji regex derleme hatası: {e}")
        # Hata durumunda orijinal metni güvenli bir şekilde döndür
        return text
    except Exception as e:
        # Diğer beklenmedik hatalar için
        logger.error(f"Emoji temizleme sırasında beklenmedik hata: {e}")
        return text

# --- API Anahtarları ve İstemci Başlatma ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CREDS_BASE64 = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_BASE64")

# Google Cloud kimlik bilgilerini base64'ten çözüp geçici dosyaya yazma
google_creds_path = None
if GOOGLE_CREDS_BASE64:
    try:
        decoded_creds = base64.b64decode(GOOGLE_CREDS_BASE64)
        # Güvenli bir geçici dosya oluşturma
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w+b') as tmp_file:
            tmp_file.write(decoded_creds)
            google_creds_path = tmp_file.name
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_creds_path
        logger.info("✅ Google Cloud kimlik bilgileri başarıyla yüklendi.")
    except Exception as e:
        logger.error(f"❌ Google Cloud kimlik bilgileri işlenirken hata: {e}")
else:
    logger.warning("⚠️ Google Cloud kimlik bilgileri (GOOGLE_APPLICATION_CREDENTIALS_BASE64) bulunamadı. Sesli yanıt özelliği çalışmayabilir.")

# OpenAI İstemcisi
if not OPENAI_API_KEY:
    logger.warning("⚠️ OpenAI API anahtarı (OPENAI_API_KEY) bulunamadı. Yanıtlama özelliği çalışmayabilir.")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Google TTS İstemcisi (Sadece kimlik bilgisi varsa başlatılır)
tts_client = None
if google_creds_path:
    try:
        tts_client = texttospeech.TextToSpeechClient()
        logger.info("✅ Google Text-to-Speech istemcisi başarıyla başlatıldı.")
    except Exception as e:
        logger.error(f"❌ Google Text-to-Speech istemcisi başlatılamadı: {e}")

# --- FastAPI Uygulaması ve Güvenlik ---
app = FastAPI(title="Neso Sipariş Asistanı API", version="1.1.0") # Versiyon eklendi
security = HTTPBasic()

# --- Middleware Ayarları ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # DİKKAT: Güvenlik için üretimde spesifik domainlere izin verin!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", "gizli-anahtar-burada-olmamali"), # DİKKAT: Üretimde güvenli bir anahtar kullanın!
    session_cookie="neso_session" # Cookie adı değiştirildi
)

# --- WebSocket Bağlantı Yönetimi ---
aktif_mutfak_websocketleri: list[WebSocket] = []
aktif_admin_websocketleri: list[WebSocket] = []
# Not: aktif_kullanicilar ve masa_durumlari global değişkenleri yerine veritabanı tabanlı takip kullanılıyor.

async def broadcast_message(connections: list[WebSocket], message: dict, source_ws: WebSocket = None):
    """Belirtilen WebSocket bağlantılarına mesaj gönderir."""
    message_json = json.dumps(message)
    disconnected_sockets = []
    for ws in connections:
        # Kaynak WebSocket'e geri gönderme (opsiyonel)
        # if ws is source_ws:
        #     continue
        try:
            await ws.send_text(message_json)
            # logger.debug(f"📢 Mesaj gönderildi: {ws.client}, Tip: {message.get('type')}")
        except Exception as e:
            logger.warning(f"🔌 WebSocket gönderme hatası (kapatılıyor): {ws.client} - {e}")
            disconnected_sockets.append(ws)

    # Bağlantısı kopan soketleri listeden temizle
    for ws in disconnected_sockets:
        if ws in connections:
            connections.remove(ws)

# --- WebSocket Endpoint'leri ---
@app.websocket("/ws/admin")
async def websocket_admin_endpoint(websocket: WebSocket):
    """Admin paneli için WebSocket bağlantı noktası."""
    await websocket.accept()
    aktif_admin_websocketleri.append(websocket)
    client_host = websocket.client.host if websocket.client else "Bilinmeyen İstemci"
    logger.info(f"🔗 Admin WebSocket bağlandı: {client_host} (Toplam: {len(aktif_admin_websocketleri)})")
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                # Gelecekte admin'den gelen başka mesajlar işlenebilir
            except json.JSONDecodeError:
                logger.warning(f"⚠️ Admin WS ({client_host}): Geçersiz JSON alındı: {data}")
            except Exception as e:
                 logger.error(f"❌ Admin WS ({client_host}) Mesaj işleme hatası: {e}")
                 # Hata durumunda bağlantıyı kapatabiliriz
                 # break
    except WebSocketDisconnect:
        logger.info(f"🔌 Admin WebSocket bağlantısı kesildi: {client_host}")
    except Exception as e:
        logger.error(f"❌ Admin WebSocket hatası ({client_host}): {e}")
    finally:
        if websocket in aktif_admin_websocketleri:
            aktif_admin_websocketleri.remove(websocket)
        logger.info(f"📉 Admin WebSocket bağlantısı kaldırıldı: {client_host} (Kalan: {len(aktif_admin_websocketleri)})")


@app.websocket("/ws/mutfak")
async def websocket_mutfak_endpoint(websocket: WebSocket):
    """Mutfak ekranı (ve Masa Asistanı bildirimleri) için WebSocket bağlantı noktası."""
    await websocket.accept()
    aktif_mutfak_websocketleri.append(websocket)
    client_host = websocket.client.host if websocket.client else "Bilinmeyen İstemci"
    logger.info(f"🔗 Mutfak/Masa WebSocket bağlandı: {client_host} (Toplam: {len(aktif_mutfak_websocketleri)})")
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                # Mutfaktan gelen mesajları işle (örn: ping)
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                # Not: Mutfak ekranı genellikle sadece mesaj alır, göndermez.
                # Masa asistanı da bu kanala bağlanıp 'durum' mesajlarını alabilir.
            except json.JSONDecodeError:
                logger.warning(f"⚠️ Mutfak WS ({client_host}): Geçersiz JSON alındı: {data}")
            except Exception as e:
                 logger.error(f"❌ Mutfak WS ({client_host}) Mesaj işleme hatası: {e}")
                 # Hata durumunda bağlantıyı kapatabiliriz
                 # break
    except WebSocketDisconnect:
        logger.info(f"🔌 Mutfak/Masa WebSocket bağlantısı kesildi: {client_host}")
    except Exception as e:
        logger.error(f"❌ Mutfak/Masa WebSocket hatası ({client_host}): {e}")
    finally:
        if websocket in aktif_mutfak_websocketleri:
            aktif_mutfak_websocketleri.remove(websocket)
        logger.info(f"📉 Mutfak/Masa WebSocket bağlantısı kaldırıldı: {client_host} (Kalan: {len(aktif_mutfak_websocketleri)})")


# --- Masa Durumu Takibi ---
DB_NAME = "neso.db"
MENU_DB_NAME = "neso_menu.db"

async def update_table_status(masa_id: str, islem: str = "Erişim"):
    """Veritabanındaki masa durumunu günceller ve admin paneline bildirir."""
    now = datetime.now()
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO masa_durumlar (masa_id, son_erisim, aktif, son_islem)
                VALUES (?, ?, TRUE, ?)
                ON CONFLICT(masa_id) DO UPDATE SET
                    son_erisim = excluded.son_erisim,
                    aktif = excluded.aktif,
                    son_islem = CASE
                        WHEN excluded.son_islem IS NOT NULL THEN excluded.son_islem
                        ELSE son_islem
                    END
            """, (masa_id, now.strftime("%Y-%m-%d %H:%M:%S"), islem))
            conn.commit()
            logger.info(f"⏱️ Masa durumu güncellendi: Masa {masa_id}, İşlem: {islem}")

        # Admin paneline bildir (sadece aktif bağlantı varsa)
        if aktif_admin_websocketleri:
             await broadcast_message(aktif_admin_websocketleri, {
                 "type": "masa_durum",
                 "data": {
                     "masaId": masa_id,
                     "sonErisim": now.isoformat(), # ISO formatı daha standart
                     "aktif": True,
                     "sonIslem": islem
                 }
             })
             logger.info(f"📢 Masa durumu admin paneline bildirildi: Masa {masa_id}")

    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (masa durumu güncellenemedi): {e}")
    except Exception as e:
        logger.error(f"❌ Masa durumu güncelleme hatası: {e}")

# --- Aktif Kullanıcı Takibi Middleware ---
@app.middleware("http")
async def track_active_users(request: Request, call_next):
    """Gelen isteklerde masa ID'si varsa durumu günceller."""
    masa_id = request.path_params.get("masaId") # URL path'inden alır örn: /masa/{masaId}/..
    if not masa_id:
        # Alternatif olarak query parametresinden veya request body'den de alınabilir
        # masa_id = request.query_params.get("masa_id")
        pass

    if masa_id:
        # Her istekte güncellemek yerine belirli endpoint'lerde güncellemek daha verimli olabilir.
        # Şimdilik her masaId içeren istekte güncelliyoruz.
        # İşlem tipini request path'inden veya method'undan anlamlandırmaya çalışabiliriz.
        islem = f"{request.method} {request.url.path}"
        await update_table_status(masa_id, islem)

    response = await call_next(request)
    return response

# --- Aktif Masalar Endpoint ---
@app.get("/aktif-masalar")
async def get_active_tables_endpoint():
    """Son 5 dakika içinde aktif olan masaları döndürür."""
    try:
        active_time_limit = datetime.now() - timedelta(minutes=5)
        active_tables_data = []
        with sqlite3.connect(DB_NAME) as conn:
            conn.row_factory = sqlite3.Row # Sütun adlarıyla erişim için
            cursor = conn.cursor()
            cursor.execute("""
                SELECT masa_id, son_erisim, aktif, son_islem
                FROM masa_durumlar
                WHERE son_erisim >= ? AND aktif = TRUE
                ORDER BY son_erisim DESC
            """, (active_time_limit.strftime("%Y-%m-%d %H:%M:%S"),))
            results = cursor.fetchall()
            active_tables_data = [dict(row) for row in results] # dict listesine çevir

        logger.info(f"📊 Aktif masalar sorgulandı, {len(active_tables_data)} adet bulundu.")
        return {"tables": active_tables_data}
    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (aktif masalar alınamadı): {e}")
        raise HTTPException(status_code=500, detail="Aktif masa bilgileri alınırken veritabanı hatası oluştu.")
    except Exception as e:
        logger.error(f"❌ Aktif masalar alınırken hata: {e}")
        raise HTTPException(status_code=500, detail=f"Aktif masalar alınırken bir hata oluştu: {str(e)}")


# --- Admin Kimlik Doğrulama ---
def check_admin(credentials: HTTPBasicCredentials = Depends(security)):
    """Admin kimlik bilgilerini doğrular."""
    correct_username = os.getenv("ADMIN_USERNAME", "admin")
    correct_password = os.getenv("ADMIN_PASSWORD", "admin123")

    is_user_ok = credentials.username == correct_username
    is_pass_ok = credentials.password == correct_password

    if not (is_user_ok and is_pass_ok):
        logger.warning(f"🔒 Başarısız admin girişi denemesi: Kullanıcı adı '{credentials.username}'")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Geçersiz kimlik bilgileri",
            headers={"WWW-Authenticate": "Basic"},
        )
    # logger.info(f"🔑 Admin girişi başarılı: {credentials.username}") # Başarılı girişleri loglamak güvenlik riski olabilir.
    return True # Başarılı ise True döner

# --- Sipariş Yönetimi Endpoint'leri ---
@app.post("/siparis-ekle")
async def add_order_endpoint(data: dict = Body(...)):
    """Yeni bir sipariş ekler, veritabanına kaydeder ve ilgili kanallara yayınlar."""
    masa = data.get("masa")
    yanit = data.get("yanit") # AI yanıtı (loglama veya referans için)
    sepet_verisi = data.get("sepet", [])
    istek_orijinal = data.get("istek") # Kullanıcının orijinal isteği
    zaman_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info(f"📥 Yeni sipariş isteği alındı: Masa {masa}, Sepet: {len(sepet_verisi)} ürün")

    if not masa:
        logger.error("❌ Sipariş ekleme hatası: Masa bilgisi eksik.")
        raise HTTPException(status_code=400, detail="Masa bilgisi eksik.")
    if not sepet_verisi or not isinstance(sepet_verisi, list):
         logger.error(f"❌ Sipariş ekleme hatası (Masa {masa}): Sepet verisi eksik veya geçersiz format.")
         raise HTTPException(status_code=400, detail="Sepet verisi eksik veya geçersiz.")

    # İstek metnini oluştur (loglama ve db için özet)
    try:
        istek_ozet = ", ".join([f"{item.get('adet', 1)}x {item.get('urun', '').strip()}" for item in sepet_verisi])
    except Exception as e:
        logger.error(f"❌ Sipariş özeti oluşturma hatası (Masa {masa}): {e}")
        istek_ozet = "Detay alınamadı"

    try:
        sepet_json = json.dumps(sepet_verisi) # Sepeti JSON string olarak kaydet
        siparis_id = None
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO siparisler (masa, istek, yanit, sepet, zaman, durum)
                VALUES (?, ?, ?, ?, ?, 'bekliyor')
            """, (masa, istek_orijinal or istek_ozet, yanit, sepet_json, zaman_str)) # Orijinal istek varsa onu kaydet
            siparis_id = cursor.lastrowid # Eklenen siparişin ID'sini al
            conn.commit()
        logger.info(f"💾 Sipariş veritabanına kaydedildi: Masa {masa}, Sipariş ID: {siparis_id}")

        # Bildirim için sipariş bilgisini hazırla
        siparis_bilgisi = {
            "type": "siparis",
            "data": {
                "id": siparis_id, # Sipariş ID'sini de gönderelim
                "masa": masa,
                "istek": istek_orijinal or istek_ozet, # Frontend'in kullanması için
                "sepet": sepet_verisi, # Tam sepet verisi
                "zaman": zaman_str,
                "durum": "bekliyor"
            }
        }

        # Mutfağa ve Admin paneline bildir
        await broadcast_message(aktif_mutfak_websocketleri, siparis_bilgisi)
        await broadcast_message(aktif_admin_websocketleri, siparis_bilgisi)
        logger.info(f"📢 Yeni sipariş bildirimi gönderildi: Mutfak ({len(aktif_mutfak_websocketleri)}), Admin ({len(aktif_admin_websocketleri)})")

        # Masa durumunu güncelle
        await update_table_status(masa, f"Sipariş verdi ({len(sepet_verisi)} ürün)")

        return {"mesaj": "Sipariş başarıyla kaydedildi ve ilgili birimlere iletildi.", "siparisId": siparis_id}

    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (sipariş eklenemedi - Masa {masa}): {e}")
        raise HTTPException(status_code=500, detail=f"Sipariş veritabanına kaydedilirken hata oluştu: {str(e)}")
    except json.JSONDecodeError as e:
         logger.error(f"❌ JSON hatası (sipariş eklenemedi - Masa {masa}): Sepet verisi JSON'a çevrilemedi. {e}")
         raise HTTPException(status_code=400, detail=f"Sipariş verisi işlenirken hata oluştu: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Sipariş ekleme sırasında genel hata (Masa {masa}): {e}")
        raise HTTPException(status_code=500, detail=f"Sipariş eklenirken beklenmedik bir hata oluştu: {str(e)}")

@app.post("/siparis-guncelle")
async def update_order_status_endpoint(request: Request, auth: bool = Depends(check_admin)): # Yetkilendirme eklendi
    """Bir siparişin durumunu günceller ve ilgili kanallara yayınlar."""
    try:
        data = await request.json()
        masa = data.get("masa")
        durum = data.get("durum")
        siparis_id = data.get("id") # Opsiyonel: Belirli bir siparişi güncellemek için

        logger.info(f"🔄 Sipariş durumu güncelleme isteği: Masa {masa}, Yeni Durum: {durum}, ID: {siparis_id}")

        if not masa or not durum:
            logger.error("❌ Sipariş güncelleme hatası: Masa veya Durum bilgisi eksik.")
            raise HTTPException(status_code=400, detail="Masa ve durum bilgileri zorunludur.")

        valid_statuses = ["hazirlaniyor", "hazir", "iptal", "bekliyor"] # Geçerli durumlar
        if durum not in valid_statuses:
             logger.error(f"❌ Sipariş güncelleme hatası (Masa {masa}): Geçersiz durum '{durum}'.")
             raise HTTPException(status_code=400, detail=f"Geçersiz durum: {durum}. Geçerli durumlar: {valid_statuses}")

        rows_affected = 0
        updated_order_id = None
        try:
            with sqlite3.connect(DB_NAME) as conn:
                cursor = conn.cursor()
                if siparis_id:
                     # Belirli bir siparişi güncelle
                     cursor.execute("UPDATE siparisler SET durum = ? WHERE id = ?", (durum, siparis_id))
                     updated_order_id = siparis_id
                else:
                     # Belirtilen masanın en son 'hazir' veya 'iptal' olmayan siparişini güncelle
                     cursor.execute("""
                         UPDATE siparisler
                         SET durum = ?
                         WHERE id = (
                             SELECT id FROM siparisler
                             WHERE masa = ? AND durum NOT IN ('hazir', 'iptal')
                             ORDER BY id DESC LIMIT 1
                         )
                     """, (durum, masa))
                     # Güncellenen ID'yi almak için ek sorgu gerekebilir veya bu yaklaşım yeterli olabilir
                     # Şimdilik hangi ID'nin güncellendiğini loglamak zor.

                rows_affected = cursor.rowcount
                conn.commit()

            if rows_affected > 0:
                 logger.info(f"💾 Sipariş durumu veritabanında güncellendi: Masa {masa}, Durum: {durum}, Etkilenen: {rows_affected}")
                 # Başarılı güncelleme sonrası bildirim gönder
                 notification = {
                     "type": "durum",
                     "data": {
                         "id": updated_order_id, # Güncellenen ID'yi de gönderelim (varsa)
                         "masa": masa,
                         "durum": durum,
                         "zaman": datetime.now().isoformat()
                     }
                 }
                 # Mutfak, Admin ve ilgili Masa Asistanına bildir
                 await broadcast_message(aktif_mutfak_websocketleri, notification)
                 await broadcast_message(aktif_admin_websocketleri, notification)
                 logger.info(f"📢 Sipariş durum güncellemesi bildirildi: Masa {masa}, Durum: {durum}")

                 # Masa durumunu da güncelle
                 await update_table_status(masa, f"Sipariş durumu -> {durum}")

                 return {"success": True, "message": f"Sipariş durumu '{durum}' olarak güncellendi."}
            else:
                 logger.warning(f"⚠️ Sipariş durumu güncellenemedi (Masa {masa}, Durum: {durum}): Uygun sipariş bulunamadı veya zaten güncel.")
                 # Frontend'e neden güncellenmediği hakkında bilgi vermek daha iyi olabilir.
                 raise HTTPException(status_code=404, detail="Güncellenecek uygun sipariş bulunamadı veya durum zaten aynı.")

        except sqlite3.Error as e:
             logger.error(f"❌ Veritabanı hatası (sipariş durumu güncellenemedi - Masa {masa}): {e}")
             raise HTTPException(status_code=500, detail=f"Sipariş durumu güncellenirken veritabanı hatası oluştu: {str(e)}")

    except json.JSONDecodeError:
        logger.error("❌ Sipariş güncelleme hatası: İstek gövdesi JSON formatında değil.")
        raise HTTPException(status_code=400, detail="İstek gövdesi JSON formatında olmalıdır.")
    except Exception as e:
        logger.error(f"❌ Sipariş durumu güncelleme sırasında genel hata: {e}")
        raise HTTPException(status_code=500, detail=f"Sipariş durumu güncellenirken beklenmedik bir hata oluştu: {str(e)}")


@app.get("/siparisler")
def get_orders_endpoint(auth: bool = Depends(check_admin)):
    """Tüm siparişleri veritabanından çeker (Admin yetkisi gerektirir)."""
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.row_factory = sqlite3.Row # Sütun adlarıyla erişim için
            cursor = conn.cursor()
            # ID sütununu da seçelim
            cursor.execute("SELECT id, masa, istek, yanit, sepet, zaman, durum FROM siparisler ORDER BY id DESC")
            rows = cursor.fetchall()
            orders_data = [dict(row) for row in rows]
        logger.info(f" Görüntülenen sipariş sayısı: {len(orders_data)}")
        return {"orders": orders_data}
    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (siparişler alınamadı): {e}")
        raise HTTPException(status_code=500, detail="Siparişler alınırken veritabanı hatası oluştu.")
    except Exception as e:
        logger.error(f"❌ Siparişler alınırken hata: {e}")
        raise HTTPException(status_code=500, detail=f"Siparişler alınırken bir hata oluştu: {str(e)}")

# --- Veritabanı Başlatma ---
def init_db(db_path: str):
    """Belirtilen yoldaki ana veritabanı tablolarını oluşturur veya doğrular."""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Siparişler Tablosu
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS siparisler (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    masa TEXT NOT NULL,
                    istek TEXT,
                    yanit TEXT,
                    sepet TEXT,
                    zaman TEXT NOT NULL,
                    durum TEXT DEFAULT 'bekliyor' CHECK(durum IN ('bekliyor', 'hazirlaniyor', 'hazir', 'iptal'))
                )
            """)
            # Masa Durumları Tablosu
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS masa_durumlar (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    masa_id TEXT UNIQUE NOT NULL,
                    son_erisim TIMESTAMP NOT NULL,
                    aktif BOOLEAN DEFAULT TRUE,
                    son_islem TEXT
                )
            """)
            # İndeksler (Performans için eklenebilir)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_siparisler_masa_zaman ON siparisler(masa, zaman DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_masa_durumlar_erisim ON masa_durumlar(son_erisim DESC)")

            conn.commit()
            logger.info(f"✅ Ana veritabanı ({db_path}) başarıyla doğrulandı/oluşturuldu.")
    except sqlite3.Error as e:
        logger.critical(f"❌ KRİTİK HATA: Ana veritabanı ({db_path}) başlatılamadı! Hata: {e}")
        raise # Uygulamanın başlamaması için hatayı tekrar yükselt
    except Exception as e:
        logger.critical(f"❌ KRİTİK HATA: Veritabanı başlatılırken beklenmedik hata! Hata: {e}")
        raise

def init_menu_db(db_path: str):
    """Belirtilen yoldaki menü veritabanı tablolarını oluşturur veya doğrular."""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Kategoriler Tablosu
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS kategoriler (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    isim TEXT UNIQUE NOT NULL COLLATE NOCASE
                )
            """)
            # Menü Tablosu
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS menu (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ad TEXT NOT NULL COLLATE NOCASE,
                    fiyat REAL NOT NULL CHECK(fiyat >= 0),
                    kategori_id INTEGER NOT NULL,
                    stok_durumu INTEGER DEFAULT 1, /* 1: Var, 0: Yok (opsiyonel) */
                    FOREIGN KEY (kategori_id) REFERENCES kategoriler(id) ON DELETE CASCADE,
                    UNIQUE(ad, kategori_id) /* Aynı kategoride aynı isimde ürün olmasın */
                )
            """)
            # İndeksler
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_menu_kategori ON menu(kategori_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_menu_ad ON menu(ad)")

            conn.commit()
            logger.info(f"✅ Menü veritabanı ({db_path}) başarıyla doğrulandı/oluşturuldu.")
    except sqlite3.Error as e:
        logger.critical(f"❌ KRİTİK HATA: Menü veritabanı ({db_path}) başlatılamadı! Hata: {e}")
        raise
    except Exception as e:
        logger.critical(f"❌ KRİTİK HATA: Menü veritabanı başlatılırken beklenmedik hata! Hata: {e}")
        raise

# Veritabanlarını başlat
init_db(DB_NAME)
init_menu_db(MENU_DB_NAME)

# --- Menü Yönetimi Yardımcıları ---
def get_menu_for_prompt():
    """AI prompt'u için menüyü formatlar."""
    try:
        with sqlite3.connect(MENU_DB_NAME) as conn:
            cursor = conn.cursor()
            # Sadece stokta olan ürünleri alabiliriz (stok_durumu=1)
            cursor.execute("""
                SELECT k.isim AS kategori_adi, m.ad AS urun_adi
                FROM menu m
                JOIN kategoriler k ON m.kategori_id = k.id
                WHERE m.stok_durumu = 1
                ORDER BY k.isim, m.ad
            """)
            urunler = cursor.fetchall()

        if not urunler:
            logger.warning("⚠️ Menü boş veya stokta ürün yok. AI prompt'u eksik olabilir.")
            return "Üzgünüm, menü bilgisi şu anda mevcut değil."

        kategorili_menu = {}
        for kategori, urun in urunler:
            kategorili_menu.setdefault(kategori, []).append(urun)

        menu_aciklama = "\n".join([
            f"- {kategori}: {', '.join(urunler_listesi)}"
            for kategori, urunler_listesi in kategorili_menu.items()
        ])
        logger.info(f"📋 Menü prompt için hazırlandı ({len(urunler)} ürün).")
        return "Mevcut menümüz şöyledir:\n" + menu_aciklama
    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (menü prompt için alınamadı): {e}")
        return "Menü bilgisi alınırken bir sorun oluştu."
    except Exception as e:
        logger.error(f"❌ Menü prompt'u oluşturulurken hata: {e}")
        return "Menü bilgisi şu anda yüklenemedi."

def get_menu_price_dict():
    """Ürün adı -> fiyat eşleşmesini içeren bir sözlük döndürür (küçük harf)."""
    fiyatlar = {}
    try:
        with sqlite3.connect(MENU_DB_NAME) as conn:
            cursor = conn.cursor()
            # Stokta olmayanları da alabiliriz, belki fiyat sormak isteyebilirler.
            cursor.execute("SELECT LOWER(TRIM(ad)), fiyat FROM menu")
            veriler = cursor.fetchall()
            fiyatlar = {ad: fiyat for ad, fiyat in veriler}
        # logger.info(f"💰 Fiyat sözlüğü oluşturuldu ({len(fiyatlar)} ürün).") # Çok sık çağrılabilir, loglamayı azaltalım.
    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (fiyat sözlüğü alınamadı): {e}")
    except Exception as e:
        logger.error(f"❌ Fiyat sözlüğü oluşturulurken hata: {e}")
    return fiyatlar

# --- İstatistik Hesaplama ---
def calculate_statistics(order_data: list):
    """Verilen sipariş verilerinden toplam ürün sayısı ve geliri hesaplar."""
    total_items = 0
    total_revenue = 0.0
    price_dict = get_menu_price_dict()

    for (sepet_json_str,) in order_data:
        if not sepet_json_str:
            continue
        try:
            items_in_cart = json.loads(sepet_json_str)
            if not isinstance(items_in_cart, list):
                logger.warning(f"⚠️ İstatistik: Geçersiz sepet formatı (liste değil): {sepet_json_str}")
                continue

            for item in items_in_cart:
                if not isinstance(item, dict):
                    logger.warning(f"⚠️ İstatistik: Geçersiz ürün formatı (sözlük değil): {item}")
                    continue
                quantity = item.get("adet", 1)
                item_name = str(item.get("urun", "")).lower().strip()
                # Fiyatı doğrudan sepetten almak yerine güncel fiyat listesinden alalım
                price = price_dict.get(item_name, 0.0) # Fiyat bulunamazsa 0 kabul et

                if isinstance(quantity, (int, float)) and quantity > 0:
                    total_items += quantity
                    total_revenue += quantity * price
                else:
                    logger.warning(f"⚠️ İstatistik: Geçersiz adet ({quantity}) veya fiyat ({price}) ürünü: {item_name}")

        except json.JSONDecodeError:
            logger.warning(f"⚠️ İstatistik: Geçersiz JSON sepet verisi: {sepet_json_str}")
        except Exception as e:
            logger.error(f"❌ İstatistik hesaplama hatası (sepet işlenirken): {e}")
            continue # Hatalı sepeti atla, devam et

    return total_items, round(total_revenue, 2)

# --- AI Yanıt Üretme ---
SISTEM_MESAJI_ICERIK = (
    "Sen, Gaziantep'teki Fıstık Kafe için özel olarak tasarlanmış, Neso adında bir sipariş asistanısın. "
    "Görevin, masadaki müşterilerin sesli veya yazılı taleplerini anlayıp menüdeki ürünlerle eşleştirerek siparişlerini almak ve bu siparişleri mutfağa doğru bir şekilde iletmektir. "
    "Siparişleri sen hazırlamıyorsun, sadece alıyorsun. "
    "Her zaman nazik, yardımsever, samimi ve çözüm odaklı olmalısın. Gaziantep ağzıyla veya şivesiyle konuşmamalısın, standart ve kibar bir Türkçe kullanmalısın. "
    "Müşterinin ne istediğini tam anlayamazsan, soruyu tekrar sormaktan veya seçenekleri netleştirmesini istemekten çekinme. "
    "Sipariş tamamlandığında veya müşteri teşekkür ettiğinde 'Afiyet olsun!' demeyi unutma.\n\n"
    f"{get_menu_for_prompt()}" # Menüyü dinamik olarak ekle
)

SYSTEM_PROMPT = {"role": "system", "content": SISTEM_MESAJI_ICERIK}

@app.post("/yanitla")
async def handle_message_endpoint(data: dict = Body(...)):
    """Kullanıcı mesajını alır, AI'dan yanıt üretir."""
    user_message = data.get("text", "")
    table_id = data.get("masa", "bilinmiyor")
    language = data.get("language", "tr-TR") # Dil bilgisi (gelecekte kullanılabilir)

    if not user_message:
        logger.warning(f"⚠️ Boş mesaj alındı: Masa {table_id}")
        raise HTTPException(status_code=400, detail="Mesaj içeriği boş olamaz.")

    logger.info(f"💬 Mesaj alındı: Masa {table_id}, Dil: {language}, Mesaj: '{user_message[:50]}...'")

    try:
        # OpenAI istemcisi başlatılmamışsa hata ver
        if not openai_client:
             logger.error("❌ OpenAI istemcisi başlatılmadığı için yanıt üretilemiyor.")
             raise HTTPException(status_code=503, detail="Yapay zeka hizmeti şu anda kullanılamıyor.")

        # Mesaj geçmişini de dahil edebiliriz (opsiyonel)
        messages_for_api = [
            SYSTEM_PROMPT,
            {"role": "user", "content": user_message}
        ]

        # OpenAI API çağrısı
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo", # veya "gpt-4" vb.
            messages=messages_for_api,
            temperature=0.6, # Biraz daha tutarlı yanıtlar için düşürüldü
            max_tokens=150 # Yanıt uzunluğunu sınırlayalım
        )

        ai_reply = response.choices[0].message.content.strip()
        logger.info(f"🤖 AI yanıtı üretildi: Masa {table_id}, Yanıt: '{ai_reply[:50]}...'")

        return {"reply": ai_reply}

    except Exception as e:
        logger.error(f"❌ AI yanıtı üretme hatası (Masa {table_id}): {e}")
        # Kullanıcıya daha genel bir hata mesajı gösterilebilir
        raise HTTPException(status_code=500, detail=f"Yapay zeka yanıtı alınırken bir sorun oluştu: {str(e)}")


# --- Menü Yönetimi Endpoint'leri ---
@app.get("/menu")
def get_full_menu_endpoint():
    """Tüm menüyü kategorilere göre gruplanmış olarak döndürür."""
    try:
        full_menu_data = []
        with sqlite3.connect(MENU_DB_NAME) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT id, isim FROM kategoriler ORDER BY isim")
            kategoriler = cursor.fetchall()

            for kategori_row in kategoriler:
                cursor.execute("""
                    SELECT ad, fiyat, stok_durumu
                    FROM menu
                    WHERE kategori_id = ?
                    ORDER BY ad
                """, (kategori_row['id'],))
                urunler_rows = cursor.fetchall()
                full_menu_data.append({
                    "kategori": kategori_row['isim'],
                    "urunler": [dict(urun) for urun in urunler_rows]
                })
        logger.info(f"📋 Menü sorgulandı, {len(full_menu_data)} kategori bulundu.")
        return {"menu": full_menu_data}
    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (menü alınamadı): {e}")
        raise HTTPException(status_code=500, detail="Menü bilgileri alınırken veritabanı hatası oluştu.")
    except Exception as e:
        logger.error(f"❌ Menü alınırken hata: {e}")
        raise HTTPException(status_code=500, detail=f"Menü bilgileri alınırken bir hata oluştu: {str(e)}")

# Not: CSV Yükleme endpoint'i önceki kodda mevcuttu, gerekirse tekrar eklenebilir.
# @app.post("/menu-yukle-csv") ...

@app.post("/menu/ekle", status_code=status.HTTP_201_CREATED)
async def add_menu_item_endpoint(item_data: dict = Body(...), auth: bool = Depends(check_admin)):
    """Yeni bir menü öğesi ekler (Admin yetkisi gerektirir)."""
    item_name = item_data.get("ad", "").strip()
    item_price_str = item_data.get("fiyat")
    item_category = item_data.get("kategori", "").strip()

    logger.info(f"➕ Menüye ekleme isteği: Ad: {item_name}, Fiyat: {item_price_str}, Kategori: {item_category}")

    if not item_name or not item_category or item_price_str is None:
        logger.error("❌ Menü ekleme hatası: Ürün adı, fiyat ve kategori zorunludur.")
        raise HTTPException(status_code=400, detail="Ürün adı, fiyat ve kategori zorunludur.")

    try:
        item_price = float(item_price_str)
        if item_price < 0:
             raise ValueError("Fiyat negatif olamaz.")
    except ValueError:
        logger.error(f"❌ Menü ekleme hatası: Geçersiz fiyat formatı '{item_price_str}'.")
        raise HTTPException(status_code=400, detail="Geçersiz fiyat formatı. Sayısal bir değer girin.")

    try:
        with sqlite3.connect(MENU_DB_NAME) as conn:
            cursor = conn.cursor()
            # Kategoriyi bul veya ekle
            cursor.execute("INSERT OR IGNORE INTO kategoriler (isim) VALUES (?)", (item_category,))
            cursor.execute("SELECT id FROM kategoriler WHERE isim = ?", (item_category,))
            category_result = cursor.fetchone()
            if not category_result:
                 # Bu durum INSERT OR IGNORE sonrası olmamalı ama garantiye alalım
                 logger.error(f"❌ Menü ekleme hatası: Kategori ID alınamadı '{item_category}'.")
                 raise HTTPException(status_code=500, detail="Kategori işlenirken hata oluştu.")
            category_id = category_result[0]

            # Menü öğesini ekle (stok durumu varsayılan 1)
            cursor.execute("""
                INSERT INTO menu (ad, fiyat, kategori_id, stok_durumu)
                VALUES (?, ?, ?, 1)
            """, (item_name, item_price, category_id))
            conn.commit()
            item_id = cursor.lastrowid
        logger.info(f"💾 Menü öğesi başarıyla eklendi: ID {item_id}, Ad: {item_name}")
        # Başarılı ekleme sonrası sistem mesajını güncelleyebiliriz (opsiyonel)
        # global SISTEM_MESAJI_ICERIK, SYSTEM_PROMPT
        # SISTEM_MESAJI_ICERIK = SISTEM_MESAJI_ICERIK.split("\n\n")[0] + "\n\n" + get_menu_for_prompt()
        # SYSTEM_PROMPT = {"role": "system", "content": SISTEM_MESAJI_ICERIK}
        return {"mesaj": f"'{item_name}' menüye başarıyla eklendi.", "itemId": item_id}

    except sqlite3.IntegrityError as e:
         # UNIQUE constraint hatası (aynı kategoride aynı isim)
         logger.warning(f"⚠️ Menü ekleme hatası (IntegrityError): '{item_name}' zaten '{item_category}' kategorisinde mevcut olabilir. Hata: {e}")
         raise HTTPException(status_code=409, detail=f"'{item_name}' ürünü '{item_category}' kategorisinde zaten mevcut.")
    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (menü öğesi eklenemedi): {e}")
        raise HTTPException(status_code=500, detail="Menü öğesi eklenirken veritabanı hatası oluştu.")
    except Exception as e:
        logger.error(f"❌ Menü öğesi eklenirken genel hata: {e}")
        raise HTTPException(status_code=500, detail="Menü öğesi eklenirken beklenmedik bir hata oluştu.")

@app.delete("/menu/sil")
async def delete_menu_item_endpoint(urun_adi: str = Query(...), auth: bool = Depends(check_admin)):
    """Belirtilen addaki menü öğesini siler (Admin yetkisi gerektirir)."""
    item_name_to_delete = urun_adi.strip()
    logger.info(f"➖ Menüden silme isteği: Ad: {item_name_to_delete}")

    if not item_name_to_delete:
         logger.error("❌ Menü silme hatası: Silinecek ürün adı belirtilmemiş.")
         raise HTTPException(status_code=400, detail="Silinecek ürün adı boş olamaz.")

    try:
        with sqlite3.connect(MENU_DB_NAME) as conn:
            cursor = conn.cursor()
            # İsme göre sil (Tüm kategorilerdeki eşleşmeleri siler - dikkat!)
            # Eğer sadece belirli bir kategoriden silmek istenirse, kategori ID'si de alınmalı.
            cursor.execute("DELETE FROM menu WHERE ad = ?", (item_name_to_delete,))
            rows_affected = cursor.rowcount
            conn.commit()

        if rows_affected > 0:
            logger.info(f"💾 Menü öğesi silindi: Ad: {item_name_to_delete}, Etkilenen: {rows_affected}")
            # Sistem mesajını güncelle (opsiyonel)
            # global SISTEM_MESAJI_ICERIK, SYSTEM_PROMPT
            # SISTEM_MESAJI_ICERIK = SISTEM_MESAJI_ICERIK.split("\n\n")[0] + "\n\n" + get_menu_for_prompt()
            # SYSTEM_PROMPT = {"role": "system", "content": SISTEM_MESAJI_ICERIK}
            return {"mesaj": f"'{item_name_to_delete}' isimli ürün(ler) menüden başarıyla silindi."}
        else:
            logger.warning(f"⚠️ Menü silme: '{item_name_to_delete}' adında ürün bulunamadı.")
            raise HTTPException(status_code=404, detail=f"'{item_name_to_delete}' adında ürün menüde bulunamadı.")

    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (menü öğesi silinemedi): {e}")
        raise HTTPException(status_code=500, detail="Menü öğesi silinirken veritabanı hatası oluştu.")
    except Exception as e:
        logger.error(f"❌ Menü öğesi silinirken genel hata: {e}")
        raise HTTPException(status_code=500, detail="Menü öğesi silinirken beklenmedik bir hata oluştu.")


# --- İstatistik Endpoint'leri ---
@app.get("/istatistik/en-cok-satilan")
def get_popular_items_endpoint():
    """En çok satan ilk 5 ürünü döndürür."""
    try:
        item_counts = {}
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            # Sadece durumu 'iptal' olmayan siparişleri dahil edelim mi? Opsiyonel.
            cursor.execute("SELECT sepet FROM siparisler WHERE durum != 'iptal'")
            all_carts_json = cursor.fetchall()

        logger.info(f"📊 Popüler ürünler hesaplanıyor ({len(all_carts_json)} sipariş üzerinden)...")

        for (sepet_json_str,) in all_carts_json:
            if not sepet_json_str:
                continue
            try:
                items_in_cart = json.loads(sepet_json_str)
                if not isinstance(items_in_cart, list): continue

                for item in items_in_cart:
                     if not isinstance(item, dict): continue
                     item_name = item.get("urun")
                     quantity = item.get("adet", 1)
                     if item_name and isinstance(quantity, (int, float)) and quantity > 0:
                         item_counts[item_name] = item_counts.get(item_name, 0) + quantity
            except json.JSONDecodeError:
                 logger.warning(f"⚠️ Popüler ürünler: Geçersiz JSON sepet: {sepet_json_str[:100]}...")
            except Exception as e:
                logger.error(f"❌ Popüler ürünler hesaplanırken sepet işleme hatası: {e}")

        # Sayaca göre sırala ve ilk 5'i al
        sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        popular_items_data = [{"urun": item, "adet": count} for item, count in sorted_items]
        logger.info(f"🏆 En çok satanlar: {popular_items_data}")
        return popular_items_data

    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (popüler ürünler alınamadı): {e}")
        raise HTTPException(status_code=500, detail="Popüler ürünler alınırken veritabanı hatası oluştu.")
    except Exception as e:
        logger.error(f"❌ Popüler ürünler hesaplanırken genel hata: {e}")
        raise HTTPException(status_code=500, detail=f"Popüler ürünler hesaplanırken bir hata oluştu: {str(e)}")

@app.get("/istatistik/gunluk")
def get_daily_stats_endpoint():
    """Bugünün sipariş istatistiklerini (toplam ürün, gelir) döndürür."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            # Sadece bugünün ve durumu 'iptal' olmayan siparişlerinin sepetlerini al
            cursor.execute("SELECT sepet FROM siparisler WHERE zaman LIKE ? AND durum != 'iptal'", (f"{today_str}%",))
            daily_data = cursor.fetchall()

        logger.info(f"📊 Günlük istatistik hesaplanıyor ({len(daily_data)} sipariş)...")
        total_items, total_revenue = calculate_statistics(daily_data)
        logger.info(f"📅 Günlük Sonuç ({today_str}): {total_items} ürün, {total_revenue} TL")
        return {"tarih": today_str, "siparis_sayisi": total_items, "gelir": total_revenue} # siparis_sayisi yerine toplam ürün sayısı daha doğru olabilir

    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (günlük istatistik alınamadı): {e}")
        raise HTTPException(status_code=500, detail="Günlük istatistikler alınırken veritabanı hatası oluştu.")
    except Exception as e:
        logger.error(f"❌ Günlük istatistik hesaplanırken genel hata: {e}")
        raise HTTPException(status_code=500, detail=f"Günlük istatistikler hesaplanırken bir hata oluştu: {str(e)}")

@app.get("/istatistik/aylik")
def get_monthly_stats_endpoint():
    """Son 30 günün sipariş istatistiklerini döndürür."""
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            # Son 30 günün ve durumu 'iptal' olmayan siparişlerinin sepetlerini al
            cursor.execute("SELECT sepet FROM siparisler WHERE zaman >= ? AND durum != 'iptal'", (start_date,))
            monthly_data = cursor.fetchall()

        logger.info(f"📊 Aylık istatistik hesaplanıyor ({len(monthly_data)} sipariş, {start_date} sonrası)...")
        total_items, total_revenue = calculate_statistics(monthly_data)
        logger.info(f"🗓️ Aylık Sonuç ({start_date}-Bugün): {total_items} ürün, {total_revenue} TL")
        return {"baslangic": start_date, "siparis_sayisi": total_items, "gelir": total_revenue}

    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (aylık istatistik alınamadı): {e}")
        raise HTTPException(status_code=500, detail="Aylık istatistikler alınırken veritabanı hatası oluştu.")
    except Exception as e:
        logger.error(f"❌ Aylık istatistik hesaplanırken genel hata: {e}")
        raise HTTPException(status_code=500, detail=f"Aylık istatistikler hesaplanırken bir hata oluştu: {str(e)}")

@app.get("/istatistik/yillik")
def get_yearly_stats_endpoint():
    """Tüm zamanlardaki siparişleri aylara göre gruplayıp toplam ürün sayısını döndürür."""
    try:
        monthly_item_counts = {}
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            # Durumu 'iptal' olmayan tüm siparişlerin zaman ve sepet bilgilerini al
            cursor.execute("SELECT zaman, sepet FROM siparisler WHERE durum != 'iptal'")
            all_data = cursor.fetchall()

        logger.info(f"📊 Yıllık (ay bazında) istatistik hesaplanıyor ({len(all_data)} sipariş)...")

        for time_str, cart_json_str in all_data:
            if not cart_json_str or not time_str: continue
            try:
                # Ay bilgisini al (YYYY-MM)
                month_key = time_str[:7]
                items_in_cart = json.loads(cart_json_str)
                if not isinstance(items_in_cart, list): continue

                month_total = 0
                for item in items_in_cart:
                     if not isinstance(item, dict): continue
                     quantity = item.get("adet", 1)
                     if isinstance(quantity, (int, float)) and quantity > 0:
                         month_total += quantity

                monthly_item_counts[month_key] = monthly_item_counts.get(month_key, 0) + month_total

            except json.JSONDecodeError:
                logger.warning(f"⚠️ Yıllık ist.: Geçersiz JSON sepet: {cart_json_str[:100]}...")
            except Exception as e:
                logger.error(f"❌ Yıllık istatistik hesaplanırken hata (sipariş işlenirken): {e}")

        # Aylara göre sıralı döndür
        sorted_monthly_data = dict(sorted(monthly_item_counts.items()))
        logger.info(f"📅 Yıllık Sonuç (Ay Bazında): {len(sorted_monthly_data)} ay verisi bulundu.")
        # Frontend'in Recharts ile uyumlu olması için formatı değiştirebiliriz:
        # formatted_data = [{"tarih": ay, "adet": adet} for ay, adet in sorted_monthly_data.items()]
        # return formatted_data
        return sorted_monthly_data # Şimdilik orijinal formatta bırakalım

    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (yıllık istatistik alınamadı): {e}")
        raise HTTPException(status_code=500, detail="Yıllık istatistikler alınırken veritabanı hatası oluştu.")
    except Exception as e:
        logger.error(f"❌ Yıllık istatistik hesaplanırken genel hata: {e}")
        raise HTTPException(status_code=500, detail=f"Yıllık istatistikler hesaplanırken bir hata oluştu: {str(e)}")

@app.get("/istatistik/filtreli")
def get_filtered_stats_endpoint(baslangic: str = Query(...), bitis: str = Query(...)):
    """Belirtilen tarih aralığındaki sipariş istatistiklerini döndürür."""
    # Tarih formatını doğrula (YYYY-MM-DD) - Basit kontrol
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    if not date_pattern.match(baslangic) or not date_pattern.match(bitis):
         logger.error(f"❌ Filtreli istatistik: Geçersiz tarih formatı. Başlangıç: {baslangic}, Bitiş: {bitis}")
         raise HTTPException(status_code=400, detail="Tarih formatı YYYY-MM-DD şeklinde olmalıdır.")

    # Bitiş tarihine bir gün ekleyerek o günün tamamını dahil et
    try:
        end_date_inclusive = (datetime.strptime(bitis, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    except ValueError:
         logger.error(f"❌ Filtreli istatistik: Geçersiz bitiş tarihi değeri: {bitis}")
         raise HTTPException(status_code=400, detail="Geçersiz bitiş tarihi değeri.")


    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            # Belirtilen aralıktaki ve durumu 'iptal' olmayan siparişlerin sepetlerini al
            cursor.execute("""
                SELECT sepet FROM siparisler
                WHERE zaman >= ? AND zaman < ? AND durum != 'iptal'
            """, (baslangic, end_date_inclusive))
            filtered_data = cursor.fetchall()

        logger.info(f"📊 Filtreli istatistik hesaplanıyor ({len(filtered_data)} sipariş, {baslangic} - {bitis})...")
        total_items, total_revenue = calculate_statistics(filtered_data)
        logger.info(f"📅 Filtreli Sonuç ({baslangic} - {bitis}): {total_items} ürün, {total_revenue} TL")
        return {"aralik": f"{baslangic} → {bitis}", "siparis_sayisi": total_items, "gelir": total_revenue}

    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası (filtreli istatistik alınamadı): {e}")
        raise HTTPException(status_code=500, detail="Filtreli istatistikler alınırken veritabanı hatası oluştu.")
    except Exception as e:
        logger.error(f"❌ Filtreli istatistik hesaplanırken genel hata: {e}")
        raise HTTPException(status_code=500, detail=f"Filtreli istatistikler hesaplanırken bir hata oluştu: {str(e)}")


# --- Sesli Yanıt Endpoint ---
@app.post("/sesli-yanit")
async def generate_speech_endpoint(data: dict = Body(...)):
    """Verilen metni Google TTS kullanarak ses dosyasına dönüştürür."""
    text_to_speak = data.get("text", "")
    language_code = data.get("language", "tr-TR") # Dil kodu (gelecekte kullanılabilir)

    if not text_to_speak or not isinstance(text_to_speak, str):
        logger.error("❌ Sesli yanıt hatası: Metin içeriği eksik veya geçersiz.")
        raise HTTPException(status_code=400, detail="Seslendirilecek metin eksik veya geçersiz.")

    # TTS istemcisi başlatılmamışsa hata ver
    if not tts_client:
         logger.error("❌ Google TTS istemcisi başlatılmadığı için sesli yanıt üretilemiyor.")
         raise HTTPException(status_code=503, detail="Sesli yanıt hizmeti şu anda kullanılamıyor.")

    try:
        # Emojileri temizle (TTS bunları okuyamaz)
        cleaned_text = temizle_emoji(text_to_speak)
        if not cleaned_text.strip():
            logger.warning("⚠️ Sesli yanıt: Temizlendikten sonra metin boş kaldı.")
            # Boş ses dosyası döndürmek yerine belki 204 No Content dönebiliriz?
            # Şimdilik kısa bir sessizlik döndürelim veya hata verelim.
            raise HTTPException(status_code=400, detail="Seslendirilecek geçerli metin bulunamadı.")


        logger.info(f"🗣️ Sesli yanıt isteği: Dil: {language_code}, Metin: '{cleaned_text[:50]}...'")

        synthesis_input = texttospeech.SynthesisInput(text=cleaned_text)
        # Ses seçimi (Türkçe Kadın varsayılan, gelecekte değiştirilebilir)
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code, # Dinamik dil kodu
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE # veya NEUTRAL
            # name='tr-TR-Wavenet-A' # Daha spesifik bir ses seçilebilir
        )
        # Ses yapılandırması (MP3 formatı, normal konuşma hızı)
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0 # Hız normale çekildi (önceki 1.3 idi)
            # pitch = 0 # Tonlama ayarı (varsayılan)
            # volume_gain_db = 0 # Ses yüksekliği ayarı (varsayılan)
        )

        # Google Cloud TTS API çağrısı
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        logger.info("✅ Sesli yanıt başarıyla oluşturuldu.")
        # MP3 içeriğini doğrudan yanıt olarak döndür
        return Response(content=response.audio_content, media_type="audio/mpeg")

    except HTTPException as http_err:
        # Kendi fırlattığımız hataları tekrar fırlat
        raise http_err
    except Exception as e:
        logger.error(f"❌ Sesli yanıt üretme hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Sesli yanıt oluşturulurken beklenmedik bir hata oluştu: {str(e)}")


# --- Kaldırılan/Yorum Satırı Yapılan Endpoint'ler ---

# @app.get("/istatistik/online")
# def online_kullanici_sayisi():
#     # Not: Bu endpoint, güncellenmeyen global 'aktif_kullanicilar' sözlüğünü kullanıyordu.
#     # Bunun yerine '/aktif-masalar' endpoint'i veritabanından güncel veriyi çeker.
#     # Bu nedenle bu endpoint kaldırıldı/yorum satırı yapıldı.
#     # su_an = datetime.now()
#     # aktifler = [kimlik for kimlik, zaman in aktif_kullanicilar.items() if (su_an - zaman).seconds < 300]
#     # return {"count": len(aktifler)}
#     logger.warning("⚠️ /istatistik/online endpoint'i kullanımdan kaldırıldı. /aktif-masalar kullanılmalıdır.")
#     raise HTTPException(status_code=410, detail="Bu endpoint kullanımdan kaldırıldı. /aktif-masalar kullanın.")


# @app.api_route("/siparisler/ornek", methods=["GET", "POST"])
# def ornek_siparis_ekle():
#     # Test için kullanılan örnek sipariş ekleme endpoint'i.
#     # İstenirse tekrar aktif edilebilir, ancak GET metodu yerine sadece POST olmalı.
#     if request.method == 'GET':
#          raise HTTPException(status_code=405, detail="Method Not Allowed. Use POST.")
#     try:
#         # ... (Örnek sipariş ekleme kodu) ...
#         logger.info("✅ Örnek sipariş başarıyla eklendi.")
#         return {"mesaj": "✅ Örnek sipariş başarıyla eklendi."}
#     except Exception as e:
#         logger.error(f"❌ Örnek sipariş ekleme hatası: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
#     pass # Şimdilik pasif


# --- Uygulama Kapatma Olayı ---
@app.on_event("shutdown")
def shutdown_event():
    """Uygulama kapatılırken kaynakları temizler."""
    logger.info("🚪 Uygulama kapatılıyor...")
    # Geçici Google kimlik bilgisi dosyasını sil
    if google_creds_path and os.path.exists(google_creds_path):
        try:
            os.remove(google_creds_path)
            logger.info("✅ Geçici Google kimlik bilgisi dosyası silindi.")
        except OSError as e:
            logger.error(f"❌ Geçici Google kimlik bilgisi dosyası silinemedi: {e}")
    # Aktif WebSocket bağlantılarını kapatmayı deneyebiliriz (genellikle ASGI sunucusu halleder)
    # for ws in aktif_admin_websocketleri[:]: await ws.close(code=status.WS_1001_GOING_AWAY)
    # for ws in aktif_mutfak_websocketleri[:]: await ws.close(code=status.WS_1001_GOING_AWAY)
    logger.info("👋 Uygulama kapatıldı.")


# --- Ana Çalıştırma Bloğu (Geliştirme için) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 FastAPI uygulaması geliştirme modunda başlatılıyor...")
    # Ortam değişkeninden port al, yoksa varsayılan kullan
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=port, log_level="info")