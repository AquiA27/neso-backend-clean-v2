import sqlite3
from datetime import datetime

# Veritabanı bağlantısı
def init_memory_db():
    conn = sqlite3.connect("neso.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            masa TEXT,
            role TEXT,
            content TEXT,
            zaman TEXT
        )
    """)
    conn.commit()
    conn.close()

init_memory_db()

# Geçmişi al (en fazla 6 satır)
def get_memory(masa):
    conn = sqlite3.connect("neso.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT role, content FROM memory
        WHERE masa = ?
        ORDER BY zaman DESC
        LIMIT 6
    """, (masa,))
    rows = cursor.fetchall()
    conn.close()

    # Ters çeviriyoruz ki en eski mesaj ilk sırada olsun
    messages = [{"role": role, "content": content} for role, content in reversed(rows)]
    return messages

# Hafızaya yeni içerik ekle
def add_to_memory(masa, role, content):
    conn = sqlite3.connect("neso.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO memory (masa, role, content, zaman)
        VALUES (?, ?, ?, ?)
    """, (masa, role, content, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
