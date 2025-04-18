import sqlite3
from pathlib import Path

class StatusDB:
    def __init__(self, db_path="ocr_status.sqlite"):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS status (
                    doc_id TEXT PRIMARY KEY,
                    status TEXT CHECK(status IN ('pending', 'success', 'failed')),
                    notes TEXT
                )
            """)

    def set_status(self, doc_id, status, notes=None):
        with self.conn:
            self.conn.execute("""
                INSERT INTO status (doc_id, status, notes)
                VALUES (?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET status=excluded.status, notes=excluded.notes
            """, (doc_id, status, notes))

    def get_status(self, doc_id):
        cur = self.conn.cursor()
        cur.execute("SELECT status FROM status WHERE doc_id = ?", (doc_id,))
        result = cur.fetchone()
        return result[0] if result else None

    def get_unprocessed(self):
        cur = self.conn.cursor()
        cur.execute("SELECT doc_id FROM status WHERE status != 'success' OR status IS NULL")
        return [row[0] for row in cur.fetchall()]

    def close(self):
        self.conn.close()
