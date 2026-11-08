"""SQLite access with schema-owned tables and per-connection foreign keys."""
import os, sqlite3
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DB_PATH=Path(os.getenv("PHOTO_ADVISOR_DB", ROOT/"backend/data/photo_advisor.sqlite3"))
STORAGE=Path(os.getenv("PHOTO_ADVISOR_STORAGE", ROOT/"backend/data/files"))
SCHEMA=ROOT/"backend/db/schema.sql"

def connect():
    DB_PATH.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(DB_PATH); db.row_factory=sqlite3.Row; db.execute("PRAGMA foreign_keys=ON")
    db.executescript(SCHEMA.read_text()); return db
