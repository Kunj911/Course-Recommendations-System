"""
Database Setup Script
=====================
Initializes the SQLite database by executing the schema.sql file.
Creates the database directory and file if they don't exist.
"""

import sqlite3
import os

# Resolve paths relative to this script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "course_recommendation.db")
SCHEMA_PATH = os.path.join(DB_DIR, "schema.sql")


def setup_database() -> str:
    """Create the database and apply the schema.

    Returns:
        str: The absolute path to the created database file.
    """
    os.makedirs(DB_DIR, exist_ok=True)

    # Remove stale database so we start from a clean state
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"[RESET] Removed existing database at {DB_PATH}")

    with open(SCHEMA_PATH, "r") as f:
        schema_sql = f.read()

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(schema_sql)
    conn.commit()
    conn.close()

    print(f"[OK] Database created successfully at {DB_PATH}")
    return DB_PATH


if __name__ == "__main__":
    setup_database()
