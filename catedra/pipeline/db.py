"""
Módulo de conexión a la base de datos.

Uso en notebooks:
    from db import engine, DB_URL
    df = pd.read_sql("SELECT 1", engine)

Los parámetros se leen de .env (o .env.local si existe).
Para usar producción, comenta/descomenta las variables en .env.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Carga .env.local primero (sobrescribe), luego .env
_base = Path(__file__).parent
load_dotenv(_base / ".env.local", override=True)
load_dotenv(_base / ".env", override=False)

DB_HOST     = os.environ["DB_HOST"]
DB_PORT     = os.environ["DB_PORT"]
DB_NAME     = os.environ["DB_NAME"]
DB_USER     = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]

DB_URL = (
    f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    f"?charset=utf8mb4"
)

engine = create_engine(
    DB_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    connect_args={"collation": "utf8mb4_unicode_ci"},
)
