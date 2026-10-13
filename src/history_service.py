
import json
import sqlite3
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "prediction_history.db"


def get_connection():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                disease TEXT NOT NULL,
                prediction TEXT NOT NULL,
                probabilities TEXT,
                input_data TEXT
            )
        """)


def save_prediction(disease, inputs, result):
    initialize_database()

    probabilities = result.get("probability")

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO prediction_history
            (timestamp, disease, prediction, probabilities, input_data)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                datetime.now().astimezone().isoformat(
                    timespec="seconds"
                ),
                str(disease),
                str(result["prediction"]),
                json.dumps(probabilities),
                json.dumps(inputs, default=str),
            ),
        )


def get_prediction_history():
    initialize_database()

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, timestamp, disease, prediction,
                   probabilities, input_data
            FROM prediction_history
            ORDER BY id DESC
            """
        ).fetchall()

    history = []

    for row in rows:
        item = dict(row)
        item["probabilities"] = json.loads(
            item["probabilities"] or "null"
        )
        item["input_data"] = json.loads(
            item["input_data"] or "{}"
        )
        history.append(item)

    return history


def clear_prediction_history():
    initialize_database()

    with get_connection() as connection:
        connection.execute("DELETE FROM prediction_history")
