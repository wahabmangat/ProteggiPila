import os
import psycopg2
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

# Folder to store uploaded images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class Database:
    def __init__(self):
        """Initialize the database: create DB and table if they don't exist."""
        self.ensure_database()
        self.conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST
        )
        self.cur = self.conn.cursor()
        self.ensure_table()

    def ensure_database(self):
        """Check if the database exists, create if not."""
        conn = psycopg2.connect(
            dbname="postgres", user=DB_USER, password=DB_PASS, host=DB_HOST
        )
        conn.autocommit = True
        cur = conn.cursor()

        cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'")
        exists = cur.fetchone()
        if not exists:
            cur.execute(f"CREATE DATABASE {DB_NAME};")
            print(f"Database '{DB_NAME}' created successfully.")

        cur.close()
        conn.close()

    def ensure_table(self):
        """Create table if it doesn't exist."""
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS video_frames (
            id SERIAL PRIMARY KEY,
            captured_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP NOT NULL,
            image_path VARCHAR(100) NOT NULL,
            processed_at TIMESTAMP DEFAULT NULL,
            is_box BOOLEAN DEFAULT NULL,
            above_pallet_thresh BOOLEAN DEFAULT NULL,
            status VARCHAR(50) DEFAULT 'pending' CHECK(status IN ('pending', 'processing', 'success', 'failed')),
            prediction_confidence_score FLOAT DEFAULT NULL
        )
        """)
        self.conn.commit()
    
    def get_video_frame(self, frame_id: int):
        """Get a video frame record by id."""
        self.cur.execute(
            "SELECT id, captured_at, image_path, status, created_at FROM video_frames WHERE id = %s",
            (frame_id,),
        )
        frame = self.cur.fetchone()
        if frame is None:
            raise KeyError(f"Frame {frame_id} not found")
        return {
            "id": frame[0],
            "captured_at": frame[1].isoformat(),
            "image_path": frame[2],
            "status": frame[3],
            "created_at": frame[4].isoformat(),
        }
    
    def get_last_five_frames(self):
        """Get the last 5 video frames sorted by captured_at descending."""
        self.cur.execute(
            """
            SELECT id, captured_at, created_at, image_path, is_box, above_pallet_thresh, prediction_confidence_score, status
            FROM video_frames
            WHERE status IN ('finished', 'errored')
            ORDER BY captured_at DESC
            LIMIT 5
            """
        )
        frames = self.cur.fetchall()
        return [
            {
                "id": frame[0],
                "captured_at": frame[1].isoformat(),
                "created_at": frame[2].isoformat(),
                "image_path": frame[3],
                "is_box": frame[4],
                "above_pallet_thresh": frame[5],
                "prediction_confidence_score": frame[6],
                "status": frame[7],
            }
            for frame in frames
        ]

    def insert_video_frame(self, captured_at: datetime, image_path: str):
        """Insert a new video frame record into the database."""
        self.cur.execute(
            "INSERT INTO video_frames (captured_at, image_path, created_at) VALUES (%s, %s, %s) RETURNING id",
            (captured_at, image_path, datetime.now()),
        )
        self.conn.commit()
        return self.cur.fetchone()[0]

    def close(self):
        """Close the database connection."""
        self.cur.close()
        self.conn.close()
      
# Initialize database service
db_service = Database()
