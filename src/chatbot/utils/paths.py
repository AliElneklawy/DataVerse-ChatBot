from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"
WEB_CONTENT_DIR = DATA_DIR / "web_content"
DATASETS_DIR = DATA_DIR / "datasets"
DATABASE_DIR = DATA_DIR / "database"
INDEXES_DIR = DATA_DIR / "indexes"
VOICES_DIR = DATA_DIR / "voices"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = DATA_DIR / "logs"
CHAT_HIST_DIR = DATA_DIR / "chat_history"
CLF_PATH = MODELS_DIR /  "clf.pkl"