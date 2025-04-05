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
TRAIN_FILES_DIR = DATA_DIR / "training_files"
CHAT_HIST_DIR = DATA_DIR / "chat_history"
CHAT_HIST_ANALYSIS_DIR = DATA_DIR / "chat_hist_analysis"
FONTS_DIR = BASE_DIR / "assets" / "fonts"
CLF_PATH = MODELS_DIR /  "clf.pkl"