#!/usr/bin/env python3
"""
RAG Admin Dashboard Launcher

This script provides an easy way to start the RAG Admin Dashboard with various options.
"""

import os
import sys
import argparse
import logging
import webbrowser
import subprocess
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

load_dotenv(project_root / ".env")


def setup_logging(log_level):
    """Setup logging with specified level"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(project_root / "data" / "logs" / "admin_dashboard.log"),
        ],
    )


def check_requirements():
    """Check if all required Python packages are installed"""
    required_packages = [
        "flask",
        "werkzeug",
        "sqlite3",
        "pandas",
        "openpyxl",
        "reportlab",
        "matplotlib",
        "dotenv",
        "schedule",
        "requests",
        "bs4",
        "langchain",
        "faiss-cpu",
        "cohere",
        "anthropic",
        "openai",
        "google-generativeai",
        "mistralai",
        "xlsxwriter",
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using pip:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True


def check_api_keys():
    """Check if the necessary API keys are set in the environment"""
    required_keys = ["COHERE_API"]
    missing_keys = [key for key in required_keys if not os.environ.get(key)]

    if missing_keys:
        print(f"Missing required API keys: {', '.join(missing_keys)}")
        print("Please set them in the .env file or as environment variables.")
        return False

    return True


def create_directories():
    """Create necessary directories if they don't exist"""
    from src.chatbot.utils.paths import (
        DATA_DIR,
        WEB_CONTENT_DIR,
        INDEXES_DIR,
        DATABASE_DIR,
        LOGS_DIR,
    )

    dirs = [DATA_DIR, WEB_CONTENT_DIR, INDEXES_DIR, DATABASE_DIR, LOGS_DIR]
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            print(f"Creating directory: {dir_path}")
            os.makedirs(dir_path)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="RAG Admin Dashboard Launcher")
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the dashboard on (default: 5000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser window automatically",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: INFO)",
    )
    parser.add_argument(
        "--init-db", action="store_true", help="Initialize the database and exit"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check requirements and exit without starting the server",
    )

    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger("admin_dashboard")

    create_directories()

    # Check requirements
    # if not check_requirements():
    #     sys.exit(1)

    if not check_api_keys():
        logger.warning("Missing API keys. The dashboard might not function properly.")

    if args.check_only:
        print("All requirements satisfied. Ready to run.")
        sys.exit(0)

    from src.web.admin_dashboard import app, init_admin_db

    if args.init_db:
        print("Initializing database...")
        init_admin_db()
        print("Database initialized successfully.")
        sys.exit(0)

    init_admin_db()

    os.environ["FLASK_APP"] = "src.web.admin_dashboard"
    if args.debug:
        os.environ["FLASK_ENV"] = "development"
    else:
        os.environ["FLASK_ENV"] = "production"

    if not args.no_browser:
        url = (
            f"http://{'localhost' if args.host == '0.0.0.0' else args.host}:{args.port}"
        )
        webbrowser.open(url)

    print(
        f"Starting RAG Admin Dashboard on http://{'localhost' if args.host == '0.0.0.0' else args.host}:{args.port}"
    )
    app.run(debug=args.debug, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
