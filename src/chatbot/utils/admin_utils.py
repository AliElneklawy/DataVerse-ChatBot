import os
import time
import sqlite3
import logging
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

from .utils import DatabaseOps, EmailService
from .paths import DATABASE_DIR, INDEXES_DIR, WEB_CONTENT_DIR, DATA_DIR

logger = logging.getLogger(__name__)


class AdminUtils:
    """Utility class for admin dashboard functions"""

    def __init__(self, admin_db_path: str = None):
        """Initialize AdminUtils with admin database path"""
        self.db_ops = DatabaseOps()
        self.admin_db_path = admin_db_path or str(Path(DATABASE_DIR) / "admin.db")

    def get_system_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get system statistics for the past n days

        Args:
            days: Number of days to get statistics for

        Returns:
            Dict containing system statistics
        """
        stats = {
            "total_users": 0,
            "active_users": 0,
            "total_queries": 0,
            "total_tokens": 0,
            "total_cost": 0,
            "daily_stats": [],
            "model_usage": [],
        }

        try:
            with sqlite3.connect(self.db_ops.db_path) as conn:
                # Get total users
                cursor = conn.execute(
                    "SELECT COUNT(DISTINCT user_id) FROM chat_history"
                )
                stats["total_users"] = cursor.fetchone()[0] or 0

                # Get active users in last 7 days
                time_window = datetime.now() - timedelta(days=days)
                cursor = conn.execute(
                    "SELECT COUNT(DISTINCT user_id) FROM chat_history WHERE timestamp >= ?",
                    (time_window,),
                )
                stats["active_users"] = cursor.fetchone()[0] or 0

                # Get total queries
                cursor = conn.execute("SELECT COUNT(*) FROM chat_history")
                stats["total_queries"] = cursor.fetchone()[0] or 0

                # Get token usage and cost
                cursor = conn.execute(
                    "SELECT SUM(input_tokens) + SUM(output_tokens) as total_tokens, SUM(request_cost) as total_cost FROM cost_monitor"
                )
                result = cursor.fetchone()
                stats["total_tokens"] = result[0] or 0
                stats["total_cost"] = result[1] or 0

                # Get daily stats for the past n days
                for i in range(days):
                    day = datetime.now() - timedelta(days=i)
                    day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
                    day_end = day.replace(
                        hour=23, minute=59, second=59, microsecond=999999
                    )

                    cursor = conn.execute(
                        """
                        SELECT COUNT(*) as queries,
                               COUNT(DISTINCT user_id) as users,
                               SUM(input_tokens) as input_tokens,
                               SUM(output_tokens) as output_tokens,
                               SUM(request_cost) as cost
                        FROM chat_history ch
                        LEFT JOIN cost_monitor cm ON ch.user_id = cm.user_id AND ch.timestamp = cm.timestamp
                        WHERE ch.timestamp BETWEEN ? AND ?
                        """,
                        (day_start, day_end),
                    )

                    row = cursor.fetchone()
                    stats["daily_stats"].append(
                        {
                            "date": day_start.strftime("%Y-%m-%d"),
                            "queries": row[0] or 0,
                            "users": row[1] or 0,
                            "input_tokens": row[2] or 0,
                            "output_tokens": row[3] or 0,
                            "cost": row[4] or 0,
                        }
                    )

                # Get model usage statistics
                cursor = conn.execute(
                    """
                    SELECT model_used, COUNT(*) as count 
                    FROM chat_history 
                    GROUP BY model_used 
                    ORDER BY count DESC
                    """
                )
                stats["model_usage"] = [
                    {"model": row[0], "count": row[1]} for row in cursor.fetchall()
                ]

        except Exception as e:
            logger.error(f"Error getting system stats: {e}")

        return stats

    def generate_usage_report(
        self, start_date: datetime, end_date: datetime
    ) -> BytesIO:
        """Generate a usage report for a date range

        Args:
            start_date: Start date for the report
            end_date: End date for the report

        Returns:
            BytesIO containing the report as Excel file
        """
        try:
            with sqlite3.connect(self.db_ops.db_path) as conn:
                # Get chat history data
                chat_df = pd.read_sql_query(
                    """
                    SELECT user_id, timestamp, question, answer, model_used, embedding_model_used
                    FROM chat_history
                    WHERE timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                    """,
                    conn,
                    params=(start_date, end_date),
                )

                # Get cost data
                cost_df = pd.read_sql_query(
                    """
                    SELECT user_id, timestamp, model_used, embedding_model_used, 
                           input_tokens, output_tokens, request_cost
                    FROM cost_monitor
                    WHERE timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                    """,
                    conn,
                    params=(start_date, end_date),
                )

                # Create Excel file
                output = BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    # Summary sheet
                    summary = {
                        "Metric": [
                            "Date Range",
                            "Total Users",
                            "Total Queries",
                            "Total Input Tokens",
                            "Total Output Tokens",
                            "Total Tokens",
                            "Total Cost ($)",
                        ],
                        "Value": [
                            f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                            chat_df["user_id"].nunique(),
                            len(chat_df),
                            cost_df["input_tokens"].sum(),
                            cost_df["output_tokens"].sum(),
                            cost_df["input_tokens"].sum()
                            + cost_df["output_tokens"].sum(),
                            cost_df["request_cost"].sum(),
                        ],
                    }
                    pd.DataFrame(summary).to_excel(
                        writer, sheet_name="Summary", index=False
                    )

                    # Model usage sheet
                    model_usage = chat_df.groupby("model_used").size().reset_index()
                    model_usage.columns = ["Model", "Query Count"]
                    model_usage.to_excel(writer, sheet_name="Model Usage", index=False)

                    # Daily usage sheet
                    chat_df["date"] = pd.to_datetime(chat_df["timestamp"]).dt.date
                    daily_usage = chat_df.groupby("date").size().reset_index()
                    daily_usage.columns = ["Date", "Query Count"]
                    daily_usage.to_excel(writer, sheet_name="Daily Usage", index=False)

                    # Cost per model sheet
                    cost_per_model = (
                        cost_df.groupby("model_used")
                        .agg(
                            {
                                "input_tokens": "sum",
                                "output_tokens": "sum",
                                "request_cost": "sum",
                            }
                        )
                        .reset_index()
                    )
                    cost_per_model.to_excel(
                        writer, sheet_name="Cost Per Model", index=False
                    )

                    # Raw data sheets
                    chat_df.to_excel(writer, sheet_name="Chat History", index=False)
                    cost_df.to_excel(writer, sheet_name="Cost Data", index=False)

                output.seek(0)
                return output

        except Exception as e:
            logger.error(f"Error generating usage report: {e}")
            return None

    def generate_charts(self, days: int = 30) -> Dict[str, BytesIO]:
        """Generate charts for the dashboard

        Args:
            days: Number of days to include in charts

        Returns:
            Dict of chart names to BytesIO objects containing the charts
        """
        charts = {}

        try:
            with sqlite3.connect(self.db_ops.db_path) as conn:
                # Daily query count chart
                time_window = datetime.now() - timedelta(days=days)
                query_data = pd.read_sql_query(
                    """
                    SELECT date(timestamp) as date, COUNT(*) as count
                    FROM chat_history
                    WHERE timestamp >= ?
                    GROUP BY date(timestamp)
                    ORDER BY date(timestamp)
                    """,
                    conn,
                    params=(time_window,),
                )

                if not query_data.empty:
                    plt.figure(figsize=(10, 6))
                    plt.bar(query_data["date"], query_data["count"])
                    plt.title("Daily Query Count")
                    plt.xlabel("Date")
                    plt.ylabel("Number of Queries")
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    img_buf = BytesIO()
                    plt.savefig(img_buf, format="png")
                    img_buf.seek(0)
                    charts["daily_queries"] = img_buf
                    plt.close()

                # Model usage pie chart
                model_data = pd.read_sql_query(
                    """
                    SELECT model_used, COUNT(*) as count
                    FROM chat_history
                    WHERE timestamp >= ?
                    GROUP BY model_used
                    ORDER BY count DESC
                    """,
                    conn,
                    params=(time_window,),
                )

                if not model_data.empty:
                    plt.figure(figsize=(8, 8))
                    plt.pie(
                        model_data["count"],
                        labels=model_data["model_used"],
                        autopct="%1.1f%%",
                    )
                    plt.title("Model Usage Distribution")
                    plt.tight_layout()

                    img_buf = BytesIO()
                    plt.savefig(img_buf, format="png")
                    img_buf.seek(0)
                    charts["model_usage"] = img_buf
                    plt.close()

                # Daily cost chart
                cost_data = pd.read_sql_query(
                    """
                    SELECT date(timestamp) as date, SUM(request_cost) as cost
                    FROM cost_monitor
                    WHERE timestamp >= ?
                    GROUP BY date(timestamp)
                    ORDER BY date(timestamp)
                    """,
                    conn,
                    params=(time_window,),
                )

                if not cost_data.empty:
                    plt.figure(figsize=(10, 6))
                    plt.plot(cost_data["date"], cost_data["cost"], marker="o")
                    plt.title("Daily API Cost")
                    plt.xlabel("Date")
                    plt.ylabel("Cost ($)")
                    plt.xticks(rotation=45)
                    plt.grid(True, linestyle="--", alpha=0.7)
                    plt.tight_layout()

                    img_buf = BytesIO()
                    plt.savefig(img_buf, format="png")
                    img_buf.seek(0)
                    charts["daily_cost"] = img_buf
                    plt.close()

        except Exception as e:
            logger.error(f"Error generating charts: {e}")

        return charts

    def get_system_config(self, key: str) -> str:
        """Get a system configuration value

        Args:
            key: Configuration key to get

        Returns:
            Configuration value or None if not found
        """
        try:
            with sqlite3.connect(self.admin_db_path) as conn:
                cursor = conn.execute(
                    "SELECT config_value FROM system_config WHERE config_key = ?",
                    (key,),
                )
                result = cursor.fetchone()
                return result[0] if result else None
        except Exception as e:
            logger.error(f"Error getting system config: {e}")
            return None

    def update_system_config(self, key: str, value: str) -> bool:
        """Update a system configuration value

        Args:
            key: Configuration key to update
            value: New configuration value

        Returns:
            True if update was successful, False otherwise
        """
        try:
            with sqlite3.connect(self.admin_db_path) as conn:
                conn.execute(
                    "UPDATE system_config SET config_value = ?, updated_at = ? WHERE config_key = ?",
                    (value, datetime.now(), key),
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating system config: {e}")
            return False

    def get_all_system_configs(self) -> Dict[str, str]:
        """Get all system configurations

        Returns:
            Dict of configuration keys to values
        """
        try:
            with sqlite3.connect(self.admin_db_path) as conn:
                cursor = conn.execute(
                    "SELECT config_key, config_value FROM system_config"
                )
                return dict(cursor.fetchall())
        except Exception as e:
            logger.error(f"Error getting all system configs: {e}")
            return {}

    def set_email_configuration(self, email: str, enable_notifications: bool) -> bool:
        """Configure email notifications

        Args:
            email: Email address for notifications
            enable_notifications: Whether to enable notifications

        Returns:
            True if configuration was successful, False otherwise
        """
        try:
            # Update email in email service
            email_service = EmailService()
            if email:
                email_service.receiver_email = email

            # Update notification setting in system config
            self.update_system_config(
                "email_notifications_enabled", str(enable_notifications).lower()
            )

            return True
        except Exception as e:
            logger.error(f"Error setting email configuration: {e}")
            return False

    async def test_rag_system(
        self,
        query: str,
        content_path: Path,
        model_name: str,
        provider: str,
        chunking_type: str,
        rerank: bool,
    ) -> Tuple[bool, str, float]:
        """Test the RAG system with a query

        Args:
            query: Query to test
            content_path: Path to content file
            model_name: Model name to use
            provider: Model provider (cohere, claude, etc.)
            chunking_type: Chunking type to use
            rerank: Whether to use reranking

        Returns:
            Tuple of (success, response, time_taken)
        """
        from uuid import uuid4

        start_time = time.time()

        try:
            if provider == "cohere":
                from ..rag.cohere_rag import CohereRAG

                rag = CohereRAG(
                    content_path,
                    INDEXES_DIR,
                    model_name=model_name,
                    chunking_type=chunking_type,
                    rerank=rerank,
                )
            elif provider == "claude":
                from ..rag.claude_rag import ClaudeRAG

                rag = ClaudeRAG(
                    content_path,
                    INDEXES_DIR,
                    model_name=model_name,
                    chunking_type=chunking_type,
                    rerank=rerank,
                )
            elif provider == "gemini":
                from ..rag.gemini_rag import GeminiRAG

                rag = GeminiRAG(
                    content_path,
                    INDEXES_DIR,
                    model_name=model_name,
                    chunking_type=chunking_type,
                    rerank=rerank,
                )
            elif provider == "grok":
                from ..rag.grok_rag import GrokRAG

                rag = GrokRAG(
                    content_path,
                    INDEXES_DIR,
                    model_name=model_name,
                    chunking_type=chunking_type,
                    rerank=rerank,
                )
            elif provider == "openai":
                from ..rag.openai_rag import OpenAIRAG

                rag = OpenAIRAG(
                    content_path,
                    INDEXES_DIR,
                    model_name=model_name,
                    chunking_type=chunking_type,
                    rerank=rerank,
                )
            elif provider == "mistral":
                from ..rag.mistral_rag import MistralRAG

                rag = MistralRAG(
                    content_path,
                    INDEXES_DIR,
                    model_name=model_name,
                    chunking_type=chunking_type,
                    rerank=rerank,
                )
            elif provider == "deepseek":
                from ..rag.deepseek_rag import DeepseekRAG

                rag = DeepseekRAG(
                    content_path,
                    INDEXES_DIR,
                    model_name=model_name,
                    chunking_type=chunking_type,
                    rerank=rerank,
                )
            else:
                return False, f"Unknown provider: {provider}", 0

            # Generate a test user ID
            test_user_id = f"test-{uuid4()}"

            # Get response
            response = await rag.get_response(query, test_user_id)

            end_time = time.time()
            time_taken = end_time - start_time

            return True, response, time_taken

        except Exception as e:
            end_time = time.time()
            time_taken = end_time - start_time
            logger.error(f"Error testing RAG system: {e}")
            return False, str(e), time_taken

    def cleanup_test_data(self) -> bool:
        """Clean up test data from the database

        Returns:
            True if cleanup was successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_ops.db_path) as conn:
                conn.execute("DELETE FROM chat_history WHERE user_id LIKE 'test-%'")
                conn.execute("DELETE FROM cost_monitor WHERE user_id LIKE 'test-%'")
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error cleaning up test data: {e}")
            return False
