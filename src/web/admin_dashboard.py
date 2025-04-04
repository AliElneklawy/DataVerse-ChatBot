import os
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import asyncio
import threading
import hashlib
import sys
from contextlib import redirect_stdout
import io

sys.path.append(str(Path(__file__).resolve().parent.parent))

from chatbot.agents.chat_hist_analyzer_agent import ChatHistortAnalyzerAgent
from chatbot.agents.chat_hist_analyzer_agent import ChatHistortAnalyzerAgent
from chatbot.utils.utils import DatabaseOps, EmailService, create_folder
from chatbot.utils.paths import (
    DATABASE_DIR,
    WEB_CONTENT_DIR,
    INDEXES_DIR,
    DATA_DIR,
    LOGS_DIR,
    LOGS_DIR,
    TRAIN_FILES_DIR,
)
from chatbot.config import (
    RAGConfig, GrokConfig, CohereConfig, ClaudeConfig, 
    GeminiConfig, MistralConfig, OpenAIConfig, DeepSeekConfig
)
from chatbot.rag.cohere_rag import CohereRAG
from chatbot.rag.claude_rag import ClaudeRAG
from chatbot.rag.gemini_rag import GeminiRAG
from chatbot.rag.grok_rag import GrokRAG
from chatbot.rag.openai_rag import OpenAIRAG
from chatbot.rag.mistral_rag import MistralRAG
from chatbot.rag.deepseek_rag import DeepseekRAG
from chatbot.crawler import Crawler
import tldextract

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler(
            create_folder(LOGS_DIR) / 'admin_dashboard.log', 
            maxBytes=1024 * 1024 * 10, 
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, 
            template_folder=str(Path(__file__).parent / 'templates'),
            static_folder=str(Path(__file__).parent / 'static'))
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev_secret_key')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['UPLOAD_FOLDER'] = str(WEB_CONTENT_DIR)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
app.config['ADMIN_DB_PATH'] = str(Path(DATABASE_DIR) / "admin.db")

# Create admin database if it doesn't exist
def init_admin_db():
    create_folder(DATABASE_DIR)
    with sqlite3.connect(app.config['ADMIN_DB_PATH']) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS admin_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        # Check if admin user exists, if not create default admin
        cursor = conn.execute("SELECT COUNT(*) FROM admin_users")
        if cursor.fetchone()[0] == 0:
            default_password = "admin123"  # Change in production
            conn.execute(
                "INSERT INTO admin_users (username, password_hash, email) VALUES (?, ?, ?)",
                ("admin", generate_password_hash(default_password), "admin@example.com")
            )
            logger.info("Created default admin user")
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS system_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_key TEXT UNIQUE NOT NULL,
                config_value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Initialize default configurations
        default_configs = [
            ("default_llm_provider", "cohere"),
            ("default_llm_model", "command-r-plus-08-2024"),
            ("default_chunking_type", "recursive"),
            ("default_chunk_size", str(RAGConfig.CHUNK_SIZE)),
            ("default_chunk_overlap", str(RAGConfig.CHUNK_OVERLAP)),
            ("rerank_enabled", "true"),
            ("email_notifications_enabled", "true"),
        ]
        
        for key, value in default_configs:
            conn.execute(
                "INSERT OR IGNORE INTO system_config (config_key, config_value) VALUES (?, ?)",
                (key, value)
            )
        
        conn.commit()

# Login decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        with sqlite3.connect(app.config['ADMIN_DB_PATH']) as conn:
            cursor = conn.execute(
                "SELECT id, username, password_hash FROM admin_users WHERE username = ?", 
                (username,)
            )
            user = cursor.fetchone()
            
            if user and check_password_hash(user[2], password):
                session['username'] = user[1]
                session['user_id'] = user[0]
                
                # Update last login time
                conn.execute(
                    "UPDATE admin_users SET last_login = ? WHERE id = ?",
                    (datetime.now(), user[0])
                )
                conn.commit()
                
                flash('Login successful', 'success')
                return redirect(url_for('dashboard'))
            
            flash('Invalid username or password', 'danger')
    
    return render_template('admin/login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('user_id', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/')
@login_required
def dashboard():
    # Get system statistics for dashboard
    db_ops = DatabaseOps()
    
    # Recent conversations
    recent_conversations = db_ops.get_chat_history(full_history=True, last_n_hours=24)
    
    # User statistics
    user_stats = {}
    with sqlite3.connect(db_ops.db_path) as conn:
        # Total unique users
        cursor = conn.execute("SELECT COUNT(DISTINCT user_id) FROM chat_history")
        user_stats['total_users'] = cursor.fetchone()[0]
        
        # Active users in last 24 hours
        time_window = datetime.now() - timedelta(hours=24)
        cursor = conn.execute(
            "SELECT COUNT(DISTINCT user_id) FROM chat_history WHERE timestamp >= ?",
            (time_window,)
        )
        user_stats['active_users_24h'] = cursor.fetchone()[0]
        
        # Total conversations
        cursor = conn.execute("SELECT COUNT(*) FROM chat_history")
        user_stats['total_conversations'] = cursor.fetchone()[0]
        
        # Conversations in last 24 hours
        cursor = conn.execute(
            "SELECT COUNT(*) FROM chat_history WHERE timestamp >= ?",
            (time_window,)
        )
        user_stats['conversations_24h'] = cursor.fetchone()[0]
        
        # Token usage and cost
        cursor = conn.execute(
            "SELECT SUM(input_tokens), SUM(output_tokens), SUM(request_cost) FROM cost_monitor"
        )
        result = cursor.fetchone()
        user_stats['total_input_tokens'] = result[0] or 0
        user_stats['total_output_tokens'] = result[1] or 0
        user_stats['total_cost'] = result[2] or 0
        
        # Token usage in last 24 hours
        cursor = conn.execute(
            "SELECT SUM(input_tokens), SUM(output_tokens), SUM(request_cost) FROM cost_monitor WHERE timestamp >= ?",
            (time_window,)
        )
        result = cursor.fetchone()
        user_stats['input_tokens_24h'] = result[0] or 0
        user_stats['output_tokens_24h'] = result[1] or 0
        user_stats['cost_24h'] = result[2] or 0
        
        # Model usage statistics
        cursor = conn.execute(
            "SELECT model_used, COUNT(*) FROM chat_history GROUP BY model_used ORDER BY COUNT(*) DESC"
        )
        user_stats['model_usage'] = cursor.fetchall()
    
    # Get content files
    content_files = []
    for file_path in WEB_CONTENT_DIR.glob('*.txt'):
        content_size = file_path.stat().st_size / 1024  # KB
        content_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
        content_files.append({
            'name': file_path.name,
            'size_kb': content_size,
            'modified': content_modified,
            'path': str(file_path),
        })
    
    # Get indexes
    indexes = []
    for index_dir in INDEXES_DIR.glob('*'):
        if index_dir.is_dir() and (index_dir / 'index.faiss').exists():
            index_size = sum(f.stat().st_size for f in index_dir.glob('**/*') if f.is_file()) / 1024  # KB
            index_modified = datetime.fromtimestamp(max(f.stat().st_mtime for f in index_dir.glob('**/*') if f.is_file()))
            indexes.append({
                'name': index_dir.name,
                'size_kb': index_size,
                'modified': index_modified,
                'path': str(index_dir),
            })
    
    # System configurations
    with sqlite3.connect(app.config['ADMIN_DB_PATH']) as conn:
        cursor = conn.execute("SELECT config_key, config_value FROM system_config")
        system_configs = dict(cursor.fetchall())
    
    return render_template(
        'admin/dashboard.html',
        user_stats=user_stats,
        recent_conversations=recent_conversations,
        content_files=content_files,
        indexes=indexes,
        system_configs=system_configs
    )

@app.route('/users')
@login_required
def users():
    db_ops = DatabaseOps()
    
    # Get all unique users with their last activity
    with sqlite3.connect(db_ops.db_path) as conn:
        cursor = conn.execute("""
            SELECT user_id, MAX(timestamp) as last_activity, COUNT(*) as conversation_count
            FROM chat_history
            GROUP BY user_id
            ORDER BY last_activity DESC
        """)
        users = [
            {
                'user_id': row[0],
                'last_activity': datetime.fromisoformat(row[1]),
                'conversation_count': row[2]
            }
            for row in cursor.fetchall()
        ]
    
    # Pass the current time to the template
    return render_template('admin/users.html', users=users, now=datetime.now())

@app.route('/user/<user_id>')
@login_required
def user_detail(user_id):
    db_ops = DatabaseOps()
    
    # Get user's chat history - make sure we're passing user_id to filter the history
    user_history = []
    with sqlite3.connect(db_ops.db_path) as conn:
        cursor = conn.execute("""
            SELECT timestamp, question, answer, model_used, embedding_model_used
            FROM chat_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
        """, (user_id,))
        
        interactions = []
        for row in cursor.fetchall():
            interactions.append({
                'timestamp': row[0],
                'user': row[1],
                'assistant': row[2],
                'llm': row[3],
                'embedder': row[4]
            })
        
        if interactions:
            user_history = [{
                'user_id': user_id,
                'interactions': interactions
            }]
    
    # Get user's token usage
    with sqlite3.connect(db_ops.db_path) as conn:
        cursor = conn.execute("""
            SELECT SUM(input_tokens), SUM(output_tokens), SUM(request_cost)
            FROM cost_monitor
            WHERE user_id = ?
        """, (user_id,))
        token_usage = cursor.fetchone()
    
    user_stats = {
        'total_input_tokens': token_usage[0] or 0,
        'total_output_tokens': token_usage[1] or 0,
        'total_cost': token_usage[2] or 0,
    }
    
    return render_template(
        'admin/user_detail.html',
        user_id=user_id,
        user_history=user_history,
        user_stats=user_stats
    )

@app.route('/content')
@login_required
def content():
    """Content management page"""
    # List content files
    content_files = []
    for file_path in WEB_CONTENT_DIR.glob('*.txt'):
        content_size = file_path.stat().st_size / 1024  # KB
        content_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
        content_files.append({
            'name': file_path.name,
            'size_kb': content_size,
            'modified': content_modified,
            'path': str(file_path),
        })
    
    # List training files
    training_files = []
    for file_path in TRAIN_FILES_DIR.glob('*'):
        if file_path.is_file():
            file_size = file_path.stat().st_size / 1024  # KB
            file_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
            training_files.append({
                'name': file_path.name,
                'size_kb': file_size,
                'modified': file_modified,
                'path': str(file_path),
            })
    
    # List indexes
    indexes = []
    for index_dir in INDEXES_DIR.glob('*'):
        if index_dir.is_dir() and (index_dir / 'index.faiss').exists():
            index_size = sum(f.stat().st_size for f in index_dir.glob('**/*') if f.is_file()) / 1024  # KB
            index_modified = datetime.fromtimestamp(max(f.stat().st_mtime for f in index_dir.glob('**/*') if f.is_file()))
            indexes.append({
                'name': index_dir.name,
                'size_kb': index_size,
                'modified': index_modified,
                'path': str(index_dir),
            })
    
    return render_template(
        'admin/content.html',
        content_files=content_files,
        training_files=training_files,
        indexes=indexes
    )

@app.route('/training/view/<filename>')
@login_required
def view_training_file(filename):
    """View training file content"""
    file_path = TRAIN_FILES_DIR / filename
    
    if not file_path.exists():
        flash('File not found', 'danger')
        return redirect(url_for('content'))
    
    # Handle binary files
    file_extension = file_path.suffix.lower()
    binary_formats = ['.pdf', '.docx', '.pptx', '.xlsx', '.png', '.jpeg', '.jpg']
    
    if file_extension in binary_formats:
        flash('Binary files cannot be viewed directly. Please index the file to extract content.', 'warning')
        return redirect(url_for('content'))
    
    # Read and display text file content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        flash('This file appears to be binary and cannot be viewed directly.', 'warning')
        return redirect(url_for('content'))
    
    return render_template(
        'admin/view_content.html',
        filename=filename,
        content=content,
        file_type='training'
    )

@app.route('/training/delete/<filename>', methods=['POST'])
@login_required
def delete_training_file(filename):
    """Delete a training file"""
    file_path = TRAIN_FILES_DIR / filename
    
    if not file_path.exists():
        flash('File not found', 'danger')
    else:
        try:
            os.remove(file_path)
            flash(f'File {filename} deleted successfully', 'success')
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            flash(f'Error deleting file: {str(e)}', 'danger')
    
    return redirect(url_for('content'))

@app.route('/training/index/<filename>', methods=['POST'])
@login_required
def index_training_file(filename):
    """Index a training file"""
    file_path = TRAIN_FILES_DIR / filename
    
    if not file_path.exists():
        flash('File not found', 'danger')
        return redirect(url_for('content'))
    
    # Get model configuration for indexing
    with sqlite3.connect(app.config['ADMIN_DB_PATH']) as conn:
        cursor = conn.execute("SELECT config_value FROM system_config WHERE config_key = 'default_chunking_type'")
        row = cursor.fetchone()
        chunking_type = row[0] if row else "recursive"
    
    try:
        # Check file extension to determine processing method
        file_extension = file_path.suffix.lower()
        binary_formats = ['.pdf', '.docx', '.pptx', '.xlsx', '.csv', '.png', '.jpeg', '.jpg']
        
        if file_extension in binary_formats:
            # This is a binary file that needs text extraction
            logger.info(f"Processing binary file: {filename} with extension {file_extension}")
            
            # Determine the base name for the content file
            base_name = filename.rsplit('.', 1)[0]  # Remove extension
            content_file_path = WEB_CONTENT_DIR / f"{base_name}.txt"
            
            # Use FileLoader to extract content
            from chatbot.utils.file_loader import FileLoader
            loader = FileLoader(str(file_path), str(content_file_path), client="langchain")
            documents = loader.extract_from_file()
            
            if not documents:
                flash(f'Failed to extract content from {filename}', 'danger')
                return redirect(url_for('content'))
            
            logger.info(f"Extracted {len(documents)} document segments from {filename}")
            
            # Check if we already have an index for this content file
            existing_index_path = None
            if content_file_path.exists():
                content_hash = (hashlib.sha256(str(content_file_path).encode('utf-8')).hexdigest(), 16)[0][:15]
                index_name = f"index_{content_hash}.faiss"
                index_path = INDEXES_DIR / index_name
                
                if index_path.exists():
                    existing_index_path = index_path
            
            if existing_index_path:
                # Update existing index
                logger.info(f"Updating existing index for {content_file_path}")
                
                # Get extracted content
                extracted_text = ""
                for doc in documents:
                    extracted_text += doc.page_content + "\n\n"
                
                # Create a RAG instance with the existing index
                rag = CohereRAG(content_file_path, 
                                INDEXES_DIR, 
                                chunking_type=chunking_type,
                                rerank=False)
                
                # Update the vectorstore with new content
                rag._update_vectorstore(extracted_text)
                
                flash(f'Content from {filename} extracted and added to existing knowledge base', 'success')
            else:
                # Create new index from the content file
                logger.info(f"Creating new index for {content_file_path}")
                
                # Create the RAG instance which will create a new index
                rag = CohereRAG(content_file_path, 
                                INDEXES_DIR, 
                                chunking_type=chunking_type,
                                rerank=False)
                
                flash(f'Content from {filename} extracted and indexed successfully', 'success')
        else:
            # For text files, copy to web_content_dir first
            content_file_path = WEB_CONTENT_DIR / filename
            
            # Copy the file content
            with open(file_path, 'r', encoding='utf-8') as src_file:
                content = src_file.read()
                
            with open(content_file_path, 'w', encoding='utf-8') as dest_file:
                dest_file.write(content)
            
            # Create the RAG instance for indexing
            rag = CohereRAG(content_file_path, 
                            INDEXES_DIR, 
                            chunking_type=chunking_type,
                            rerank=False)
            
            flash(f'Content indexed successfully', 'success')
    except Exception as e:
        logger.error(f"Error indexing training file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        flash(f'Error indexing content: {str(e)}', 'danger')
    
    return redirect(url_for('content'))

@app.route('/content/upload', methods=['POST'])
@login_required
def upload_content():
    """Upload a file to be used as knowledge base content"""
    if 'content_file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('content'))
    
    file = request.files['content_file']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('content'))
    
    # Check file type
    allowed_extensions = {
        # Text formats
        '.txt', '.md', '.html', '.xml', '.json', '.csv', 
        # Document formats
        '.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls',
        # Image formats that can be processed for text
        '.png', '.jpg', '.jpeg'
    }
    
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        flash(f'File type {file_extension} is not supported. Supported types: {", ".join(allowed_extensions)}', 'danger')
        return redirect(url_for('content'))
    
    if file:
        if not os.path.exists(TRAIN_FILES_DIR):
            os.makedirs(TRAIN_FILES_DIR)
            
        filename = secure_filename(file.filename)
        file_path = os.path.join(TRAIN_FILES_DIR, filename)
        
        # Check if file already exists
        if os.path.exists(file_path):
            # Add timestamp to filename to make it unique
            name, ext = os.path.splitext(filename)
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            filename = f"{name}_{timestamp}{ext}"
            file_path = os.path.join(TRAIN_FILES_DIR, filename)
        
        file.save(file_path)
        
        # For binary formats, let the user know they need to index the file to extract content
        if file_extension in ['.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.png', '.jpg', '.jpeg']:
            flash(f'File {filename} uploaded successfully. Click "Index" to extract and index the content.', 'success')
        else:
            flash(f'File {filename} uploaded successfully. Click "Index" to index the content.', 'success')
    
    return redirect(url_for('content'))

@app.route('/content/crawl', methods=['POST'])
@login_required
def crawl_website():
    """Crawl a website and save its content"""
    url = request.form.get('website_url')
    max_depth = int(request.form.get('max_depth', 2))
    
    if not url:
        flash('No URL provided', 'danger')
        return redirect(url_for('content'))
    
    try:
        domain_name = tldextract.extract(url).domain
        crawler = Crawler(url, domain_name)
        crawl_id = crawler.get_crawl_id()
        
        # Start the crawler in a background thread
        def run_crawler_task():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                content_path = loop.run_until_complete(
                    crawler.extract_content(url, webpage_only=False, max_depth=max_depth)
                )
                loop.close()
                logger.info(f"Crawling completed: {content_path}")
            except Exception as e:
                logger.error(f"Error in crawler task: {e}")
        
        thread = threading.Thread(target=run_crawler_task)
        thread.daemon = True
        thread.start()
        
        # Return the crawl ID for frontend to poll status
        return jsonify({
            'success': True,
            'message': 'Website crawling started',
            'crawl_id': crawl_id
        })
        
    except Exception as e:
        logger.error(f"Error starting crawler: {e}")
        return jsonify({
            'success': False,
            'message': f'Error starting crawler: {str(e)}'
        })
    
@app.route('/api/crawler/progress/<crawl_id>')
@login_required
def crawler_progress(crawl_id):
    """Get progress of a crawler session"""
    from chatbot.utils.crawler_progress import CrawlerProgress
    
    progress = CrawlerProgress.get_progress(crawl_id)
    if progress:
        return jsonify({
            'success': True,
            'progress': progress
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Crawl session not found'
        }), 404
    
@app.route('/content/view/<filename>')
@login_required
def view_content(filename):
    file_path = WEB_CONTENT_DIR / filename
    
    if not file_path.exists():
        flash('File not found', 'danger')
        return redirect(url_for('content'))
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return render_template(
        'admin/view_content.html',
        filename=filename,
        content=content
    )

@app.route('/content/delete/<filename>', methods=['POST'])
@login_required
def delete_content(filename):
    file_path = WEB_CONTENT_DIR / filename
    
    if not file_path.exists():
        flash('File not found', 'danger')
    else:
        try:
            os.remove(file_path)
            flash(f'File {filename} deleted successfully', 'success')
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            flash(f'Error deleting file: {str(e)}', 'danger')
    
    return redirect(url_for('content'))

@app.route('/system/email', methods=['POST'])
@login_required
def system_email():
    """Update email notification settings"""
    email = request.form.get('email')
    enable_notifications = 'enable_notifications' in request.form
    
    # Save email to the database
    with sqlite3.connect(app.config['ADMIN_DB_PATH']) as conn:
        # First, check if email_address config exists
        cursor = conn.execute("SELECT COUNT(*) FROM system_config WHERE config_key = 'email_address'")
        if cursor.fetchone()[0] > 0:
            # Update existing email setting
            conn.execute(
                "UPDATE system_config SET config_value = ?, updated_at = ? WHERE config_key = ?",
                (email, datetime.now(), 'email_address')
            )
        else:
            # Insert new email setting
            conn.execute(
                "INSERT INTO system_config (config_key, config_value, updated_at) VALUES (?, ?, ?)",
                ('email_address', email, datetime.now())
            )
        
        # Update notification setting
        conn.execute(
            "UPDATE system_config SET config_value = ?, updated_at = ? WHERE config_key = ?",
            (str(enable_notifications).lower(), datetime.now(), 'email_notifications_enabled')
        )
        conn.commit()
    
    # Also update the current EmailService instance
    email_service = EmailService()
    if email:
        email_service.receiver_email = email
    
    flash('Email configuration updated successfully', 'success')
    return redirect(url_for('system'))

@app.route('/content/index/<filename>', methods=['POST'])
@login_required
def index_content(filename):
    """Index content for RAG, handling different file formats appropriately"""
    file_path = TRAIN_FILES_DIR / filename
    
    if not file_path.exists():
        flash('File not found', 'danger')
        return redirect(url_for('content'))
    
    # Get model configuration for indexing
    with sqlite3.connect(app.config['ADMIN_DB_PATH']) as conn:
        cursor = conn.execute("SELECT config_value FROM system_config WHERE config_key = 'default_chunking_type'")
        row = cursor.fetchone()
        chunking_type = row[0] if row else "recursive"
    
    try:
        # Check file extension to determine processing method
        file_extension = file_path.suffix.lower()
        binary_formats = ['.pdf', '.docx', '.pptx', '.xlsx', '.csv', '.png', '.jpeg', '.jpg']
        
        if file_extension in binary_formats:
            # This is a binary file that needs text extraction
            logger.info(f"Processing binary file: {filename} with extension {file_extension}")
            
            # Determine the base name for the content file
            base_name = filename.rsplit('.', 1)[0]  # Remove extension
            content_file_path = WEB_CONTENT_DIR / f"{base_name}.txt"
            
            # Use FileLoader to extract content
            from chatbot.utils.file_loader import FileLoader
            loader = FileLoader(str(file_path), str(content_file_path), client="docling")
            documents = loader.extract_from_file()
            
            if not documents:
                flash(f'Failed to extract content from {filename}', 'danger')
                return redirect(url_for('content'))
            
            logger.info(f"Extracted {len(documents)} document segments from {filename}")
            
            # If we already have an existing index for this content, update it
            # Otherwise create a new one
            existing_index_path = None
            
            # Check if we already have an index for this content file
            for existing_index in INDEXES_DIR.glob("index_*.faiss"):
                if existing_index.is_dir():
                    # Try to determine the source file for this index
                    try:
                        # Load index to check its source
                        from langchain_community.vectorstores import FAISS
                        from chatbot.embeddings.base_embedding import CohereEmbedding
                        
                        # Just check if this index is associated with our content file
                        if content_file_path.exists():
                            content_hash = (hashlib.sha256(str(content_file_path).encode('utf-8')).hexdigest(), 16)[0][:15]
                            index_name = f"index_{content_hash}.faiss"
                            
                            if existing_index.name == index_name:
                                existing_index_path = existing_index
                                break
                    except Exception as e:
                        logger.error(f"Error checking index {existing_index}: {e}")
            
            if existing_index_path:
                # Update existing index
                logger.info(f"Updating existing index for {content_file_path}")
                
                # Get extracted content
                extracted_text = ""
                for doc in documents:
                    extracted_text += doc.page_content + "\n\n"
                
                # Create a RAG instance with the existing index
                rag = CohereRAG(content_file_path, 
                                INDEXES_DIR, 
                                chunking_type=chunking_type,
                                rerank=False)
                
                # Update the vectorstore with new content
                rag._update_vectorstore(extracted_text)
                
                flash(f'Content from {filename} extracted and added to existing knowledge base', 'success')
            else:
                # Create new index from the content file
                logger.info(f"Creating new index for {content_file_path}")
                
                # Create the RAG instance which will create a new index
                rag = CohereRAG(content_file_path, 
                                INDEXES_DIR, 
                                chunking_type=chunking_type,
                                rerank=False)
                
                flash(f'Content from {filename} extracted and indexed successfully', 'success')
        else:
            # Regular text file - index directly
            logger.info(f"Indexing text file: {filename}")
            
            # Create the RAG instance for indexing
            rag = CohereRAG(file_path, 
                            INDEXES_DIR, 
                            chunking_type=chunking_type,
                            rerank=False)
            
            flash(f'Content indexed successfully', 'success')
    except Exception as e:
        logger.error(f"Error indexing content: {e}")
        import traceback
        logger.error(traceback.format_exc())
        flash(f'Error indexing content: {str(e)}', 'danger')
    
    return redirect(url_for('content'))

@app.route('/indexes/delete/<index_name>', methods=['POST'])
@login_required
def delete_index(index_name):
    index_path = INDEXES_DIR / index_name
    
    if not index_path.exists():
        flash('Index not found', 'danger')
    else:
        try:
            import shutil
            shutil.rmtree(index_path)
            flash(f'Index {index_name} deleted successfully', 'success')
        except Exception as e:
            logger.error(f"Error deleting index: {e}")
            flash(f'Error deleting index: {str(e)}', 'danger')
    
    return redirect(url_for('content'))

@app.route('/models')
@login_required
def models():
    # Get all available models from config classes
    available_models = {
        'grok': GrokConfig.AVAILABLE_MODELS,
        'cohere': CohereConfig.AVAILABLE_MODELS,
        'claude': ClaudeConfig.AVAILABLE_MODELS,
        'gemini': GeminiConfig.AVAILABLE_MODELS,
        'mistral': MistralConfig.AVAILABLE_MODELS,
        'openai': OpenAIConfig.AVAILABLE_MODELS,
        'deepseek': DeepSeekConfig.AVAILABLE_MODELS,
    }
    
    # Get current configurations
    with sqlite3.connect(app.config['ADMIN_DB_PATH']) as conn:
        cursor = conn.execute("SELECT config_key, config_value FROM system_config")
        system_configs = dict(cursor.fetchall())
    
    return render_template(
        'admin/models.html',
        available_models=available_models,
        system_configs=system_configs
    )

@app.route('/models/update', methods=['POST'])
@login_required
def update_model_config():
    llm_provider = request.form.get('llm_provider')
    llm_model = request.form.get('llm_model')
    chunking_type = request.form.get('chunking_type')
    chunk_size = request.form.get('chunk_size')
    chunk_overlap = request.form.get('chunk_overlap')
    rerank_enabled = 'rerank_enabled' in request.form
    
    with sqlite3.connect(app.config['ADMIN_DB_PATH']) as conn:
        configs_to_update = [
            ('default_llm_provider', llm_provider),
            ('default_llm_model', llm_model),
            ('default_chunking_type', chunking_type),
            ('default_chunk_size', chunk_size),
            ('default_chunk_overlap', chunk_overlap),
            ('rerank_enabled', str(rerank_enabled).lower()),
        ]
        
        for key, value in configs_to_update:
            conn.execute(
                "UPDATE system_config SET config_value = ?, updated_at = ? WHERE config_key = ?",
                (value, datetime.now(), key)
            )
        
        conn.commit()
    
    flash('Model configuration updated successfully', 'success')
    return redirect(url_for('models'))

@app.route('/system')
@login_required
def system():
    # Get system logs
    log_file = LOGS_DIR / 'logs.log'
    logs = []
    
    if log_file.exists():
        with open(log_file, 'r') as f:
            logs = [line.strip() for line in f.readlines()[-100:]]  # Get last 100 lines
    
    # Get settings from database
    with sqlite3.connect(app.config['ADMIN_DB_PATH']) as conn:
        # Get email notification setting
        cursor = conn.execute("SELECT config_value FROM system_config WHERE config_key = 'email_notifications_enabled'")
        row = cursor.fetchone()
        email_notifications_enabled = row[0] == 'true' if row else False
        
        # Get email address
        cursor = conn.execute("SELECT config_value FROM system_config WHERE config_key = 'email_address'")
        row = cursor.fetchone()
        current_email = row[0] if row else ''
    
    return render_template(
        'admin/system.html',
        logs=logs,
        email_notifications_enabled=email_notifications_enabled,
        current_email=current_email
    )

@app.route('/system/email', methods=['POST'])
@login_required
def update_email_config():
    email = request.form.get('email')
    enable_notifications = 'enable_notifications' in request.form
    
    # Update email in email service
    email_service = EmailService()
    if email:
        email_service.receiver_email = email
    
    # Update notification setting in system config
    with sqlite3.connect(app.config['ADMIN_DB_PATH']) as conn:
        conn.execute(
            "UPDATE system_config SET config_value = ?, updated_at = ? WHERE config_key = ?",
            (str(enable_notifications).lower(), datetime.now(), 'email_notifications_enabled')
        )
        conn.commit()
    
    flash('Email configuration updated successfully', 'success')
    return redirect(url_for('system'))

@app.route('/system/test_email', methods=['POST'])
@login_required
def test_email():
    """Send a test email"""
    # Get email from database
    with sqlite3.connect(app.config['ADMIN_DB_PATH']) as conn:
        cursor = conn.execute("SELECT config_value FROM system_config WHERE config_key = 'email_address'")
        row = cursor.fetchone()
        email_address = row[0] if row else None
    
    if not email_address:
        flash('No email address configured', 'danger')
        return redirect(url_for('system'))
    
    # Create EmailService and set the email
    email_service = EmailService()
    email_service.receiver_email = email_address
    
    try:
        email_service.send_email(
            subject="RAG Admin Dashboard - Test Email",
            unknowns=[("Test question", "This is a test response")]
        )
        flash('Test email sent successfully', 'success')
    except Exception as e:
        logger.error(f"Error sending test email: {e}")
        flash(f'Error sending test email: {str(e)}', 'danger')
    
    return redirect(url_for('system'))

@app.route('/history')
@login_required
def chat_history():
    """Display chat history with most recent conversations at the top"""
    db_ops = DatabaseOps()
    
    # Get full history with all conversations
    with sqlite3.connect(db_ops.db_path) as conn:
        cursor = conn.execute("""
            SELECT user_id, MAX(timestamp) as latest_timestamp
            FROM chat_history
            GROUP BY user_id
            ORDER BY latest_timestamp DESC
        """)
        user_latest_times = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Get all conversations
        cursor = conn.execute("""
            SELECT user_id, timestamp, question, answer, model_used, embedding_model_used
            FROM chat_history
            ORDER BY timestamp DESC
        """)
        
        all_interactions = cursor.fetchall()
        
        # Group by user_id
        history_by_user = {}
        for user_id, timestamp, question, answer, model, embedder in all_interactions:
            if user_id not in history_by_user:
                history_by_user[user_id] = {
                    'user_id': user_id,
                    'interactions': []
                }
            
            history_by_user[user_id]['interactions'].append({
                'timestamp': timestamp,
                'user': question,
                'assistant': answer,
                'llm': model,
                'embedder': embedder
            })
    
    # Sort users by their latest interaction time (most recent first)
    sorted_history = sorted(
        history_by_user.values(),
        key=lambda x: user_latest_times.get(x['user_id'], ''),
        reverse=True
    )
    
    # Get model usage statistics
    model_usage = []
    daily_queries = []
    
    with sqlite3.connect(db_ops.db_path) as conn:
        # Model usage
        cursor = conn.execute("""
            SELECT model_used, COUNT(*) as count
            FROM chat_history
            GROUP BY model_used
            ORDER BY count DESC
        """)
        model_usage = [{'model': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        # Daily queries for the last 30 days
        cursor = conn.execute("""
            SELECT date(timestamp) as date, COUNT(*) as count
            FROM chat_history
            WHERE timestamp >= datetime('now', '-30 days')
            GROUP BY date(timestamp)
            ORDER BY date(timestamp)
        """)
        daily_queries = [{'date': row[0], 'count': row[1]} for row in cursor.fetchall()]
    
    return render_template(
        'admin/history.html',
        history=sorted_history,
        model_usage=model_usage,
        daily_queries=daily_queries
    )

@app.route('/api/stats')
@login_required
def api_stats():
    db_ops = DatabaseOps()
    
    days = 7
    stats = []
    
    with sqlite3.connect(db_ops.db_path) as conn:
        for i in range(days):
            day = datetime.now() - timedelta(days=i)
            day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day.replace(hour=23, minute=59, second=59, microsecond=999999)
            
            cursor = conn.execute(
                """
                SELECT COUNT(*) as queries,
                       COUNT(DISTINCT ch.user_id) as users,
                       SUM(cm.input_tokens) as input_tokens,
                       SUM(cm.output_tokens) as output_tokens,
                       SUM(cm.request_cost) as cost
                FROM chat_history ch
                LEFT JOIN cost_monitor cm ON ch.user_id = cm.user_id AND ch.timestamp = cm.timestamp
                WHERE ch.timestamp BETWEEN ? AND ?
                """,
                (day_start, day_end)
            )
            
            row = cursor.fetchone()
            stats.append({
                'date': day_start.strftime('%Y-%m-%d'),
                'queries': row[0] or 0,
                'users': row[1] or 0,
                'input_tokens': row[2] or 0,
                'output_tokens': row[3] or 0,
                'cost': row[4] or 0
            })
    
    return jsonify({'stats': stats})

@app.route('/api/history/stats')
@login_required
def history_stats():
    """API endpoint for chat history statistics"""
    db_ops = DatabaseOps()
    
    daily_queries = []
    with sqlite3.connect(db_ops.db_path) as conn:
        cursor = conn.execute(
            """
            SELECT date(timestamp) as date, COUNT(*) as count
            FROM chat_history
            WHERE timestamp >= datetime('now', '-30 days')
            GROUP BY date(timestamp)
            ORDER BY date(timestamp)
            """
        )
        daily_queries = [{'date': row[0], 'count': row[1]} for row in cursor.fetchall()]
    
    model_usage = []
    with sqlite3.connect(db_ops.db_path) as conn:
        cursor = conn.execute(
            """
            SELECT model_used, COUNT(*) as count
            FROM chat_history
            GROUP BY model_used
            ORDER BY count DESC
            """
        )
        model_usage = [{'model': row[0], 'count': row[1]} for row in cursor.fetchall()]
    
    return jsonify({
        'daily_queries': daily_queries,
        'model_usage': model_usage
    })

@app.route('/content/files')
@login_required
def list_content_files():
    """API endpoint to list content files"""
    content_files = []
    for file_path in WEB_CONTENT_DIR.glob('*.txt'):
        content_size = file_path.stat().st_size / 1024  # KB
        content_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
        content_files.append({
            'name': file_path.name,
            'size_kb': content_size,
            'modified': content_modified.strftime('%Y-%m-%d %H:%M:%S'),
            'path': str(file_path),
        })
    
    return jsonify({'files': content_files})

@app.route('/content/test', methods=['POST'])
@login_required
def test_content():
    """API endpoint to test RAG system with content"""
    from chatbot.utils.admin_utils import AdminUtils
    
    data = request.json
    query = data.get('query')
    filename = data.get('filename')
    provider = data.get('provider')
    model = data.get('model')
    rerank = data.get('rerank', True)
    
    if not query or not filename or not provider or not model:
        return jsonify({
            'success': False,
            'error': 'Missing required parameters'
        })
    
    content_path = WEB_CONTENT_DIR / filename
    if not content_path.exists():
        return jsonify({
            'success': False,
            'error': f'Content file not found: {filename}'
        })
    
    admin_utils = AdminUtils()
    success, response, time_taken = asyncio.run(
        admin_utils.test_rag_system(
            query=query,
            content_path=content_path,
            model_name=model,
            provider=provider,
            chunking_type="recursive",
            rerank=rerank
        )
    )
    
    return jsonify({
        'success': success,
        'response': response,
        'time_taken': time_taken
    })

@app.route('/models/test', methods=['POST'])
@login_required
def test_model():
    """API endpoint to test a specific model configuration"""
    from chatbot.utils.admin_utils import AdminUtils
    
    data = request.json
    query = data.get('query')
    filename = data.get('filename')
    provider = data.get('provider')
    model = data.get('model')
    chunking_type = data.get('chunking_type', 'recursive')
    rerank = data.get('rerank', True)
    
    if not query or not filename or not provider or not model:
        return jsonify({
            'success': False,
            'error': 'Missing required parameters'
        })
    
    content_path = WEB_CONTENT_DIR / filename
    if not content_path.exists():
        return jsonify({
            'success': False,
            'error': f'Content file not found: {filename}'
        })
    
    admin_utils = AdminUtils()
    success, response, time_taken = asyncio.run(
        admin_utils.test_rag_system(
            query=query,
            content_path=content_path,
            model_name=model,
            provider=provider,
            chunking_type=chunking_type,
            rerank=rerank
        )
    )
    
    return jsonify({
        'success': success,
        'response': response,
        'time_taken': time_taken
    })

@app.route('/update_api_keys', methods=['POST'])
@login_required
def update_api_keys():
    """Update API keys for various providers"""
    import os
    import dotenv
    
    dotenv_path = Path(__file__).parent.parent.parent / '.env'
    
    # Load existing .env file if it exists
    if dotenv_path.exists():
        dotenv.load_dotenv(dotenv_path)
    
    # Update environment variables with new API keys
    api_keys = {
        'COHERE_API': request.form.get('cohere_api_key'),
        'OPENAI_API': request.form.get('openai_api_key'),
        'ANTHROPIC_API': request.form.get('anthropic_api_key'),
        'GOOGLE_API': request.form.get('google_api_key'),
        'MISTRAL_API': request.form.get('mistral_api_key'),
        'XAI_API': request.form.get('xai_api_key'),
        'DEEPSEEK_API': request.form.get('deepseek_api_key'),
    }
    
    # Only update keys that have a value
    updated_keys = {}
    for key, value in api_keys.items():
        if value:
            os.environ[key] = value
            updated_keys[key] = value
    
    # Save to .env file
    if updated_keys:
        for key, value in updated_keys.items():
            dotenv.set_key(dotenv_path, key, value)
    
    flash('API keys updated successfully', 'success')
    return redirect(url_for('models'))

@app.route('/export_user_data/<user_id>')
@login_required
def export_user_data(user_id):
    """Export user data in specified format"""
    import csv
    import json
    import pandas as pd
    from io import StringIO, BytesIO
    
    format_type = request.args.get('format', 'json')
    
    db_ops = DatabaseOps()
    user_history = db_ops.get_chat_history(user_id=user_id, full_history=True)
    
    # Convert to flat structure for CSV/Excel export
    flat_data = []
    for interaction in user_history[0].get('interactions', []):
        flat_data.append({
            'user_id': user_id,
            'timestamp': interaction.get('timestamp'),
            'question': interaction.get('user'),
            'answer': interaction.get('assistant'),
            'model': interaction.get('llm'),
            'embedder': interaction.get('embedder')
        })
    
    if format_type == 'json':
        response = jsonify(user_history)
        response.headers['Content-Disposition'] = f'attachment; filename={user_id}_export.json'
        return response
    
    elif format_type == 'csv':
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=['user_id', 'timestamp', 'question', 'answer', 'model', 'embedder'])
        writer.writeheader()
        writer.writerows(flat_data)
        
        response = app.response_class(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename={user_id}_export.csv'}
        )
        return response
    
    elif format_type == 'xlsx':
        output = BytesIO()
        df = pd.DataFrame(flat_data)
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='User Data', index=False)
        
        output.seek(0)
        return send_file(
            output,
            as_attachment=True,
            download_name=f'{user_id}_export.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    
    flash('Unsupported export format', 'danger')
    return redirect(url_for('user_detail', user_id=user_id))

@app.route('/delete_user_data/<user_id>', methods=['POST'])
@login_required
def delete_user_data(user_id):
    """Delete all data for a specific user"""
    with sqlite3.connect(DatabaseOps().db_path) as conn:
        conn.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM cost_monitor WHERE user_id = ?", (user_id,))
        conn.commit()
    
    flash(f'All data for user {user_id} has been deleted', 'success')
    return redirect(url_for('users'))

@app.route('/export_history')
@login_required
def export_history():
    """Export chat history in specified format and timeframe"""
    import csv
    import json
    import pandas as pd
    from io import StringIO, BytesIO
    
    format_type = request.args.get('format', 'json')
    timeframe = request.args.get('timeframe', '24h')
    include_tokens = 'include_tokens' in request.args
    
    # Determine time window based on timeframe
    time_window = None
    if timeframe == '24h':
        time_window = datetime.now() - timedelta(hours=24)
    elif timeframe == '7d':
        time_window = datetime.now() - timedelta(days=7)
    elif timeframe == '30d':
        time_window = datetime.now() - timedelta(days=30)
    
    db_ops = DatabaseOps()
    history = db_ops.get_chat_history(full_history=True, last_n_hours=None if timeframe == 'all' else int(timeframe.rstrip('dhm')))
    
    # Convert to flat structure for CSV/Excel export
    flat_data = []
    for user in history:
        for interaction in user.get('interactions', []):
            entry = {
                'user_id': user.get('user_id'),
                'timestamp': interaction.get('timestamp'),
                'question': interaction.get('user'),
                'answer': interaction.get('assistant'),
                'model': interaction.get('llm'),
                'embedder': interaction.get('embedder')
            }
            
            # Add token usage if requested
            if include_tokens:
                with sqlite3.connect(db_ops.db_path) as conn:
                    cursor = conn.execute(
                        """
                        SELECT input_tokens, output_tokens, request_cost
                        FROM cost_monitor
                        WHERE user_id = ? AND timestamp = ?
                        """,
                        (user.get('user_id'), interaction.get('timestamp'))
                    )
                    row = cursor.fetchone()
                    if row:
                        entry['input_tokens'] = row[0]
                        entry['output_tokens'] = row[1]
                        entry['cost'] = row[2]
            
            flat_data.append(entry)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if format_type == 'json':
        response = jsonify({
            'export_date': datetime.now().isoformat(),
            'timeframe': timeframe,
            'include_tokens': include_tokens,
            'data': history
        })
        response.headers['Content-Disposition'] = f'attachment; filename=chat_history_{timestamp}.json'
        return response
    
    elif format_type == 'csv':
        output = StringIO()
        fieldnames = ['user_id', 'timestamp', 'question', 'answer', 'model', 'embedder']
        if include_tokens:
            fieldnames.extend(['input_tokens', 'output_tokens', 'cost'])
            
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_data)
        
        response = app.response_class(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename=chat_history_{timestamp}.csv'}
        )
        return response
    
    elif format_type == 'xlsx':
        output = BytesIO()
        df = pd.DataFrame(flat_data)
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Chat History', index=False)
            
            # Add summary sheet
            summary_data = {
                'Metric': [
                    'Export Date',
                    'Timeframe',
                    'Total Users',
                    'Total Conversations',
                ],
                'Value': [
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    timeframe,
                    len({user.get('user_id') for user in history}),
                    sum(len(user.get('interactions', [])) for user in history),
                ]
            }
            
            if include_tokens:
                summary_data['Metric'].extend([
                    'Total Input Tokens',
                    'Total Output Tokens',
                    'Total Cost'
                ])
                summary_data['Value'].extend([
                    sum(entry.get('input_tokens', 0) for entry in flat_data),
                    sum(entry.get('output_tokens', 0) for entry in flat_data),
                    sum(entry.get('cost', 0) for entry in flat_data)
                ])
            
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        output.seek(0)
        return send_file(
            output,
            as_attachment=True,
            download_name=f'chat_history_{timestamp}.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    
    flash('Unsupported export format', 'danger')
    return redirect(url_for('chat_history'))

@app.route('/system/logs')
@login_required
def get_logs():
    """API endpoint to get system logs"""
    log_file = LOGS_DIR / 'logs.log'
    logs = []
    
    if log_file.exists():
        with open(log_file, 'r') as f:
            logs = [line.strip() for line in f.readlines()[-100:]]  # Get last 100 lines
    
    return jsonify({'logs': logs})

@app.route('/system/logs/download')
@login_required
def download_logs():
    """Download system logs as a file"""
    log_file = LOGS_DIR / 'logs.log'
    
    if not log_file.exists():
        flash('Log file not found', 'danger')
        return redirect(url_for('system'))
    
    return send_file(
        log_file,
        as_attachment=True,
        download_name=f'system_logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )

@app.route('/system/backup')
@login_required
def backup_system():
    """Backup system data"""
    import shutil
    import tempfile
    
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Copy database
            shutil.copy2(DatabaseOps().db_path, temp_path / 'history_and_usage.db')
            shutil.copy2(app.config['ADMIN_DB_PATH'], temp_path / 'admin.db')
            
            # Create zip file
            backup_filename = f'rag_system_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
            shutil.make_archive(str(temp_path / 'backup'), 'zip', temp_path)
            
            return send_file(
                str(temp_path / 'backup.zip'),
                as_attachment=True,
                download_name=backup_filename
            )
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        flash(f'Error creating backup: {str(e)}', 'danger')
        return redirect(url_for('system'))

@app.route('/system/clear-cache', methods=['POST'])
@login_required
def clear_cache():
    """Clear system cache"""
    try:
        # Clear any temporary files
        import glob
        import os
        
        # Clear temp index files
        for temp_file in glob.glob(str(INDEXES_DIR / '*.temp')):
            os.remove(temp_file)
        
        # Clear temp progress files
        for temp_file in glob.glob(str(INDEXES_DIR / '*.progress')):
            os.remove(temp_file)
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/system/reset', methods=['POST'])
@login_required
def reset_system():
    """Reset the entire system"""
    try:
        with sqlite3.connect(DatabaseOps().db_path) as conn:
            # Delete all chat history and usage data
            conn.execute("DELETE FROM chat_history")
            conn.execute("DELETE FROM cost_monitor")
            conn.commit()
        
        # Reset system configurations to defaults
        with sqlite3.connect(app.config['ADMIN_DB_PATH']) as conn:
            default_configs = [
                ("default_llm_provider", "cohere"),
                ("default_llm_model", "command-r-plus-08-2024"),
                ("default_chunking_type", "recursive"),
                ("default_chunk_size", str(RAGConfig.CHUNK_SIZE)),
                ("default_chunk_overlap", str(RAGConfig.CHUNK_OVERLAP)),
                ("rerank_enabled", "true"),
                ("email_notifications_enabled", "true"),
            ]
            
            for key, value in default_configs:
                conn.execute(
                    "UPDATE system_config SET config_value = ?, updated_at = ? WHERE config_key = ?",
                    (value, datetime.now(), key)
                )
            
            conn.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error resetting system: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/system/status')
@login_required
def system_status():
    """Get system component status"""
    try:
        # Check database connection
        db_status = True
        try:
            with sqlite3.connect(DatabaseOps().db_path) as conn:
                conn.execute("SELECT 1")
        except:
            db_status = False
        
        # Check embedding service
        embedding_status = True
        try:
            from chatbot.embeddings.base_embedding import CohereEmbedding
            embedding = CohereEmbedding(os.getenv("COHERE_API"))
            embedding.embed(["test"], is_query=True)
        except:
            embedding_status = False
        
        # Check LLM connection
        llm_status = True
        try:
            import cohere
            client = cohere.Client(os.getenv("COHERE_API"))
            client.chat(message="test")
        except:
            llm_status = False
        
        # Check email service
        email_status = True
        try:
            email_service = EmailService()
            if not email_service.app_password:
                email_status = False
        except:
            email_status = False
        
        # Check file storage
        storage_status = True
        try:
            WEB_CONTENT_DIR.exists()
            INDEXES_DIR.exists()
        except:
            storage_status = False
        
        return jsonify({
            'status': [db_status, embedding_status, llm_status, email_status, storage_status]
        })
    except Exception as e:
        logger.error(f"Error checking system status: {e}")
        return jsonify({
            'status': [False, False, False, False, False],
            'error': str(e)
        })

@app.route('/account', methods=['GET', 'POST'])
@login_required
def account_settings():
    """User account settings to change username and password"""
    error = None
    success = None
    
    # Get current user info
    user_id = session.get('user_id')
    
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'change_username':
            new_username = request.form.get('new_username')
            current_password = request.form.get('current_password_for_username')
            
            if not new_username or not current_password:
                error = 'All fields are required'
            else:
                # Verify current password
                with sqlite3.connect(app.config['ADMIN_DB_PATH']) as conn:
                    cursor = conn.execute(
                        "SELECT password_hash FROM admin_users WHERE id = ?", 
                        (user_id,)
                    )
                    user = cursor.fetchone()
                    
                    if user and check_password_hash(user[0], current_password):
                        # Check if new username already exists
                        cursor = conn.execute(
                            "SELECT COUNT(*) FROM admin_users WHERE username = ? AND id != ?", 
                            (new_username, user_id)
                        )
                        if cursor.fetchone()[0] > 0:
                            error = 'Username already exists'
                        else:
                            # Update username
                            conn.execute(
                                "UPDATE admin_users SET username = ? WHERE id = ?",
                                (new_username, user_id)
                            )
                            conn.commit()
                            session['username'] = new_username
                            success = 'Username updated successfully'
                    else:
                        error = 'Current password is incorrect'
        
        elif action == 'change_password':
            current_password = request.form.get('current_password')
            new_password = request.form.get('new_password')
            confirm_password = request.form.get('confirm_password')
            
            if not current_password or not new_password or not confirm_password:
                error = 'All fields are required'
            elif new_password != confirm_password:
                error = 'New passwords do not match'
            elif len(new_password) < 8:
                error = 'New password must be at least 8 characters long'
            else:
                # Verify current password
                with sqlite3.connect(app.config['ADMIN_DB_PATH']) as conn:
                    cursor = conn.execute(
                        "SELECT password_hash FROM admin_users WHERE id = ?", 
                        (user_id,)
                    )
                    user = cursor.fetchone()
                    
                    if user and check_password_hash(user[0], current_password):
                        # Update password
                        new_password_hash = generate_password_hash(new_password)
                        conn.execute(
                            "UPDATE admin_users SET password_hash = ? WHERE id = ?",
                            (new_password_hash, user_id)
                        )
                        conn.commit()
                        success = 'Password updated successfully'
                    else:
                        error = 'Current password is incorrect'
    
    # Get current user info for display
    with sqlite3.connect(app.config['ADMIN_DB_PATH']) as conn:
        cursor = conn.execute(
            "SELECT username, email, created_at, last_login FROM admin_users WHERE id = ?", 
            (user_id,)
        )
        user_info = cursor.fetchone()
    
    if error:
        flash(error, 'danger')
    if success:
        flash(success, 'success')
    
    return render_template(
        'admin/account.html',
        username=user_info[0],
        email=user_info[1],
        created_at=user_info[2],
        last_login=user_info[3]
    )


@app.route("/data-analysis")
@login_required
def data_analysis():
    """Data analysis dashboard page - Chat History Analyzer"""
    db_ops = DatabaseOps()

    # Get conversation count
    with sqlite3.connect(db_ops.db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM chat_history")
        conversation_count = cursor.fetchone()[0]

    # Check if we have a record of the last analysis
    last_analysis = None
    with sqlite3.connect(app.config["ADMIN_DB_PATH"]) as conn:
        cursor = conn.execute(
            "SELECT config_value FROM system_config WHERE config_key = 'last_analysis_time'"
        )
        result = cursor.fetchone()
        if result:
            try:
                last_analysis = datetime.fromisoformat(result[0]).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            except:
                last_analysis = None

    return render_template(
        "admin/data_analysis.html",
        conversation_count=conversation_count,
        last_analysis=last_analysis,
        analysis_result=None,
        thinking_steps=None,
        query=None,
        time_period=None,
    )


@app.route("/analyze-data", methods=["POST"])
@login_required
def analyze_data():
    """Process data analysis request"""
    query = request.form.get("analysis_query")

    # Get time period and enforce limits
    try:
        time_period = int(request.form.get("time_period", 7))
        # Enforce limits (1-90 days)
        time_period = max(1, min(90, time_period))
    except (ValueError, TypeError):
        time_period = 7  # Default to 7 days if invalid input

    if not query:
        flash("Please provide an analysis query", "warning")
        return redirect(url_for("data_analysis"))

    try:
        # Initialize the agent
        agent = ChatHistortAnalyzerAgent()

        # Capture the agent's thinking process
        f = io.StringIO()
        with redirect_stdout(f):
            # Update the query to include time period information
            enhanced_query = f"{query} for the past {time_period} days"

            # Run the analysis
            result = agent.analyze(enhanced_query)

        # Get the thinking steps
        thinking_steps = f.getvalue()

        # Extract the actual response content
        analysis_result = result.get("output", "")

        # Convert markdown to HTML
        import markdown

        analysis_html = markdown.markdown(analysis_result)

        # Update the last analysis time
        with sqlite3.connect(app.config["ADMIN_DB_PATH"]) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO system_config (config_key, config_value, updated_at)
                VALUES (?, ?, ?)
                """,
                ("last_analysis_time", datetime.now().isoformat(), datetime.now()),
            )
            conn.commit()

        # Get conversation count
        db_ops = DatabaseOps()
        with sqlite3.connect(db_ops.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM chat_history")
            conversation_count = cursor.fetchone()[0]

        # Get last analysis time
        with sqlite3.connect(app.config["ADMIN_DB_PATH"]) as conn:
            cursor = conn.execute(
                "SELECT config_value FROM system_config WHERE config_key = 'last_analysis_time'"
            )
            result = cursor.fetchone()
            last_analysis = (
                datetime.fromisoformat(result[0]).strftime("%Y-%m-%d %H:%M:%S")
                if result
                else None
            )

        return render_template(
            "admin/data_analysis.html",
            analysis_result=analysis_html,
            thinking_steps=thinking_steps,
            conversation_count=conversation_count,
            last_analysis=last_analysis,
            query=query,
            time_period=time_period,
        )

    except Exception as e:
        logger.error(f"Error processing data analysis: {e}")
        import traceback

        logger.error(traceback.format_exc())
        flash(f"Error processing analysis: {str(e)}", "danger")
        return redirect(url_for("data_analysis"))


@app.route("/save-analysis", methods=["POST"])
@login_required
def save_analysis():
    """Save analysis results to file"""
    try:
        title = request.form.get(
            "title", f"Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        description = request.form.get("description", "")
        content = request.form.get("content", "")
        query = request.form.get("query", "")
        time_period = request.form.get("time_period", "")

        # Create analysis directory if it doesn't exist
        analysis_dir = Path(DATA_DIR) / "analysis"
        if not analysis_dir.exists():
            os.makedirs(analysis_dir)

        # Sanitize filename
        safe_title = "".join(c for c in title if c.isalnum() or c in "._- ").replace(
            " ", "_"
        )

        # Create a file with metadata and content
        filename = analysis_dir / f"{safe_title}.html"

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .metadata {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .content {{ padding: 20px; }}
    </style>
</head>
<body>
    <div class="metadata">
        <h1>{title}</h1>
        <p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>Query:</strong> {query}</p>
        <p><strong>Time Period:</strong> {time_period} days</p>
        <p><strong>Description:</strong> {description}</p>
    </div>
    <div class="content">
        {content}
    </div>
</body>
</html>""")

        logger.info(f"Analysis saved to {filename}")
        return jsonify({"success": True, "filename": str(filename)})

    except Exception as e:
        logger.error(f"Error saving analysis: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return jsonify({"success": False, "message": str(e)})


@app.route("/save-agent-steps", methods=["POST"])
@login_required
def save_agent_steps():
    """Save agent thinking steps to file"""
    try:
        title = request.form.get(
            "title", f"AgentSteps_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        content = request.form.get("content", "")
        query = request.form.get("query", "")

        # Create analysis directory if it doesn't exist
        analysis_dir = Path(DATA_DIR) / "analysis"
        if not analysis_dir.exists():
            os.makedirs(analysis_dir)

        # Sanitize filename
        safe_title = "".join(c for c in title if c.isalnum() or c in "._- ").replace(
            " ", "_"
        )

        # Create a text file with the steps
        filename = analysis_dir / f"{safe_title}.txt"

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Query: {query}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 80 + "\n\n")
            f.write(content)

        logger.info(f"Agent steps saved to {filename}")
        return jsonify({"success": True, "filename": str(filename)})

    except Exception as e:
        logger.error(f"Error saving agent steps: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return jsonify({"success": False, "message": str(e)})


if __name__ == "__main__":
    init_admin_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
