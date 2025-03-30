Usage Guide
===========

Getting Started
---------------

After installation, you can use DataVerse ChatBot in several different ways:

1. **CLI Chat Interface**: Interact directly via the command line
2. **Web Interface**: Embed or deploy as a web application
3. **Telegram Bot**: Deploy as a Telegram bot
4. **WhatsApp Bot**: Deploy as a WhatsApp bot via Twilio
5. **Admin Dashboard**: Manage the system through a web-based admin panel

Running the CLI Interface
-------------------------

The simplest way to start is with the command-line interface:

.. code-block:: bash

   python src/main.py

This will:

1. Load the configured content
2. Create or load the vector index
3. Start an interactive chat session

Data Ingestion Methods
----------------------

Web Crawling
~~~~~~~~~~~~

To extract content from a website:

.. code-block:: python

   from chatbot.crawler import Crawler
   from chatbot.utils.paths import WEB_CONTENT_DIR
   import asyncio
   import tldextract

   url = "https://example.com"
   domain_name = tldextract.extract(url).domain
   
   # Create crawler instance (options: "crawl4ai" or "scrapegraph")
   crawler = Crawler(url, domain_name, client="crawl4ai")
   
   # Extract content (webpage_only=False to crawl linked pages)
   content_path = asyncio.run(crawler.extract_content(
       url, webpage_only=False, max_depth=2



   print(f"Content saved to: {content_path}")

File Loading
~~~~~~~~~~~~

To extract content from files:

.. code-block:: python

   from chatbot.utils.file_loader import FileLoader
   from chatbot.utils.paths import WEB_CONTENT_DIR
   
   # Path to your file (supports PDF, DOCX, CSV, XLSX, TXT, PPT, etc.)
   file_path = "data/training_files/document.pdf"
   
   # Output path for the content
   output_path = WEB_CONTENT_DIR / "extracted_content.txt"
   
   # Create loader (options: "langchain" or "docling")
   loader = FileLoader(file_path, output_path, client="docling")
   
   # Extract content
   documents = loader.extract_from_file()
   
   if documents:
       print(f"Successfully extracted {len(documents)} documents")

Creating a RAG System
---------------------

Choose your preferred LLM to create a RAG system:

.. code-block:: python

   from chatbot.rag.openai_rag import OpenAIRAG
   from chatbot.rag.claude_rag import ClaudeRAG
   from chatbot.rag.cohere_rag import CohereRAG
   from chatbot.utils.paths import WEB_CONTENT_DIR, INDEXES_DIR
   
   # Path to your content
   content_path = WEB_CONTENT_DIR / "example.txt"
   
   # Create RAG instance (example with Claude)
   rag = ClaudeRAG(
       content_path,          # Content to use for RAG
       INDEXES_DIR,           # Where to store vector indexes
       model_name="claude-3-5-sonnet-20241022",  # Specific model to use
       chunking_type="recursive",  # Chunking strategy (options: "recursive", "semantic", "basic")
       rerank=True            # Whether to use Cohere's reranking model


   
   # Get a response (asynchronous)
   import asyncio
   user_id = "user123"        # Used for chat history
   query = "Tell me about DataVerse."
   
   response = asyncio.run(rag.get_response(query, user_id))
   print(response)

Deploying as a Web Interface
----------------------------

To run the web interface:

.. code-block:: bash

   python src/web/chat_web_app.py

This starts a FastAPI server on port 5001. You can access the chat interface at:
http://localhost:5001/

Deploying as a Telegram Bot
---------------------------

To start the Telegram bot:

.. code-block:: bash

   python src/tg_bot.py

Ensure you've set up the Telegram Bot API token in your `.env` file:

.. code-block:: ini

   # Telegram Bot
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token

Deploying as a WhatsApp Bot
---------------------------

To deploy as a WhatsApp bot via Twilio:

.. code-block:: bash

   python src/whatsapp_bot.py

Configure your Twilio credentials in the `.env` file:

.. code-block:: ini

   # Twilio for WhatsApp
   TWILIO_ACCOUNT_SID=your_twilio_account_sid
   TWILIO_AUTH_TOKEN=your_twilio_auth_token
   TWILIO_PHONE_NUMBER=your_twilio_phone_number

Running the Admin Dashboard
---------------------------

To launch the admin dashboard:

.. code-block:: bash

   python src/admin_dashboard_launcher.py

Navigate to: http://localhost:8050/ to access the dashboard.

Default credentials are:
- Username: `admin`
- Password: `password`

(Change these in production!)

Advanced Features
-----------------

Voice Mode
~~~~~~~~~~

DataVerse ChatBot supports voice interaction:

.. code-block:: python

   from chatbot.voice_mode import VoiceMode
   
   voice_mode = VoiceMode()
   
   # Record audio and transcribe
   wav_path = voice_mode.start_recording()
   transcribed_text = voice_mode.transcribe(wav_path)
   
   # Convert text response to speech
   voice_mode.text_to_speech("This is a spoken response.")

Creating a Custom Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~

To create a dataset for uncertainty classification:

.. code-block:: bash

   python src/chatbot/utils/make_dataset.py

Training a Classifier
~~~~~~~~~~~~~~~~~~~~~

To train a classifier for uncertainty detection:

.. code-block:: bash

   python src/chatbot/utils/train_clf.py