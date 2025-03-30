Overview
========

System Architecture
-------------------

DataVerse ChatBot follows a modular architecture designed with extensibility and maintainability in mind:

.. code-block:: text

   ┌─────────────────┐       ┌───────────────────┐       ┌─────────────────┐
   │  Data Ingestion │       │ Vector Processing │       │   LLM Systems   │
   │                 │       │                   │       │                 │
   │ - Web Crawling  │──────▶│ - Embeddings     │──────▶│ - RAG Interface │
   │ - File Loading  │       │ - Vectorization  │       │ - LLM Providers │
   │                 │       │ - FAISS Indexing │       │                 │
   └─────────────────┘       └───────────────────┘       └─────────────────┘
           ▲                                                     │
           │                                                     ▼
           │                 ┌───────────────────┐       ┌─────────────────┐
           │                 │  Chat Interfaces  │       │   Monitoring    │
           └─────────────────│                   │◀──────│                 │
                             │ - Telegram Bot    │       │ - Uncertainty   │
                             │ - WhatsApp Bot    │       │ - Email Alerts  │
                             │ - Web Interface   │       │ - Chat History  │
                             └───────────────────┘       └─────────────────┘

Key Components
--------------

1. **Data Ingestion**
   
   * `Crawler`: Extracts content from websites using configurable crawlers
   * `FileLoader`: Processes various file formats into uniform text representations

2. **Vector Processing**
   
   * `BaseEmbedding`: Creates vector embeddings from text
   * `CohereEmbedding`, `OpenAIEmbedding`, etc.: Provider-specific embedding implementations
   * `FAISS` integration: Efficient similarity search for document retrieval

3. **RAG Systems**
   
   * `BaseRAG`: Core Retrieval-Augmented Generation functionality
   * `ClaudeRAG`, `OpenAIRAG`, etc.: LLM-specific implementations
   * Context management and reranking

4. **Chat Interfaces**
   
   * Telegram bot for messenger integration
   * WhatsApp bot via Twilio integration
   * Web-based chat interface with iframe embedding support

5. **Monitoring & Utilities**
   
   * Uncertainty detection via trained classifier
   * Response monitoring and email alerting
   * Database operations for chat history and usage tracking

Technology Stack
----------------

* **Programming Language**: Python 3.11+
* **Vector Database**: FAISS (Facebook AI Similarity Search)
* **LLM Providers**: OpenAI, Anthropic, Google, Mistral, Cohere, DeepSeek, Grok
* **Web Crawling**: Crawl4AI, ScrapegraphAI
* **Data Processing**: LangChain, DocLing, unstructured
* **Classification**: scikit-learn, XGBoost
* **Embedding Models**: sentence-transformers, provider-specific embedding APIs
* **Web Framework**: FastAPI, Flask
* **Messenger Integrations**: python-telegram-bot, Twilio
* **Database**: SQLite
* **Voice Support**: OpenAI Whisper, TTS libraries