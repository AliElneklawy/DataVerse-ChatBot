API Reference
=============

This section provides detailed API documentation for the DataVerse ChatBot components.

chatbot Package
---------------

rag Module
~~~~~~~~~~

BaseRAG
'''''''

.. py:class:: BaseRAG

   The base class for all RAG implementations.

   .. py:method:: __init__(content_path, index_path=None, rerank=True, model_name=None, chunking_type="recursive")

      Initialize the RAG system.

      :param content_path: Path to content files
      :param index_path: Path to store/load vector indexes
      :param rerank: Whether to enable reranking
      :param model_name: LLM model name
      :param chunking_type: Method for chunking text

   .. py:method:: get_response(query, user_id)

      Get a response for the user query.

      :param query: User query text
      :param user_id: Unique user identifier
      :return: Generated response

   .. py:method:: _create_embeddings(texts, is_query=False)

      Create embeddings for text chunks.

      :param texts: List of text chunks
      :param is_query: Whether these are query embeddings
      :return: List of embeddings

   .. py:method:: _generate_system_prompt(query, user_id, context, include_query=True, include_context=True, include_prev_conv=True)

      Generate a standardized system prompt for all LLMs.

      :param query: User query
      :param user_id: User identifier
      :param context: Retrieved context
      :param include_query: Whether to include the query
      :param include_context: Whether to include context
      :param include_prev_conv: Whether to include previous conversation
      :return: Formatted system prompt

   .. py:method:: _get_index_path(content_path)

      Generate unique index path based on content.

      :param content_path: Path to content
      :return: Path to index

   .. py:method:: _clean_html_content(content)

      Clean HTML content and convert to markdown.

      :param content: HTML content
      :return: Cleaned markdown content

   .. py:method:: _create_chunks(text)

      Create chunks using the specified chunking method.

      :param text: Text to chunk
      :return: List of text chunks

   .. py:method:: _load_or_create_vectorstore(content_path)

      Load existing index or create new one.

      :param content_path: Path to content
      :return: FAISS vectorstore

   .. py:method:: _create_vectorstore(content_path)

      Create FAISS vectorstore from content with incremental embedding saving.

      :param content_path: Path to content
      :return: FAISS vectorstore

   .. py:method:: _update_vectorstore(new_content)

      Update existing vectorstore with new content.

      :param new_content: New content to add
      :return: None

   .. py:method:: _save_vectorstore(vectorstore, path)

      Save vectorstore to disk.

      :param vectorstore: FAISS vectorstore
      :param path: Path to save to
      :return: None

   .. py:method:: _load_vectorstore(path)

      Load vectorstore from disk.

      :param path: Path to load from
      :return: FAISS vectorstore

   .. py:method:: _rerank_docs(query, docs)

      Refine the top-k retrieved chunks for relevance.

      :param query: User query
      :param docs: Retrieved documents
      :return: Reranked documents

   .. py:method:: _find_relevant_context(query, top_k=5)

      Find relevant context using similarity search.

      :param query: User query
      :param top_k: Number of top chunks to retrieve
      :return: Relevant context as string

   .. py:classmethod:: get_models()

      Get available models for this RAG implementation.

      :return: List of available models

   .. py:classmethod:: get_config_class()

      Get the configuration class for this RAG implementation.

      :return: Configuration class

ClaudeRAG
'''''''''

.. py:class:: ClaudeRAG

   RAG implementation using Anthropic's Claude models.

   .. py:method:: __init__(content_path, index_path=None, rerank=True, model_name=None, chunking_type="recursive")

      Initialize the Claude RAG system.

      :param content_path: Path to content files
      :param index_path: Path to store/load vector indexes
      :param rerank: Whether to enable reranking
      :param model_name: Claude model name
      :param chunking_type: Method for chunking text

   .. py:method:: get_response(query, user_id)

      Get a Claude-powered response.

      :param query: User query
      :param user_id: User identifier
      :return: Generated response

   .. py:method:: _initialize_models()

      Initialize Claude API client.

      :return: None

   .. py:classmethod:: get_config_class()

      Get the Claude configuration class.

      :return: ClaudeConfig class

OpenAIRAG
'''''''''

.. py:class:: OpenAIRAG

   RAG implementation using OpenAI's models.

   .. py:method:: __init__(content_path, index_path=None, rerank=True, model_name=None, chunking_type="recursive")

      Initialize the OpenAI RAG system.

      :param content_path: Path to content files
      :param index_path: Path to store/load vector indexes
      :param rerank: Whether to enable reranking
      :param model_name: OpenAI model name
      :param chunking_type: Method for chunking text

   .. py:method:: get_response(query, user_id)

      Get an OpenAI-powered response.

      :param query: User query
      :param user_id: User identifier
      :return: Generated response

   .. py:method:: _initialize_models()

      Initialize OpenAI API client.

      :return: None

   .. py:classmethod:: get_config_class()

      Get the OpenAI configuration class.

      :return: OpenAIConfig class

CohereRAG
'''''''''

.. py:class:: CohereRAG

   RAG implementation using Cohere's models.

   .. py:method:: __init__(content_path, index_path=None, rerank=True, model_name=None, chunking_type="recursive")

      Initialize the Cohere RAG system.

      :param content_path: Path to content files
      :param index_path: Path to store/load vector indexes
      :param rerank: Whether to enable reranking
      :param model_name: Cohere model name
      :param chunking_type: Method for chunking text

   .. py:method:: get_response(query, user_id)

      Get a Cohere-powered response.

      :param query: User query
      :param user_id: User identifier
      :return: Generated response

   .. py:method:: _initialize_models()

      Initialize Cohere API client.

      :return: None

   .. py:classmethod:: get_config_class()

      Get the Cohere configuration class.

      :return: CohereConfig class

GeminiRAG
'''''''''

.. py:class:: GeminiRAG

   RAG implementation using Google's Gemini models.

   .. py:method:: __init__(content_path, index_path=None, rerank=True, model_name=None, chunking_type="recursive")

      Initialize the Gemini RAG system.

      :param content_path: Path to content files
      :param index_path: Path to store/load vector indexes
      :param rerank: Whether to enable reranking
      :param model_name: Gemini model name
      :param chunking_type: Method for chunking text

   .. py:method:: get_response(query, user_id)

      Get a Gemini-powered response.

      :param query: User query
      :param user_id: User identifier
      :return: Generated response

   .. py:method:: _initialize_models()

      Initialize Gemini API client.

      :return: None

   .. py:classmethod:: get_config_class()

      Get the Gemini configuration class.

      :return: GeminiConfig class

MistralRAG
''''''''''

.. py:class:: MistralRAG

   RAG implementation using Mistral AI models.

   .. py:method:: __init__(content_path, index_path=None, rerank=True, model_name=None, chunking_type="recursive")

      Initialize the Mistral RAG system.

      :param content_path: Path to content files
      :param index_path: Path to store/load vector indexes
      :param rerank: Whether to enable reranking
      :param model_name: Mistral model name
      :param chunking_type: Method for chunking text

   .. py:method:: get_response(query, user_id)

      Get a Mistral-powered response.

      :param query: User query
      :param user_id: User identifier
      :return: Generated response

   .. py:method:: _initialize_models()

      Initialize Mistral API client.

      :return: None

   .. py:classmethod:: get_config_class()

      Get the Mistral configuration class.

      :return: MistralConfig class

DeepseekRAG
'''''''''''

.. py:class:: DeepseekRAG

   RAG implementation using Deepseek models.

   .. py:method:: __init__(content_path, index_path=None, rerank=True, model_name=None, chunking_type="recursive")

      Initialize the Deepseek RAG system.

      :param content_path: Path to content files
      :param index_path: Path to store/load vector indexes
      :param rerank: Whether to enable reranking
      :param model_name: Deepseek model name
      :param chunking_type: Method for chunking text

   .. py:method:: get_response(query, user_id)

      Get a Deepseek-powered response.

      :param query: User query
      :param user_id: User identifier
      :return: Generated response

   .. py:method:: _initialize_models()

      Initialize Deepseek API client.

      :return: None

   .. py:classmethod:: get_config_class()

      Get the Deepseek configuration class.

      :return: DeepSeekConfig class

GrokRAG
'''''''

.. py:class:: GrokRAG

   RAG implementation using Grok models.

   .. py:method:: __init__(content_path, index_path=None, rerank=True, model_name=None, chunking_type="recursive")

      Initialize the Grok RAG system.

      :param content_path: Path to content files
      :param index_path: Path to store/load vector indexes
      :param rerank: Whether to enable reranking
      :param model_name: Grok model name
      :param chunking_type: Method for chunking text

   .. py:method:: get_response(query, user_id)

      Get a Grok-powered response.

      :param query: User query
      :param user_id: User identifier
      :return: Generated response

   .. py:method:: _initialize_models()

      Initialize Grok API client.

      :return: None

   .. py:classmethod:: get_config_class()

      Get the Grok configuration class.

      :return: GrokConfig class

embeddings Module
~~~~~~~~~~~~~~~~~

BaseEmbedding
'''''''''''''

.. py:class:: BaseEmbedding

   Base class for text embedding providers.

   .. py:method:: __init__(api_key=None)

      Initialize the embedding provider.

      :param api_key: API key for the embedding service
      
   .. py:method:: embed(texts, is_query=False)

      Create embeddings for a list of texts.

      :param texts: List of text strings
      :param is_query: Whether these are query embeddings
      :return: List of embedding vectors

CohereEmbedding
'''''''''''''''

.. py:class:: CohereEmbedding

   Cohere embedding provider.

   .. py:method:: __init__(api_key=None)

      Initialize the Cohere embedding provider.

      :param api_key: Cohere API key

   .. py:method:: embed(texts, is_query=False)

      Create embeddings using Cohere.

      :param texts: List of text strings
      :param is_query: Whether these are query embeddings
      :return: List of embedding vectors

MistralEmbedding
''''''''''''''''

.. py:class:: MistralEmbedding

   Mistral embedding provider.

   .. py:method:: __init__(api_key=None)

      Initialize the Mistral embedding provider.

      :param api_key: Mistral API key

   .. py:method:: embed(texts, is_query=False)

      Create embeddings using Mistral.

      :param texts: List of text strings
      :param is_query: Whether these are query embeddings
      :return: List of embedding vectors

OpenAIEmbedding
'''''''''''''''

.. py:class:: OpenAIEmbedding

   OpenAI embedding provider.

   .. py:method:: __init__(api_key=None)

      Initialize the OpenAI embedding provider.

      :param api_key: OpenAI API key

   .. py:method:: embed(texts, is_query=False)

      Create embeddings using OpenAI.

      :param texts: List of text strings
      :param is_query: Whether these are query embeddings
      :return: List of embedding vectors

HuggingFaceEmbedding
''''''''''''''''''''

.. py:class:: HuggingFaceEmbedding

   Hugging Face embedding provider.

   .. py:method:: __init__(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu")

      Initialize the Hugging Face embedding provider.

      :param model_name: Name of the Hugging Face model
      :param device: Device to run the model on (cpu or cuda)

   .. py:method:: embed(texts, is_query=False)

      Create embeddings using Hugging Face models.

      :param texts: List of text strings
      :param is_query: Whether these are query embeddings
      :return: List of embedding vectors

crawler Module
~~~~~~~~~~~~~~

Crawler
'''''''

.. py:class:: Crawler

   Web content crawler.

   .. py:method:: __init__(base_url, domain_name, max_depth=2, max_pages=50, wait_time=1.0, follow_links=True, ignore_query_params=True, client="crawl4ai")

      Initialize the crawler.

      :param base_url: Starting URL
      :param domain_name: Target domain name
      :param max_depth: Maximum crawl depth
      :param max_pages: Maximum pages to crawl
      :param wait_time: Time to wait between requests
      :param follow_links: Whether to follow links
      :param ignore_query_params: Whether to ignore URL query parameters
      :param client: Client library to use for crawling

   .. py:method:: extract_content(link, webpage_only=True, max_depth=None)

      Extract content from a webpage or multiple webpages.

      :param link: URL to crawl
      :param webpage_only: Whether to only extract content from a single page
      :param max_depth: Maximum depth to crawl
      :return: Path to the extracted content

   .. py:method:: _clean_html(html)

      Clean HTML content.

      :param html: HTML content
      :return: Cleaned text content
      
   .. py:method:: _save_extracted_content(text, file_path)

      Save extracted content to a file.

      :param text: Content to save
      :param file_path: Path to save the content to
      :return: Path to the saved content

utils Module
~~~~~~~~~~~~

General Utilities
''''''''''''''''

.. py:function:: create_folder(path)

   Create a folder if it doesn't exist.

   :param path: Path to create
   :return: Path object of the created folder

File Operations
''''''''''''''

.. py:class:: FileLoader

   File loading and processing utility.

   .. py:method:: __init__(file_path, content_path=None, client="docling")

      Initialize the file loader.

      :param file_path: Path to the file to load
      :param content_path: Path to save extracted content
      :param client: Document processing client to use

   .. py:method:: extract_from_file()

      Extract and process content from a file.

      :return: List of document objects
      
   .. py:method:: _get_extension(file_path)

      Get the extension of a file.

      :param file_path: Path to the file
      :return: File extension
      
   .. py:method:: supported_extensions()

      Get list of supported file extensions.

      :return: List of supported extensions

Database Operations
''''''''''''''''''

.. py:class:: DatabaseOps

   Database operations utility.

   .. py:method:: __init__(db_path=None)

      Initialize database operations.

      :param db_path: Path to SQLite database
      
   .. py:method:: _init_db()

      Initialize database tables if they don't exist.
      
      :return: None

   .. py:method:: get_chat_history(user_id=None, last_n=3, full_history=False, last_n_hours=24)

      Retrieve chat history for user.

      :param user_id: User identifier
      :param last_n: Maximum entries to retrieve when not using full_history
      :param full_history: Whether to retrieve full history for all users
      :param last_n_hours: Number of hours to look back when using full_history
      :return: Chat history as formatted string or list of interactions
      
   .. py:method:: append_chat_history(user_id, question, answer, model_used, embedding_model_used)

      Save chat interaction to database.

      :param user_id: User identifier
      :param question: User question
      :param answer: System response
      :param model_used: LLM model used
      :param embedding_model_used: Embedding model used
      :return: None
      
   .. py:method:: append_cost(user_id, model_used, embedding_model_used, input_tokens, output_tokens, cost_per_input_token, cost_per_output_token)

      Track token usage and cost.
      
      :param user_id: User identifier
      :param model_used: LLM model used
      :param embedding_model_used: Embedding model used
      :param input_tokens: Number of input tokens
      :param output_tokens: Number of output tokens
      :param cost_per_input_token: Cost per million input tokens
      :param cost_per_output_token: Cost per million output tokens
      :return: None
      
   .. py:method:: get_monitored_resp()

      Get monitored responses from the last 24 hours.
      
      :return: List of question-answer tuples
      
   .. py:method:: append_bot_sub(user_id, first_name, platform)

      Add a new bot subscriber.

      :param user_id: User identifier
      :param first_name: User's first name
      :param platform: Platform (Telegram, WhatsApp)
      :return: None
      
   .. py:method:: get_bot_sub(user_id=None)

      Get bot subscribers.

      :param user_id: Optional user ID to filter by
      :return: List of subscribers or single subscriber

Email Services
'''''''''''''

.. py:class:: EmailService

   Email notification service.

   .. py:method:: __init__(smtp_server=None, smtp_port=None, sender_email=None, sender_password=None, receiver_email=None)

      Initialize the email service.

      :param smtp_server: SMTP server address
      :param smtp_port: SMTP server port
      :param sender_email: Sender email address
      :param sender_password: Sender email password
      :param receiver_email: Receiver email address
      
   .. py:method:: subscribe(callback)

      Allow other classes to subscribe to email state changes.
      
      :param callback: Callback function to notify
      :return: None
      
   .. py:method:: unsubscibe(callback)

      Remove a subscriber.
      
      :param callback: Callback function to remove
      :return: None
      
   .. py:method:: _notify_subscribers(old_email, new_email)

      Notify subscribers of email changes.
      
      :param old_email: Previous email
      :param new_email: New email
      :return: None
      
   .. py:method:: _format_email_content(unknowns)

      Format the email content with a table of uncertain responses.
      
      :param unknowns: List of (question, answer) tuples
      :return: Formatted HTML content

   .. py:method:: _send_without_attachment(message, unknowns)

      Prepare message without attachments for uncertain responses.
      
      :param message: The email message object
      :param unknowns: List of (question, answer) tuples
      :return: HTML content for the message
      
   .. py:method:: _add_file_attachment(message, file_path, content_type=None)

      Add a file attachment to the email message.
      
      :param message: The email message object
      :param file_path: Path to the file to attach
      :param content_type: Content type of the file
      :return: True if attachment was successful, False otherwise
      
   .. py:method:: send_email_with_attachments(subject, message_body, file_paths=None)

      Send an email with multiple file attachments.
      
      :param subject: The email subject
      :param message_body: The email body text
      :param file_paths: List of file paths to attach
      :return: None
      
   .. py:method:: _send_with_attachment(message, json_data, filename)

      Add JSON data as an attachment to the email.
      
      :param message: The email message object
      :param json_data: JSON data to attach
      :param filename: Filename for the attachment
      :return: The JSON attachment

   .. py:method:: send_email(subject, unknowns=None, json_data=None, filename="conversations.json")

      Send an email with either uncertain responses or JSON data.
      
      :param subject: The email subject line
      :param unknowns: List of uncertain responses
      :param json_data: JSON data to attach
      :param filename: Filename for JSON attachment
      :return: None
      
   .. py:property:: receiver_email

      Get the receiver email address.
      
      :return: Email address
      
   .. py:method:: receiver_email.setter

      Set the receiver email address.
      
      :param value: New email address
      :return: None

Data Processing
''''''''''''''

.. py:function:: count_labels(df, column)

   Count the occurrences of each label in a DataFrame column.
   
   :param df: Pandas DataFrame
   :param column: Column name to count
   :return: Series with label counts

.. py:function:: standardize_length(df, max_length=250)

   Standardize the length of text in a DataFrame column.
   
   :param df: Pandas DataFrame
   :param max_length: Maximum length for text
   :return: DataFrame with standardized text

.. py:function:: truncate_to_n_tokens(text, tokenizer, max_tokens=50)

   Truncate text to a maximum number of tokens.
   
   :param text: Text to truncate
   :param tokenizer: Tokenizer to use
   :param max_tokens: Maximum number of tokens
   :return: Truncated text

Monitoring Services
''''''''''''''''''

.. py:class:: UncertainResponseMonitor

   Monitor and detect uncertain responses.

   .. py:method:: __init__(email_service, every_hours=24, start_service=True)

      Initialize the monitor.

      :param email_service: Email service for notifications
      :param every_hours: Check frequency in hours
      :param start_service: Whether to start monitoring immediately

   .. py:method:: check_for_uncertain_responses()

      Check database for potentially uncertain responses.

      :return: List of uncertain responses
      
   .. py:method:: _start_monitoring()

      Start the monitoring service.
      
      :return: None
      
   .. py:method:: _stop_monitoring()

      Stop the monitoring service.
      
      :return: None
      
   .. py:method:: _schedule_monitoring()

      Schedule periodic monitoring.
      
      :return: None
      
   .. py:method:: _on_exception(e)

      Handle exceptions during monitoring.
      
      :param e: Exception object
      :return: None

.. py:class:: ChatHistoryMonitor

   Monitor chat history and generate reports.

   .. py:method:: __init__(email_service, every_hours=24, start_service=True)

      Initialize the monitor.

      :param email_service: Email service for notifications
      :param every_hours: Check frequency in hours
      :param start_service: Whether to start monitoring immediately

   .. py:method:: generate_report()

      Generate usage report from chat history.

      :return: Report data
      
   .. py:method:: _start_monitoring()

      Start the monitoring service.
      
      :return: None
      
   .. py:method:: _stop_monitoring()

      Stop the monitoring service.
      
      :return: None
      
   .. py:method:: _schedule_monitoring()

      Schedule periodic monitoring.
      
      :return: None

Path Management
''''''''''''''

.. py:data:: BASE_DIR

   Base directory of the project.

.. py:data:: DATA_DIR

   Directory for all data.

.. py:data:: WEB_CONTENT_DIR

   Directory for web content.

.. py:data:: DATASETS_DIR

   Directory for datasets.

.. py:data:: DATABASE_DIR

   Directory for database files.

.. py:data:: INDEXES_DIR

   Directory for vector indexes.

.. py:data:: VOICES_DIR

   Directory for voice recordings.

.. py:data:: MODELS_DIR

   Directory for ML models.

.. py:data:: LOGS_DIR

   Directory for log files.

.. py:data:: TRAIN_FILES_DIR

   Directory for training files.

.. py:data:: CHAT_HIST_DIR

   Directory for chat history.

.. py:data:: FONTS_DIR

   Directory for fonts.

.. py:data:: CLF_PATH

   Path to the classifier model.

config Module
~~~~~~~~~~~~~

.. py:function:: get_api_key(provider)

   Get API key for a specified provider.

   :param provider: Provider name (e.g., "OPENAI", "COHERE")
   :return: API key string
   :raises MissingAPIKeyError: If the API key is not found

MissingAPIKeyError
''''''''''''''''''

.. py:exception:: MissingAPIKeyError

   Exception raised when an API key is missing.

Config
''''''

.. py:class:: Config

   Global configuration container.

   .. py:attribute:: TEMPERATURE
      :type: float

      Temperature setting for language models (0.0-1.0)

   .. py:attribute:: MAX_TOKENS
      :type: int

      Maximum tokens for LLM responses

   .. py:attribute:: CHUNKING_CONFIGS
      :type: dict

      Configuration for different text chunking methods

   .. py:attribute:: AVAILABLE_MODELS
      :type: list

      List of available language models

LLM Provider Configs
''''''''''''''''''''

.. py:class:: OpenAIConfig

   OpenAI-specific configuration.

   .. py:attribute:: AVAILABLE_MODELS
      :type: list

      List of available OpenAI models

.. py:class:: ClaudeConfig

   Claude-specific configuration.

   .. py:attribute:: AVAILABLE_MODELS
      :type: list

      List of available Claude models

.. py:class:: CohereConfig

   Cohere-specific configuration.

   .. py:attribute:: AVAILABLE_MODELS
      :type: list

      List of available Cohere models

.. py:class:: GeminiConfig

   Gemini-specific configuration.

   .. py:attribute:: AVAILABLE_MODELS
      :type: list

      List of available Gemini models

.. py:class:: MistralConfig

   Mistral-specific configuration.

   .. py:attribute:: AVAILABLE_MODELS
      :type: list

      List of available Mistral models

.. py:class:: DeepSeekConfig

   DeepSeek-specific configuration.

   .. py:attribute:: AVAILABLE_MODELS
      :type: list

      List of available DeepSeek models

.. py:class:: GrokConfig

   Grok-specific configuration.

   .. py:attribute:: AVAILABLE_MODELS
      :type: list

      List of available Grok models

web Package
-----------

Chat Web App
~~~~~~~~~~~~

.. py:function:: home()

   Serve the iframe HTML interface.

   :return: HTML response with the chat interface

.. py:function:: chat(request)

   Handle chat requests from the web interface.

   :param request: ChatRequest object containing the query
   :return: JSON response with the chatbot's answer

.. py:function:: transcribe_audio(file)

   Handle audio transcription requests.

   :param file: Uploaded audio file
   :return: JSON response with the transcription

.. py:class:: ChatRequest

   Pydantic model for chat requests.

   .. py:attribute:: query
      :type: str

      The user's query text

Admin Dashboard
~~~~~~~~~~~~~~~

.. py:function:: serve_layout()

   Create the dashboard layout.

   :return: Dash HTML layout components

.. py:function:: register_callbacks(app)

   Register all dashboard callbacks.

   :param app: Dash application instance

.. py:function:: authenticate_user(username, password)

   Authenticate a user against stored credentials.

   :param username: Username to authenticate
   :param password: Password to verify
   :return: True if authentication succeeds, False otherwise

.. py:function:: generate_metrics()

   Generate system metrics for the dashboard.

   :return: Dictionary of metrics (users, queries, token usage, costs)

Bot Implementations
------------------

Telegram Bot
~~~~~~~~~~~

.. py:class:: TelegramBot

   Telegram bot implementation.
   
   .. py:attribute:: EMAIL_REGEX
      :type: str
      
      Regular expression for validating email addresses
      
   .. py:attribute:: ADMINS
      :type: list
      
      List of admin user IDs
      
   .. py:method:: __init__(link)
   
      Initialize the Telegram bot.
      
      :param link: Website link to initialize the RAG system with
      
   .. py:method:: extract_domain_name(link)
   
      Extract domain name from a URL.
      
      :param link: URL to extract domain from
      :return: Domain name
      
   .. py:method:: fetch_content(link, domain_name, max_depth=None, file_path=None, webpage_only=True)
   
      Fetch content from a URL or file.
      
      :param link: URL to fetch content from
      :param domain_name: Domain name
      :param max_depth: Maximum crawl depth
      :param file_path: Path to file
      :param webpage_only: Whether to only fetch a single page
      :return: Path to fetched content
      
   .. py:method:: _init_rag_system()
   
      Initialize the RAG system with content from the website.
      
      :return: None
      
   .. py:method:: _setup_handlers()
   
      Set up Telegram command and message handlers.
      
      :return: None
      
   .. py:method:: start(update, context)
   
      Handle the /start command.
      
      :param update: Update from Telegram
      :param context: CallbackContext for the bot
      :return: Next conversation state
      
   .. py:method:: transcribe(audio_buffer)
   
      Transcribe voice messages to text.
      
      :param audio_buffer: Audio buffer containing voice message
      :return: Transcribed text
      
   .. py:method:: add_content(update, context)
   
      Add new content to the RAG system.
      
      :param update: Update from Telegram
      :param context: CallbackContext for the bot
      :return: None
      
   .. py:method:: add_admin(update, context)
   
      Add a new admin user.
      
      :param update: Update from Telegram
      :param context: CallbackContext for the bot
      :return: None
      
   .. py:method:: remove_admin(update, context)
   
      Remove an admin user.
      
      :param update: Update from Telegram
      :param context: CallbackContext for the bot
      :return: None
      
   .. py:method:: get_admins(update, context)
   
      List current admin users.
      
      :param update: Update from Telegram
      :param context: CallbackContext for the bot
      :return: None
      
   .. py:method:: _is_admin(user_id)
   
      Check if a user is an admin.
      
      :param user_id: User ID to check
      :return: True if user is admin, False otherwise
      
   .. py:method:: _user_exists(id)
   
      Check if a user exists in the database.
      
      :param id: User ID to check
      :return: True if user exists, False otherwise
      
   .. py:method:: _run_rag_query(question, user_id)
   
      Run a RAG query in a separate thread.
      
      :param question: User question
      :param user_id: User ID
      :return: RAG response
      
   .. py:method:: _extract_question(msg, context)
   
      Extract question from text or voice message.
      
      :param msg: Message from Telegram
      :param context: CallbackContext for the bot
      :return: Extracted question text
      
   .. py:method:: handle_question(update, context)
   
      Process questions from users.
      
      :param update: Update from Telegram
      :param context: CallbackContext for the bot
      :return: Next conversation state
      
   .. py:method:: cancel_conversation(update, context)
   
      Cancel the current conversation.
      
      :param update: Update from Telegram
      :param context: CallbackContext for the bot
      :return: End of conversation
      
   .. py:method:: set_email(update, context)
   
      Set email for receiving notifications.
      
      :param update: Update from Telegram
      :param context: CallbackContext for the bot
      :return: None
      
   .. py:method:: _is_valid_email(new_email)
   
      Check if an email address is valid.
      
      :param new_email: Email address to check
      :return: True if valid, False otherwise
      
   .. py:method:: broadcast(update, context)
   
      Broadcast a message to all bot subscribers.
      
      :param update: Update from Telegram
      :param context: CallbackContext for the bot
      :return: None
      
   .. py:method:: run_async()
   
      Run the bot asynchronously.
      
      :return: None
      
   .. py:method:: _handle_exit(signum, frame)
   
      Handle exit signals gracefully.
      
      :param signum: Signal number
      :param frame: Current stack frame
      :return: None
      
   .. py:method:: run()
   
      Run the bot using asyncio.run.
      
      :return: None

WhatsApp Bot
~~~~~~~~~~~

.. py:class:: TwilioClient

   WhatsApp bot implementation using Twilio.
   
   .. py:attribute:: TWILIO_SID
      :type: str
      
      Twilio account SID
      
   .. py:attribute:: TWILIO_PHONE_NUMBER
      :type: str
      
      Twilio phone number
      
   .. py:attribute:: TWILIO_AUTH_TOKEN
      :type: str
      
      Twilio authentication token
      
   .. py:method:: __init__()
   
      Initialize the Twilio client for WhatsApp messaging.
      
   .. py:method:: send_whatsapp_message(to_number, message)
   
      Send a WhatsApp message.
      
      :param to_number: Recipient's phone number
      :param message: Message to send
      :return: Success status
      
   .. py:method:: webhook(request)
   
      Handle incoming webhook requests from Twilio.
      
      :param request: FastAPI request object
      :return: TwiML response for Twilio

.. py:function:: app.post("/sms")(twilio_bot.webhook)
   
   Route for Twilio SMS webhook.
   
   :return: Response from webhook handler