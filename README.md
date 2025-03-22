# DataVerse ChatBot

![image](https://github.com/user-attachments/assets/357e1df2-7f28-44b3-b242-6c39e9784d32)

**DataVerse ChatBot** is a powerful Python-based application that enables real-time, AI-driven chat interactions by extracting and processing data from virtually any source—web pages, nearly all file formats, and more. Combining advanced web crawling, multi-format data extraction, and Retrieval-Augmented Generation (RAG) techniques, it integrates with leading Large Language Models (LLMs) to deliver context-aware responses. Deployable as WhatsApp and Telegram bots or embedded via an iframe, DataVerse ChatBot also supports voice messages, making it a versatile tool for conversational AI.

## Features

### Data Extraction and Processing
- **Web Crawling**: Supports 2 libraries for web crawling; [`crawl4ai`](https://docs.crawl4ai.com/) and [`scrapegraphai`](https://docs.scrapegraphai.com/introduction) to efficiently gather content from specified web sources, with customizable parameters (e.g., crawl depth, preferred client).
- **Multi-Format Data Extraction**: Supports 2 libraries for data extraction; [`langchain`](https://python.langchain.com/docs/introduction/) and [`docling`](https://python.langchain.com/docs/integrations/document_loaders/docling/) to extract data from nearly all file formats (e.g., PDFs, text files, docx, csv, xlsx, etc...), broadening its knowledge base beyond web content.
- **Content Storage**: Saves extracted data in `data/web_content/` as text files in a clean markdown format (which LLMs love) for indexing and retrieval.
  

### Monitoring and Uncertainty Detection
- **Response Monitoring**: Implements a monitoring service to track responses and detect the questions that the LLM couldn't answer.
- **Uncertain Response Classification**: Uses a trained classifier to detect uncertain responses from LLM.
- **Email Notifications**: Sends alerts via email when uncertain responses are detected.
- **Chat History Monitoring**: Periodically emails the chat history to a configured email address.

### Dataset Creation and Model Training
- **Dataset Creation**: Uses the `make_dataset.py` script to create standardized datasets from RAG responses. It supports:
  - Data cleaning and tokenization using `sentence-transformers/all-MiniLM-L6-v2`.
  - Saving structured responses in CSV format for further processing.
  - Shuffling and saving datasets for training purposes.
- **Content Storage**: Saves the dataset in `data/datasets/`.
- **Training Scripts**: Utilizes `train_clf.py` to train classification models with support for:
  - Random Forest and XGBoost classifiers.
  - Hyperparameter tuning using `RandomizedSearchCV`.
  - Embedding generation with `sentence-transformers/all-MiniLM-L6-v2`.
- **Evaluation Metrics**:
  - Accuracy, Precision, and Recall.
  - ROC and Precision-Recall Curve plotting.
  - The model achieved 92.7% accuracy on the test set.
- **Model Persistence**: Saves trained models as `.pkl` files and logs metadata (e.g., hyperparameters, evaluation results).


### Chat Interfaces
- **WhatsApp Bot**: Deployable as a WhatsApp bot for seamless, mobile-friendly conversations.
- **Telegram Bot**: Available as a Telegram bot, integrating with Telegram’s messaging ecosystem.
- **Iframe Embedding**: Embeddable via an iframe for easy integration into websites or applications.

### Large Language Model (LLM) Integration
- **Multiple LLM Support**: Integrates with various LLMs, including:
  - OpenAI
  - Claude
  - Cohere
  - DeepSeek
  - Gemini
  - Grok
  - Mistral
- **Flexible LLM Selection**: Configurable via settings to switch between LLMs based on user preference or use case.

### Retrieval-Augmented Generation (RAG)
- **Base RAG Framework**: Provides a consistent RAG interface for retrieval and generation.
- **LLM-Specific RAG**: Custom implementations for each supported LLM, optimizing performance.
- **Vector Store Integration**: Uses FAISS (in `data/indexes/`) for fast, efficient document retrieval.
- **Context-Aware Responses**: Combines extracted data with LLM capabilities for accurate replies.

### Embedding Generation
- **Base Embedding System**: Generates embeddings for data and queries via a reusable interface.
- **Multiple Embedding Models**: Supports embedding APIs from LLMs (Cohere, Mistral, OpenAI) or standalone models (HuggingFace).
- **Content Indexing**: Stores embeddings in FAISS indexes (`index.faiss`, `index.pkl`) for quick retrieval.

### Chat Functionality
- **Chat History Persistence**: Saves conversations in a SQLite database (`chat_history.db`).
- **Context Retention**: Maintains conversational context using history and retrieved data.
- **Query Processing**: Processes user inputs through embeddings and RAG for response generation.

### Modular Design
- **Package Structure**: Organized into reusable modules:
  - `crawler.py`: Web crawling logic.
  - `embeddings/`: Embedding generation.
  - `rag/`: RAG implementations.
  - `utils/`: Helper functions.
- **Extensibility**: Easy to add new LLMs, embedding models, or features.

### Setup and Installation
- **Automated Setup**: Single-command installation via `install.bat` (Windows) for dependencies and configuration.
- **Dependency Management**: Installs packages from `pyproject.toml`.
- **Environment Configuration**: Automated configuration.

### Data Management
- **Persistent Storage**: Organizes data into:
  - `data/web_content/`: Extracted content.
  - `data/indexes/`: Vectorized indexes.
  - `data/database/`: Chat history database (also monitors the induced costs for using the LLMs across all chat interfaces).
- **Efficient Retrieval**: Uses FAISS for scalable similarity searches.
- **Database Support**: Lightweight SQLite for chat history.


## Installation

### Windows
1. Clone the repository:
   ```bash
   git clone https://github.com/AliElneklawy/chat-with-your-data.git
   cd chat-with-your-data
2. Run the installation script:
   ```bash
   install.bat
  This installs dependencies and configures the environment.

## Usage
1. Configure your .env file with API keys.
2. Run the application (you can choose to run the telegram bot, the whatsapp bot, the ifram or just the `main.py` file):
     ```bash
     python main.py
3. You can create your own dataset using `make_dataset.py` script.
4. You can train your own classifier using `train_clf.py` script. 
