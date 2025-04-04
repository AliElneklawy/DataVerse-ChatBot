# DataVerse ChatBot

![image](https://github.com/user-attachments/assets/357e1df2-7f28-44b3-b242-6c39e9784d32)

**DataVerse ChatBot** is a powerful Python-based application that enables real-time, AI-driven chat interactions by extracting and processing data from virtually any source—web pages, nearly all file formats, and more. Combining advanced web crawling, multi-format data extraction, and Retrieval-Augmented Generation (RAG) techniques, it integrates with leading Large Language Models (LLMs) to deliver context-aware responses. The system includes a sophisticated chat history analysis agent that provides actionable insights on user engagement patterns and response quality. Deployable as WhatsApp and Telegram bots or embedded via an iframe, DataVerse ChatBot also supports voice messages, making it a versatile tool for conversational AI.

## Features

### Data Extraction and Processing
- **Web Crawling**: Supports 2 libraries for web crawling; [`crawl4ai`](https://docs.crawl4ai.com/) and [`scrapegraphai`](https://docs.scrapegraphai.com/introduction) to efficiently gather content from specified web sources, with customizable parameters (e.g., crawl depth, preferred client).
- **Multi-Format Data Extraction**: Supports 2 libraries for data extraction; [`langchain`](https://python.langchain.com/docs/introduction/) and [`docling`](https://python.langchain.com/docs/integrations/document_loaders/docling/) to extract data from nearly all file formats (e.g., PDFs, text files, docx, csv, xlsx, etc...), broadening its knowledge base beyond web content.
- **Content Storage**: Saves extracted data in `data/web_content/` as text files in a clean markdown format (which LLMs love) for indexing and retrieval.

### Chat History Agent

![image](https://github.com/user-attachments/assets/a2a348c3-8755-4260-a1cd-2e858f604f72)


An intelligent agent powered by LangChain that analyzes conversation data to extract insights about common user questions, peak usage times, response quality, user engagement patterns, ...etc.

### Monitoring and Uncertainty Detection
- **Response Monitoring**: Implements a monitoring service to track responses and detect the questions that the LLM couldn't answer.
- **Uncertain Response Classification**: Uses a trained classifier to detect uncertain responses from LLM.
- **Email Notifications**: Sends alerts via email when uncertain responses are detected.
- **Chat History Monitoring**: Periodically emails the chat history to a configured email address.
- The inference and the monitoring services all run in a separate thread. This ensures that the main program doesn't freeze.

### Dataset Creation and Model Training
- **Dataset Creation**: Uses the `make_dataset.py` script to create standardized datasets from RAG responses. It supports:
  - Data cleaning and tokenization using [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
  - Saving structured responses in CSV format for further processing.
  - Shuffling and saving datasets for training purposes.
- **Content Storage**: Saves the dataset in `data/datasets/`.
- **Training Scripts**: Utilizes `train_clf.py` to train classification models with support for:
  - [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) and [XGBoost](https://xgboost.readthedocs.io/en/stable/) classifiers.
  - Hyperparameter tuning using [`RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html).
  - Embedding generation with `sentence-transformers/all-MiniLM-L6-v2`.
- **Evaluation Metrics**:
  - Accuracy, Precision, and Recall.
  - [ROC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html) and [Precision-Recall](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html) Curve plotting.
  - The model achieved 92.7% accuracy on the test set.
- **Model Persistence**: Saves trained models as `.pkl` files and logs metadata (e.g., hyperparameters, evaluation results, versions of the libraries, ...).


### Chat Interfaces
- **WhatsApp Bot**: Deployable as a WhatsApp bot using [Twilio](https://github.com/twilio/twilio-python) for seamless, mobile-friendly conversations.
- **Telegram Bot**: Available as a [Telegram bot](https://docs.python-telegram-bot.org/en/stable/), integrating with Telegram’s messaging ecosystem.
- **Iframe Embedding**: Embeddable via an iframe for easy integration into websites or applications. Built with [FastAPI](https://fastapi.tiangolo.com/) for the backend; HTML, CSS, and JavaScript for the frontend.

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
- **Vector Store Integration**: Uses [FAISS](https://github.com/facebookresearch/faiss?tab=readme-ov-file) (in `data/indexes/`) for fast, efficient document retrieval.
- **Context-Aware Responses**: Combines extracted data with LLM capabilities for accurate replies.

### Embedding Generation
- **Base Embedding System**: Generates embeddings for data and queries via a reusable interface.
- **Multiple Embedding Models**: Supports embedding APIs from LLMs (Cohere, Mistral, OpenAI) or standalone models (HuggingFace).
- **Content Indexing**: Stores embeddings in FAISS indexes (`index.faiss`, `index.pkl`) for quick retrieval.

### Chat Functionality
- **Chat History Persistence**: Saves conversations in a [SQLite](https://docs.python.org/3/library/sqlite3.html) database (`chat_history.db`).
- **Context Retention**: Maintains conversational context using history and retrieved data.
- **Query Processing**: Processes user inputs through embeddings and RAG for response generation.

### Modular Design
- **Package Structure**: Organized into reusable modules:
  - `crawler.py`: Web crawling logic.
  - `embeddings/`: Embedding generation.
  - `rag/`: RAG implementations.
  - `utils/`: Helper functions.
- **Extensibility**: Easy to add new LLMs, embedding models, or features.


### Admin Dashboard
The admin dashboard provides a centralized interface for managing the RAG system, offering tools to monitor usage, manage content, and update account settings.


![image](https://github.com/user-attachments/assets/ecd4917a-967d-4b96-971e-bb72c03509a9)




- **Admin Login**: Secure access to the admin panel with a username and password.
- **System Overview Dashboard**: Displays key metrics such as total users, active conversations, token usage, and costs over the last 24 hours. It also includes a table of recent conversations and a pie chart showing model usage distribution.
- **Content Management** : Allows the admin to upload files or crawl websites to expand the RAG system's knowledge base.
- **Account Settings** : Admins can update their username and password.
  
### Setup and Installation
- **Automated Setup**: Single-command installation via `install.bat` (Windows) for dependencies and configuration.
- **Dependency Management**: Installs packages from `pyproject.toml` (built by [uv](https://github.com/astral-sh/uv)).
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
   git clone https://github.com/AliElneklawy/DataVerse-ChatBot.git
   cd DataVerse-ChatBot
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

## Project Structure

```
├── DataVerse-Chatbot
│   ├── __init__.py 
│   ├── data 
│   │   ├── chat_history 
│   │   │   ├── ....
│   │   ├── database 
│   │   │   ├── ....
│   │   ├── datasets 
│   │   │   ├── ....
│   │   ├── indexes 
│   │   │   ├── ....
│   │   ├── logs 
│   │   │   ├── ....
│   │   ├── models 
│   │   │   ├── clf.pkl 
│   │   │   ├── metadata.json 
│   │   ├── training_files 
│   │   │   ├── ....
│   │   ├── web_content 
│   │   │   ├── ....
│   ├── src 
│   │   ├── admin_dashboard_launcher.py 
│   │   ├── main.py 
│   │   ├── tg_bot.py 
│   │   ├── whatsapp_bot.py 
│   │   ├── __init__.py 
│   │   ├── chatbot 
│   │   │   ├── config.py 
│   │   │   ├── crawler.py 
│   │   │   ├── voice_mode.py 
│   │   │   ├── __init__.py 
│   │   │   ├── embeddings 
│   │   │   │   ├── base_embedding.py 
│   │   │   │   ├── __init__.py 
│   │   │   ├── rag 
│   │   │   │   ├── base_rag.py 
│   │   │   │   ├── claude_rag.py 
│   │   │   │   ├── cohere_rag.py 
│   │   │   │   ├── deepseek_rag.py 
│   │   │   │   ├── gemini_rag.py 
│   │   │   │   ├── grok_rag.py 
│   │   │   │   ├── mistral_rag.py 
│   │   │   │   ├── openai_rag.py 
│   │   │   │   ├── __init__.py 
│   │   │   ├── utils 
│   │   │   │   ├── admin_utils.py 
│   │   │   │   ├── crawler_progress.py 
│   │   │   │   ├── file_loader.py 
│   │   │   │   ├── inference.py 
│   │   │   │   ├── make_dataset.py 
│   │   │   │   ├── monitor_service.py 
│   │   │   │   ├── paths.py 
│   │   │   │   ├── train_clf.py 
│   │   │   │   ├── utils.py 
│   │   │   │   ├── __init__.py 
│   │   ├── web 
│   │   │   ├── admin_dashboard.py 
│   │   │   ├── chat_web_app.py 
│   │   │   ├── chat_web_template.py 
│   │   │   ├── how to run.txt 
│   │   │   ├── __init__.py 
│   │   │   ├── static 
│   │   │   │   ├── __init__.py 
│   │   │   │   ├── css 
│   │   │   │   │   ├── admin.css 
│   │   │   │   │   ├── dark_mode.css 
│   │   │   │   │   ├── __init__.py 
│   │   │   │   ├── js 
│   │   │   │   │   ├── admin.js 
│   │   │   │   │   ├── __init__.py 
│   │   │   ├── templates 
│   │   │   │   ├── __init__.py 
│   │   │   │   ├── admin 
│   │   │   │   │   ├── account.html 
│   │   │   │   │   ├── base.html 
│   │   │   │   │   ├── content.html 
│   │   │   │   │   ├── dashboard.html 
│   │   │   │   │   ├── history.html 
│   │   │   │   │   ├── login.html 
│   │   │   │   │   ├── models.html 
│   │   │   │   │   ├── system.html 
│   │   │   │   │   ├── users.html 
│   │   │   │   │   ├── user_detail.html 
│   │   │   │   │   ├── view_content.html 
│   │   │   │   │   ├── __init__.py 
│   ├── tests 
│   │   ├── locustfile.py 
```
