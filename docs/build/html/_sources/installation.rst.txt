Installation
============

Requirements
------------

* Python 3.11 or higher
* Git (for cloning the repository)
* Dependencies listed in `pyproject.toml`

Quick Installation
------------------

On Windows
~~~~~~~~~~

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/AliElneklawy/DataVerse-ChatBot.git
      cd DataVerse-ChatBot

2. Run the installation script:

   .. code-block:: bash

      install.bat

   This script will:
   
   * Create a virtual environment (if needed)
   * Install all dependencies from `pyproject.toml`
   * Set up environment configuration

Manual Installation
-------------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/AliElneklawy/DataVerse-ChatBot.git
      cd DataVerse-ChatBot

2. Create a virtual environment (recommended):

   .. code-block:: bash

      python -m venv venv
      # On Windows
      venv\Scripts\activate
      # On macOS/Linux
      source venv/bin/activate

3. Install dependencies using uv:

   .. code-block:: bash

      pip install uv
      uv pip install -e .

   Alternatively, use pip directly:

   .. code-block:: bash

      pip install -e .

4. Create and configure the `.env` file:

   .. code-block:: bash

      cp .env.example .env
      # Edit .env with your API keys and configuration

API Key Configuration
---------------------

The system requires API keys from various LLM providers depending on which ones you plan to use.
Edit your `.env` file to include the appropriate keys:

.. code-block:: ini

   # OpenAI
   OPENAI_API=your_openai_api_key

   # Anthropic (Claude)
   CLAUDE_API=your_claude_api_key

   # Cohere
   COHERE_API=your_cohere_api_key

   # Google (Gemini)
   GOOGLE_API=your_google_api_key

   # Mistral
   MISTRAL_API=your_mistral_api_key

   # DeepSeek
   DEEPSEEK_API=your_deepseek_api_key

   # Grok (X.AI)
   GROK_API=your_grok_api_key

   # Email configuration for monitoring
   GMAIL_APP_PASSWORD=your_gmail_app_password

Directory Structure
-------------------

After installation, the following directory structure will be created:

.. code-block:: text

   DataVerse-ChatBot/
   ├── assets/              # Fonts and images
   ├── data/                # Data storage
   │   ├── database/        # SQLite databases
   │   ├── datasets/        # Training datasets
   │   ├── indexes/         # FAISS vector indexes
   │   ├── models/          # Trained models
   │   ├── training_files/  # Files for training
   │   ├── voices/          # Voice recordings
   │   └── web_content/     # Extracted content
   ├── docs/                # Documentation
   ├── src/                 # Source code
   │   ├── chatbot/         # Core chatbot functionality
   │   │   ├── embeddings/  # Embedding implementations
   │   │   ├── rag/         # RAG implementations
   │   │   └── utils/       # Utility functions
   │   ├── web/             # Web interfaces
   │   └── *.py             # Bot implementations
   └── tests/               # Test files