# WebRAG: Real-Time Chat with AI-Augmented Web Data

**WebRAG** is a sophisticated Python-based application designed to combine web crawling capabilities with Retrieval-Augmented Generation (RAG) techniques, enabling real-time chat interactions powered by advanced AI models. This project leverages an external `crawl4ai` library for crawling and integrates various RAG implementations to provide accurate, context-aware responses.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Windows](#windows)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Features

- **Web Crawling**: Efficiently gathers content from specified web sources using the `crawl4ai` library.
- **LLM Integration**: Supports multiple LLMs (e.g., OpenAI, Claude, Cohere) for enhanced response generation.
- **Modular Design**: Organized into reusable packages for embeddings, RAG implementations, and utilities.
- **Automated Setup**: Installs dependencies and configures the environment with a single command.

## Requirements

- **Python**: Version 3.11 or higher.
- **Environment Variables**: A `.env` file with necessary API keys (see `.env.example` for template).
- **Internet Connection**: Required for dependency installation and web crawling.

## Installation

Follow the instructions below to set up the project on your system. The installation process automatically installs all dependencies and runs required setup commands from the `crawl4ai` library.

### Windows

1. **Clone the Repository**:
   
   Open a Command Prompt (CMD) and execute:
   ```cmd
   git clone https://github.com/AliElneklawy/WebRAG.git
   cd WebRAG
   
3. **Install and Configure**:
   
   Run the provided batch script to install dependencies and set up the project:
   ```cmd
   install.bat
   
  This command Installs all dependencies listed in `requirements.txt`.
  Executes `crawl4ai-setup` and `crawl4ai-doctor` to configure the environment.

## Usage

After installation, run the `main.py` file. This launches the main application entry point. API configurations should be provided in a `.env` file in the project root. Refer to `.env` for the required format.

## Project Structure

The repository is organized as follows:

```plaintext
├── WebRAG
│   ├── .env 
│   ├── install.bat  # Windows installation script
│   ├── requirements.txt 
│   ├── setup.py 
│   ├── setup_project.py 
│   ├── data 
│   │   ├── database  # Chat history database
│   │   │   ├── chat_history.db 
│   │   ├── indexes # Vector store indexes
│   │   │   ├── index_623e7ea40a8e68c.faiss 
│   │   │   │   ├── index.faiss 
│   │   │   │   ├── index.pkl 
│   │   ├── web_content  # Crawled web content
│   │   │   ├── pydantic.txt 
│   ├── src # Source code
│   │   ├── main.py 
│   │   ├── chatbot 
│   │   │   ├── config.py 
│   │   │   ├── crawler.py  # Web crawling logic
│   │   │   ├── __init__.py 
│   │   │   ├── embeddings  # Embedding-related modules
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
│   │   │   │   ├── utils.py 
│   │   │   │   ├── __init__.py 
│   ├── tests 
