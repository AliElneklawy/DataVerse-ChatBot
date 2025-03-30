Chatbot Module
==============

This module contains the core chatbot functionality.

RAG (Retrieval-Augmented Generation)
------------------------------------

The RAG module provides retrieval-augmented generation capabilities.

Base RAG
~~~~~~~~

The BaseRAG class is the foundation for all RAG implementations. It provides common functionality
for document retrieval, reranking, and context generation.

Key features:
- Document vectorization and retrieval
- Context preparation and cleaning
- Query reformulation
- Response monitoring

RAG Implementations
~~~~~~~~~~~~~~~~~~

Different implementations are available for various LLM providers:

- **ClaudeRAG**: Uses Anthropic's Claude models
- **OpenAIRAG**: Integrates with OpenAI's models
- **CohereRAG**: Leverages Cohere's language models
- **GeminiRAG**: Connects to Google's Gemini AI
- **MistralRAG**: Works with Mistral AI models
- **DeepseekRAG**: Utilizes Deepseek's language models
- **GrokRAG**: Integrates with Grok's AI capabilities

Embeddings
----------

The embeddings module provides text embedding capabilities for vector search.

Available embedding models:
- **CohereEmbedding**: Uses Cohere's embedding models
- **OpenAIEmbedding**: Leverages OpenAI's embedding capabilities
- **MistralEmbedding**: Works with Mistral's embedding models
- **HuggingFaceEmbedding**: Uses open-source Hugging Face models

Crawler
-------

The Crawler component handles web content extraction with features:
- Configurable depth and breadth for crawling
- URL filtering
- Content extraction and cleaning
- Rate limiting and politeness controls

Configuration
-------------

The config module provides centralized configuration management:
- API key management
- Model selection and parameters
- Default system prompts
- Chunking strategies

Utilities
---------

Various utility components provide supporting functionality:

File Loader
~~~~~~~~~~
Handles loading and processing various file formats including:
- PDF documents
- Microsoft Office files
- Plain text
- HTML content

Database Operations
~~~~~~~~~~~~~~~~~~
Provides database interaction for:
- Storing chat history
- Managing document indices
- Tracking usage metrics

Monitoring
~~~~~~~~~~
Service monitoring capabilities include:
- Uncertain response detection
- Chat history analysis
- Usage statistics
- Alerting mechanisms