import os
import logging
from dotenv import load_dotenv
from .embeddings.base_embedding import CohereEmbedding
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

load_dotenv()
logger = logging.getLogger(__name__)

def get_api_key(service: str):
    api_key = os.getenv(f"{service}_API")
    
    if api_key is None:
        msg = f"API key for {service} not found in environment variables."
        logger.error(msg)
        raise MissingAPIKeyError(msg)
    else:
        logger.debug(f"Successfully retrieved API key for {service}.")
        
    return api_key

class MissingAPIKeyError(Exception):
    """Custom exception for missing API keys."""
    pass

class Config:
    CHUNK_SIZE: int = 1500
    TEMPERATURE: float = 0.0
    CHUNK_OVERLAP: int = 150
    MAX_OUT_TOKENS: int = 1500

    CHUNKING_CONFIGS = {
        "recursive": {
            "message": f"Performing recursive chunking with chunk size {CHUNK_SIZE} "
                      f"and chunk overlap {CHUNK_OVERLAP}",
            "splitter": RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
        },
        "semantic": {
            "message": "Performing semantic chunking using Cohere's embedding model.",
            "splitter": SemanticChunker(CohereEmbedding(get_api_key("COHERE")))
        },
        "basic": {
            "message": "Performing basic chunking.",
            "splitter": CharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len
            )
        }
    }

# AVAILABLE_MODELS = {"model_name": [input_price, output_price]}
class GrokConfig:
     AVAILABLE_MODELS = {
         "grok-2": [2.00, 10.00],
         "grok-2-latest": [2.00, 10.00],
         "grok-2-vision": [2.00, 10.00],
         "grok-2-vision-latest": [2.00, 10.00],
         "grok-vision-beta": [5.00, 15.00],
         "grok-beta": [5.00, 15.00],
    }
     BASE_URL = "https://api.x.ai/v1"

class CohereConfig:
     AVAILABLE_MODELS = {
         "command-r-08-2024": [0.30, 1.20],
         "command-r-plus-04-2024": [3.00, 15.00],
         "command-r-plus-08-2024": [2.50, 10.00],
         "command-r7b-12-2024": [0.0375,0.15],
         "command-r-03-2024": [0.50, 1.50],
         "command-a-03-2025": [2.50, 10.00]
    }

class ClaudeConfig:
    AVAILABLE_MODELS = {
        "claude-3-5-haiku-20241022":[0.80, 4.0],
        "claude-3-haiku-20240307":[0.25, 1.25],
        "claude-3-7-sonnet-20250219":[3.0, 15.0],
        "claude-3-5-sonnet-20241022":[3.0, 15.0],
        "claude-3-5-sonnet-20240620":[3.0, 15.0],
        "claude-3-opus-20240229":[15.0, 75.0],
       # "claude-3-sonnet-20240229",
    }

class GeminiConfig:
     AVAILABLE_MODELS = {
         "gemini-1.5-flash": [0.15, 0.60],
         "gemini-1.5-flash-8b": [0.075, 0.30],
         "gemini-1.5-pro": [2.50, 10.00],
         "gemini-2.0-flash-lite-preview-02-05": [0.75, 0.30],
         "gemini-2.0-flash": [0.70, 0.40],
     }

class MistralConfig:
     AVAILABLE_MODELS = {
         "mistral-small-latest": [0.1, 0.3],
         "mistral-saba-2502": [0.2, 0.6],
         "mistral-large-latest": [2.0, 6.0],
         "pixtral-large-latest": [2.0, 6.0],
         "codestral-latest": [0.3, 0.9],
         "ministral-3b-latest": [0.04, 0.04],
         "ministral-8b-latest": [0.1, 0.1],
         #"pixtral-12b-2409": [in, out],
     }

class OpenAIConfig:
    AVAILABLE_MODELS = { 
        "gpt-3.5-turbo-0125": [0.5, 1.5],
        "gpt-3.5-turbo-1106": [1, 2],
        "gpt-3.5-turbo-0613": [1.5, 2],
        "gpt-4-0314": [30, 60],
        "gpt-4-0613": [30, 60],
        "gpt-4-1106-preview": [10, 30],
        "gpt-4-0125-preview": [10, 30],
        "gpt-4-turbo-2024-04-09": [10, 30],
        "o3-mini": [1.10, 4.40],
        "o1-mini": [1.10, 4.40],
        "o1": [15, 60],
        "gpt-4o-mini": [0.15, 0.60],
        "chatgpt-4o-latest": [5, 15],
        "gpt-4o": [2.5, 10],
        "gpt-4.5-preview": [75, 150],
    }

class DeepSeekConfig:
    AVAILABLE_MODELS = {
        "deepseek-chat": [0.27, 1.10],
        "deepseek-reasoner": [0.55, 2.19],
    }
    BASE_URL = "https://api.deepseek.com"
