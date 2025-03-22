import cohere
import openai
import logging
from typing import List
from mistralai import Mistral
from abc import ABC, abstractmethod
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class BaseEmbedding(ABC):
    @abstractmethod
    def embed(self, texts: list) -> list:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of texts to embed.
        
        Returns:
            List[List[float]]: List of embedding vectors.
        """
        pass

    def embed_documents(self, text: str):
        """This method is used as a wrapper for langchain's semantic chunker."""
        return self.embed(text, False)


class CohereEmbedding(BaseEmbedding):
    def __init__(self, api_key, 
                 embedding_model: str = "embed-multilingual-v3.0"):
        self.client = cohere.Client(api_key)
        self.embedding_model = embedding_model

        logger.info(f"Using cohere's {embedding_model} embedding model.")
    
    def embed(self, texts: list | str, is_query: bool) -> list:
        input_type = "search_query" if is_query else "search_document"
        return self.client.embed(
            texts=texts,
            model=self.embedding_model,
            input_type=input_type,
        ).embeddings
    
class MistralEmbedding(BaseEmbedding):
    def __init__(self, api_key,
                 embedding_model: str = "mistral-embed"):
        self.client = Mistral(api_key=api_key)
        self.embedding_model = embedding_model

        logger.info(f"Using Mistral's {embedding_model} embedding model.")

    def embed(self, texts, _):
        if isinstance(texts, str):
                texts = [texts]

        response = self.client.embeddings.create(
            inputs=texts,
            model=self.embedding_model
        )

        embeddings = [list(item.embedding) for item in response.data]
        logger.info(f"Generated {len(embeddings)} embeddings with Mistral.")

        return embeddings
    
class OpenAIEmbedding(BaseEmbedding):
    def __init__(self, api_key: str,
                 embedding_model: str = "text-embedding-3-small"):
        self.api_key = api_key
        openai.api_key = api_key
        self.embedding_model = embedding_model

        logger.info(f"Using OpenAI's {embedding_model} embedding model.")
    
    def embed(self, texts: List[str], _) -> List[List[float]]:
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            response = openai.embeddings.create(
                input=texts,
                model=self.embedding_model,
            )
            
            embeddings = [list(item.embedding) for item in response.data]
            logger.debug(f"Generated {len(embeddings)} embeddings with OpenAI.")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings with OpenAI: {str(e)}")
            raise

class HuggingFaceEmbedding(BaseEmbedding):
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
                 device: str = "cpu",
                 normalize_embeddings: bool = False):
        self.embedding_model = embedding_model
        self.embedding_client = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': normalize_embeddings},
        )

    def embed(self, texts: list, is_query: bool):
        if is_query:
            return [self.embedding_client.embed_query(texts[0])]
        return self.embedding_client.embed_documents(texts)
