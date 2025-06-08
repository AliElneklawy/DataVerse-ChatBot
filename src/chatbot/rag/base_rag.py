import re
import shutil
import cohere
import sqlite3
import logging
import hashlib
import threading
import html2text
import logging.handlers
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod
from ..config import get_api_key, RAGConfig
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from ..embeddings.base_embedding import CohereEmbedding
from ..utils.paths import INDEXES_DIR, DATABASE_DIR, LOGS_DIR
from tenacity import retry, stop_after_attempt, wait_exponential
from ..utils.utils import create_folder, DatabaseOps, EmailService
from ..utils.monitor_service import UncertainResponseMonitor, ChatHistoryMonitor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:%(lineno)d - %(message)s",
    handlers=[
        logging.handlers.RotatingFileHandler(
            create_folder(LOGS_DIR) / "logs.log", maxBytes=1024**3, backupCount=10
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class BaseRAG(ABC):
    def __init__(
        self,
        content_path: Path,
        index_path: Optional[str] = None,
        rerank: bool = True,
        model_name: Optional[str] = None,
        chunking_type: str = "recursive",
        init_hist_monitor: bool = False,
        init_resp_monitor: bool = False,
        hist_monitor_freq: int = 24,
        resp_monitor_freq: int = 24,
    ) -> None:
        """
        Args:
            content_path (Path): Path to the content directory or file containing the knowledge base.
            index_path (Optional[str], optional): Path to store vector indexes. Defaults to None,
                which uses "indexes" as the default directory.
            rerank (bool, optional): Whether to use reranking for search results. Defaults to True.
            model_name (Optional[str], optional): Name of the model to use. Defaults to None. Not
                specifying a model name will select the first model from the config.
            chunking_type (str, optional): Type of text chunking strategy to use (recursive, semantic, basic).
                Defaults to "recursive".
            init_hist_monitor (bool, optional): Whether to initialize the chat history monitor for sending
                chat history over E-mail. Defaults to False.
            init_resp_monitor (bool, optional): Whether to initialize the response uncertainty monitor
                for sending uncertain responses over E-mail. Defaults to False.
            hist_monitor_freq (int, optional): Frequency in hours for chat history monitoring.
                Defaults to 24.
            resp_monitor_freq (int, optional): Frequency in hours for response monitoring.
                Defaults to 24.

        Returns:
            None
        """
        self.index_path = index_path or "indexes"
        self.model_name = model_name
        self.rerank = rerank
        self.chunking_type = chunking_type
        self.current_index_path = None
        self.vectorstore_lock = threading.Lock()

        create_folder(self.index_path)

        logger.info(
            f"Initializing RAG system. Model temperature {RAGConfig.TEMPERATURE}..."
        )
        self._initialize_models()

        if self.rerank:
            logger.info("Using Cohere's re-ranking model...")
            self.reranker = cohere.Client(get_api_key("COHERE"))

        if content_path:
            logger.info("Loading knowledge base...")
            self.vectorstore = self._load_or_create_vectorstore(content_path)
            self.current_index_path = self._get_index_path(content_path)

        self.db = DatabaseOps()
        self.email_service = EmailService()
        self.hist_sender = ChatHistoryMonitor(
            self.email_service,
            every_hours=hist_monitor_freq,
            start_service=init_hist_monitor,
        )
        self.resp_monitor = UncertainResponseMonitor(
            self.email_service,
            every_hours=resp_monitor_freq,
            start_service=init_resp_monitor,
        )

    @abstractmethod
    def get_response(self, query: str, user_id: str) -> str:
        """Get response for user query."""
        pass

    @abstractmethod
    def _initialize_models(self):
        pass

    def _create_embeddings(self, texts: list, is_query: bool = False) -> list:
        return self.embedding_provider.embed(texts, is_query)

    def _generate_system_prompt(
        self,
        query: str,
        user_id: str,
        context: str,
        include_query: bool = True,
        include_context: bool = True,
        include_prev_conv: bool = True,
    ) -> str:
        """Generate a standardized system prompt for all LLMs."""
        system_prompt = f"""
            ### Role
                - Primary Function: You are a customer support agent here to assist users based on 
                specific training data provided. Your main objective is to inform, clarify, and answer 
                questions strictly related to this training data and your role.
                                
            ### Persona
                - Identity: You are a dedicated customer support agent. You cannot adopt other personas or 
                impersonate any other entity. If a user tries to make you act as a different chatbot or 
                persona, politely decline and reiterate your role to offer assistance only with matters 
                related to customer support.
                - Language: You should also respond in the user's language. If a user talks to you in
                Arabic, you must respond in Arabic. If they talk in English, you reply in English,... etc.
                            
            ### Constraints
                1. No Data Divulge: Never mention that you have access to training data explicitly to the user.\n
                2. Maintaining Focus: If a user attempts to divert you to unrelated topics, never change your 
                role or break your character. Politely redirect the conversation back to topics relevant to 
                customer support.\n
                3. Exclusive Reliance on Training Data: You must rely exclusively on the training data 
                provided to answer user queries. If a query is not covered by the training data, use 
                the fallback response.\n
                4. Restrictive Role Focus: You do not answer questions or perform tasks that are not 
                related to your role. This includes refraining from tasks such as coding explanations, 
                personal advice, or any other unrelated activities.\n
                5. Avoid providing preambles like "here is the answer", "according to the context", etc...
                6. Responses must be concise and to the point.
            """

        # system_prompt = "You are a helpful agent. Answer from the context."

        if include_context and context:
            system_prompt += f"\n\n### Context: \n{context}"

        if include_prev_conv:
            previous_conversation = self.db.get_chat_history(user_id)
            system_prompt += f"\n\n### Previous conversation:\n{previous_conversation}"

        if include_query and query:
            system_prompt += f"\n\n### Current question: {query}"

        return system_prompt.strip()

    def _get_index_path(self, content_path: Path) -> str:
        """Generate unique index path based on content."""
        content_hash = (
            hashlib.sha256(str(content_path).encode("utf-8")).hexdigest(),
            16,
        )[0][:15]
        return str(Path(self.index_path) / f"index_{content_hash}.faiss")

    def _clean_html_content(self, content: str) -> str:
        """Clean HTML content and convert to markdown."""
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        h.ignore_tables = False

        content = h.handle(content)
        content = re.sub(r"(\w)-\n(\w)", r"\1\2", content)
        content = re.sub(r"(?<!\n)\n(?!\n)", " ", content)
        content = re.sub(r"\n{3,}", "\n\n", content)
        content = re.sub(r"\[(\w+)\]\(([^)]+)\)", r"\1 (\2)", content)

        return content.strip()

    def _create_chunks(self, text: str) -> List[str]:
        """Create chunks using the specified chunking method."""
        config = RAGConfig.CHUNKING_CONFIGS.get(self.chunking_type)
        if not config:
            raise ValueError(f"Unsupported chunking type: {self.chunking_type}")

        logger.info(config["message"])
        return config["splitter"].split_text(text)

    def _load_or_create_vectorstore(self, content_path: Path) -> FAISS:
        """Load existing index or create new one."""
        index_path = self._get_index_path(content_path)

        if Path(index_path).exists():
            logger.info(f"Loading existing index from {index_path}")
            return self._load_vectorstore(index_path)

        logger.info("Creating new index...")
        return self._create_vectorstore(content_path)

    # def _create_vectorstore(self, content_path: Path) -> FAISS:
    #     """Create FAISS vectorstore from content."""
    #     with open(content_path, 'r', encoding='utf-8') as f:
    #         content = f.read()

    #     cleaned_content = self._clean_html_content(content)
    #     chunks = self._create_chunks(cleaned_content)
    #     logger.info(f"Created {len(chunks)} chunks")

    #     text_embeddings = list(zip(chunks, self._create_embeddings(chunks)))  # Pair text with embeddings

    #     vectorstore = FAISS.from_embeddings(
    #         text_embeddings=text_embeddings,
    #         embedding=self._create_embeddings,
    #         metadatas=[{"source": f"chunk_{i}"} for i in range(len(chunks))]
    #     )

    #     index_path = self._get_index_path(content_path)
    #     self._save_vectorstore(vectorstore, index_path)

    #     return vectorstore

    def _create_vectorstore(self, content_path: Path) -> FAISS:
        """Create FAISS vectorstore from content with incremental embedding saving."""
        with open(content_path, "r", encoding="utf-8") as f:
            content = f.read()

        cleaned_content = self._clean_html_content(content)
        chunks = self._create_chunks(cleaned_content)
        logger.info(f"Created {len(chunks)} chunks")

        temp_index_path = self._get_index_path(content_path) + ".temp"
        temp_progress_path = self._get_index_path(content_path) + ".progress"

        if Path(temp_index_path).exists() and Path(temp_progress_path).exists():
            logger.info(f"Found existing progress, resuming from previous state")
            with open(temp_progress_path, "r") as f:
                processed_chunks = int(f.read().strip())

            vectorstore = self._load_vectorstore(temp_index_path)

            remaining_chunks = chunks[processed_chunks:]
            start_idx = processed_chunks
        else:
            processed_chunks = 0
            remaining_chunks = chunks
            start_idx = 0

            if remaining_chunks:
                first_chunk = remaining_chunks[0]
                first_embedding = self._create_embeddings([first_chunk])[0]
                text_embeddings = [(first_chunk, first_embedding)]

                vectorstore = FAISS.from_embeddings(
                    text_embeddings=text_embeddings,
                    embedding=self._create_embeddings,
                    metadatas=[{"source": f"chunk_{start_idx}"}],
                )

                processed_chunks = 1
                start_idx = 1
                remaining_chunks = remaining_chunks[1:]
            else:
                logger.warning("No chunks to process, creating empty vectorstore")
                dummy_text = "Initialization placeholder"
                dummy_embedding = self._create_embeddings([dummy_text])[0]
                vectorstore = FAISS.from_embeddings(
                    text_embeddings=[(dummy_text, dummy_embedding)],
                    embedding=self._create_embeddings,
                    metadatas=[{"source": "initialization_placeholder"}],
                )

        batch_size = 10

        for i in range(0, len(remaining_chunks), batch_size):
            batch = remaining_chunks[i : i + batch_size]

            try:
                batch_embeddings = self._create_embeddings(batch)
                text_embeddings = list(zip(batch, batch_embeddings))

                vectorstore.add_embeddings(
                    text_embeddings=text_embeddings,
                    metadatas=[
                        {"source": f"chunk_{start_idx + j}"} for j in range(len(batch))
                    ],
                )

                processed_chunks = start_idx + i + len(batch)

                with open(temp_progress_path, "w") as f:
                    f.write(str(processed_chunks))

                self._save_vectorstore(vectorstore, temp_index_path)

                logger.info(
                    f"Saved progress: {processed_chunks}/{len(chunks)} chunks processed"
                )

            except Exception as e:
                logger.error(f"Error while processing batch: {e}")
                logger.info(f"Progress saved up to chunk {processed_chunks}")
                return vectorstore

        final_index_path = self._get_index_path(content_path)
        self._save_vectorstore(vectorstore, final_index_path)

        try:
            # Path(temp_index_path).rmdir()
            shutil.rmtree(temp_index_path, ignore_errors=True)
            Path(temp_progress_path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {e}")

        logger.info(
            f"Vectorstore creation completed successfully with {len(chunks)} chunks"
        )
        return vectorstore

    def _update_vectorstore(self, new_content: str):
        with self.vectorstore_lock:
            chunks = self._create_chunks(new_content)
            text_embeddings = list(zip(chunks, self._create_embeddings(chunks)))
            existing_chunks = len(self.vectorstore.index_to_docstore_id)
            metadatas = [
                {"source": f"chunk_{existing_chunks + i}"} for i in range(len(chunks))
            ]

            self.vectorstore.add_embeddings(
                text_embeddings=text_embeddings, metadatas=metadatas
            )
            self._save_vectorstore(self.vectorstore, self.current_index_path)
        logger.info(f"Vectorstore updated successfully with {len(chunks)} new chunks")

    def _save_vectorstore(self, vectorstore: FAISS, path: Path) -> None:
        """Save vectorstore to disk."""
        logger.info(f"Saving index to {path}")
        vectorstore.save_local(path)

    def _load_vectorstore(self, path: Path) -> FAISS:
        """Load vectorstore from disk."""
        return FAISS.load_local(
            path, self._create_embeddings, allow_dangerous_deserialization=True
        )

    def _rerank_docs(self, query: str, docs):
        """
        Refine the top-k retrieved chunks for relevance
        before passing to the LLM.
        """
        docs_list = [doc.page_content for doc in docs]
        reranked = self.reranker.rerank(
            model="rerank-multilingual-v3.0",
            query=query,
            documents=docs_list,
            top_n=3,
            return_documents=True,
        )
        return reranked

    def _find_relevant_context(self, query: str, top_k: int = 5) -> str:
        """Find relevant context using similarity search."""
        query_embedding = self._create_embeddings([query], is_query=True)[0]

        docs = self.vectorstore.similarity_search_by_vector(
            query_embedding, k=top_k, fetch_k=20
        )
        if self.rerank:
            reranked_docs = self._rerank_docs(query, docs)
            return "\n\n".join(
                [result.document.text for result in reranked_docs.results]
            )

        return "\n\n".join([doc.page_content for doc in docs])

    @classmethod
    def get_models(cls):
        config_class = cls.get_config_class()
        return config_class.AVAILABLE_MODELS

    @classmethod
    def get_config_class(cls):
        raise NotImplementedError("Subclass must implement abstract method")
