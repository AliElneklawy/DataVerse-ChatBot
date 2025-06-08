import cohere
from openai import AsyncOpenAI
from chatbot.rag.base_rag import *
from chatbot.rag.openai_rag import OpenAIRAG
from ..config import DeepSeekConfig, get_api_key
from ..embeddings.base_embedding import CohereEmbedding


class DeepseekRAG(OpenAIRAG, BaseRAG):
    def _initialize_models(self):
        """Initialize Deepseek chat model and Cohere embedding model."""
        config = DeepSeekConfig()
        api_key = get_api_key("DEEPSEEK")
        self.deepseek_client = AsyncOpenAI(api_key=api_key, base_url=config.BASE_URL)

        self.deepseek_model = self.model_name or next(iter(config.AVAILABLE_MODELS))
        self.in_price, self.out_price = config.AVAILABLE_MODELS[self.deepseek_model]
        self.embedding_provider = CohereEmbedding(get_api_key("COHERE"))

        logger.info(f"Using DeepSeek model {self.deepseek_model}")

    # @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_response(self, query, user_id):
        context = self._find_relevant_context(query)
        messages = self._create_messages(query, context, user_id)

        try:
            response = await self.deepseek_client.chat.completions.create(
                model=self.deepseek_model,
                messages=messages,
                max_tokens=RAGConfig.MAX_OUT_TOKENS,
                temperature=RAGConfig.TEMPERATURE,
                stream=False,
            )
            response_text = response.choices[0].message.content
            request_usage = response.usage

            self.db.append_cost(
                user_id,
                self.deepseek_model,
                self.embedding_provider.embedding_model,
                request_usage.prompt_tokens,
                request_usage.completion_tokens,
                self.in_price,
                self.out_price,
            )

            self.db.append_chat_history(
                user_id,
                query,
                response_text,
                self.deepseek_model,
                self.embedding_provider.embedding_model,
            )
            return response_text
        except Exception as e:
            logger.error(f"Error getting DeepSeek response: {e}")
            raise
