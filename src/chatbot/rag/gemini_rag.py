from chatbot.rag.base_rag import *
import google.generativeai as genai
from chatbot.config import GeminiConfig, get_api_key
from chatbot.embeddings.base_embedding import CohereEmbedding


class GeminiRAG(BaseRAG):
    def _initialize_models(self) -> None:
        """Initialize Gemini models."""
        config = GeminiConfig()
        api_key = get_api_key("GEMINI")
        genai.configure(api_key=api_key)
        self.gemini_model = self.model_name or next(iter(config.AVAILABLE_MODELS))
        self.in_price, self.out_price = config.AVAILABLE_MODELS[self.gemini_model]
        self.model = genai.GenerativeModel(self.gemini_model)
        self.embedding_provider = CohereEmbedding(get_api_key("COHERE"))

        logger.info(f"Using gemini model: {self.gemini_model}")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def get_response(self, query: str, user_id: str) -> str:
        """Get response using Gemini."""
        context = self._find_relevant_context(query, top_k=10)
        prompt = self._generate_system_prompt(query, user_id, context)

        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=RAGConfig.MAX_OUT_TOKENS,
                    temperature=RAGConfig.TEMPERATURE,
                ),
            )

            response_text = response.text.strip()
            response_text = re.sub(r"\{\{.*?\}\}", "", response_text)
            response_text = response_text.replace("Answer:", "").strip()
            request_usage = response.usage_metadata

            self.db.append_cost(
                user_id,
                self.gemini_model,
                self.embedding_provider.embedding_model,
                request_usage.prompt_token_count,
                request_usage.candidates_token_count,
                self.in_price,
                self.out_price,
            )

            self.db.append_chat_history(
                user_id,
                query,
                response_text,
                self.gemini_model,
                self.embedding_provider.embedding_model,
            )
            return response_text

        except Exception as e:
            logger.error(f"Error getting Gemini response: {e}")
            raise

    @classmethod
    def get_config_class(cls):
        return GeminiConfig
