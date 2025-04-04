import cohere
from ..rag.base_rag import *
from ..config import CohereConfig, get_api_key
from ..embeddings.base_embedding import CohereEmbedding


class CohereRAG(BaseRAG):
    def _initialize_models(self) -> None:
        """Initialize Cohere embedding model."""
        config = CohereConfig()
        api_key = get_api_key("COHERE")
        self.cohere_client = cohere.AsyncClient(api_key)
        self.cohere_model = self.model_name or next(iter(config.AVAILABLE_MODELS)) # get the first model in the list
        self.in_price, self.out_price = config.AVAILABLE_MODELS[self.cohere_model]
        self.embedding_provider = CohereEmbedding(api_key)
        # self.embedding_provider = HuggingFaceEmbedding()
        # self.embedding_provider = OpenAIEmbedding(get_api_key("OPENAI"))

        logger.info(f"Using Cohere model: {self.cohere_model}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_response(self, query: str, user_id: str) -> str:
        """Get response using Cohere."""        
        context = self._find_relevant_context(query)
        system_prompt = self._generate_system_prompt(query, user_id, context)

        try:
            response = await self.cohere_client.generate(
                model=self.cohere_model,
                prompt=system_prompt,
                max_tokens=RAGConfig.MAX_OUT_TOKENS,
                temperature=RAGConfig.TEMPERATURE
            )
            response_text = response.generations[0].text.strip()
            request_usage = response.meta.billed_units

            self.db.append_cost(
                user_id,
                             self.cohere_model,
                             self.embedding_provider.embedding_model,
                             request_usage.input_tokens,
                             request_usage.output_tokens,
                             self.in_price,
                             self.out_price)
            
            self.db.append_chat_history(user_id, 
                                     query, 
                                     response_text, 
                                     self.cohere_model, 
                                     self.embedding_provider.embedding_model)

            return response_text
        except Exception as e:
            logger.error(f"Error getting Cohere response: {e}")
            return "I'm sorry, I couldn't process your request at the moment."

    @classmethod
    def get_config_class(cls):
        return CohereConfig
