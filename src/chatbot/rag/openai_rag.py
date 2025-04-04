from openai import AsyncOpenAI
from chatbot.rag.base_rag import *
from chatbot.config import OpenAIConfig, get_api_key
from chatbot.embeddings.base_embedding import OpenAIEmbedding, CohereEmbedding


class OpenAIRAG(BaseRAG):
    def _initialize_models(self) -> None:
        """Initialize OpenAI models."""
        config = OpenAIConfig()
        api_key = get_api_key("OPENAI")
        self.openai_client = AsyncOpenAI(api_key=api_key)
        self.openai_model = self.model_name or next(iter(config.AVAILABLE_MODELS))
        self.in_price, self.out_price = config.AVAILABLE_MODELS[self.openai_model]
        # self.embedding_provider = OpenAIEmbedding(get_api_key("OPENAI"))
        self.embedding_provider = CohereEmbedding(get_api_key("COHERE"))

        logger.info(f"Using OpenAI model: {self.openai_model}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_response(self, query: str, user_id: str) -> str:
        """Get response using OpenAI."""
        context = self._find_relevant_context(query)
        messages = self._create_messages(query, context, user_id)

        try:
            response = await self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                max_tokens=RAGConfig.MAX_OUT_TOKENS,
                temperature=RAGConfig.TEMPERATURE,
                stream=False,
            )
            
            response_text = response.choices[0].message.content.strip()
            request_usage = response.usage

            self.db.append_cost(user_id,
                             self.openai_model,
                             self.embedding_provider.embedding_model,
                             request_usage.prompt_tokens,
                             request_usage.completion_tokens,
                             self.in_price,
                             self.out_price)
 
            self.db.append_chat_history(user_id, 
                                     query, 
                                     response_text, 
                                     self.openai_model, 
                                     self.embedding_provider.embedding_model)
            return response_text
            
        except Exception as e:
            logger.error(f"Error getting OpenAI response: {e}")
            raise
    
    def _create_messages(self, query: str, context: str, user_id: str) -> List[Dict[str, str]]:
        """Create messages for chat completion."""
        system_prompt = self._generate_system_prompt(query, user_id, context, 
                                                     include_context=False,
                                                     include_prev_conv=False,
                                                     include_query=False)
        system_message = {
            "role": "system",
            "content": system_prompt
        }
        
        context_message = {
            "role": "system",
            "content": f"Context:\n{context}"
        }
        
        messages = [system_message, context_message]

        history = self.db.get_chat_history(user_id) # returns "No previous conversations." for new users
        if "No previous conversations." not in history:
            for hist in history:
                messages.extend([
                    {"role": "user", "content": hist["question"]},
                    {"role": "assistant", "content": hist["answer"]}
                ])
        
        messages.append({"role": "user", "content": query})
        return messages
    
    @classmethod
    def get_config_class(cls):
        return OpenAIConfig
