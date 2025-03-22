from mistralai import Mistral
from chatbot.rag.base_rag import *
from chatbot.config import MistralConfig, get_api_key
from chatbot.embeddings.base_embedding import CohereEmbedding, MistralEmbedding

class MistralRAG(BaseRAG):
    def _initialize_models(self):
        config = MistralConfig()
        api_key = get_api_key("MISTRAL")
        self.mistral_client = Mistral(api_key=api_key)
        self.mistral_model = self.model_name or next(iter(config.AVAILABLE_MODELS))
        self.in_price, self.out_price = config.AVAILABLE_MODELS[self.mistral_model]

        self.embedding_provider = CohereEmbedding(get_api_key("COHERE"))
        # self.embedding_provider = MistralEmbedding(get_api_key("MISTRAL"))

        logger.info(f"Using mistral model: {self.mistral_model}")

    def _create_messages(self, query: str, context: str, user_id: str) -> list:
        """Format the conversation messages for Mistral API."""
        system_prompt = self._generate_system_prompt(query, user_id, context, 
                                                     include_prev_conv=False,
                                                     include_query=False)

        messages = [
            {"role": "system", "content": system_prompt}
        ]

        history = self.db.get_chat_history(user_id) # returns "No previous conversations." for new users
        if "No previous conversations." not in history:
            for hist in history:
                messages.extend([
                    {"role": "user", "content": hist["question"]},
                    {"role": "assistant", "content": hist["answer"]}
                ])

        messages.append({"role": "user", "content": query})
        return messages
    
    # @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_response(self, query: str, user_id: str) -> str:
        """Get response from Mistral API."""
        context = self._find_relevant_context(query)
        messages = self._create_messages(query, context, user_id)

        try:
            response = await self.mistral_client.chat.complete_async(
                model=self.model_name,
                messages=messages,
                temperature=Config.TEMPERATURE
            )

            response_text = response.choices[0].message.content.strip()
            request_usage = response.usage

            self.db.append_chat_history(user_id, 
                                     query, 
                                     response_text, 
                                     self.mistral_model, 
                                     self.embedding_provider.embedding_model)
            
            self.db.append_cost(user_id,
                             self.mistral_model,
                             self.embedding_provider.embedding_model,
                             request_usage.prompt_tokens,
                             request_usage.completion_tokens,
                             self.in_price,
                             self.out_price)
            
            return response_text

        except Exception as e:
            logger.error(f"Mistral API error: {e}")
            return "I'm sorry, I couldn't process your request."

    @classmethod
    def get_config_class(cls):
        return MistralConfig

