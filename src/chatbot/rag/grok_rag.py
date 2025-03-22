from openai import AsyncOpenAI
from chatbot.rag.base_rag import *
from chatbot.config import GrokConfig, get_api_key
from chatbot.embeddings.base_embedding import CohereEmbedding, MistralEmbedding

 
class GrokRAG(BaseRAG):
    def _initialize_models(self):
        config = GrokConfig()
        api_key = get_api_key("GROK")
        self.grok_client = AsyncOpenAI(
            api_key=api_key,
            base_url=config.BASE_URL,
        )
        self.grok_model = self.model_name or next(iter(config.AVAILABLE_MODELS))
        self.in_price, self.out_price = config.AVAILABLE_MODELS[self.grok_model]

        self.embedding_provider = CohereEmbedding(get_api_key("COHERE"))
        # self.embedding_provider = MistralEmbedding(get_api_key("Mistral"))

        
        logger.info(f"Using grok model: {self.grok_model}")
    
    def _create_messages(self, query: str, context: str, user_id: str) -> List[Dict[str, str]]:
        """Generate messages for Grok API request."""
        system_prompt = self._generate_system_prompt(query, user_id, context,
                                                     include_prev_conv=False,
                                                     include_query=False)

        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_response(self, query, user_id):
        context = self._find_relevant_context(query)
        messages = self._create_messages(query, context, user_id)

        response = await self.grok_client.chat.completions.create(
            model=self.grok_model,
            messages=messages,
            max_tokens=Config.MAX_OUT_TOKENS,
            temperature=Config.TEMPERATURE,
        )

        response_text = response.choices[0].message.content.strip()
        request_usage = response.usage

        self.db.append_cost(user_id,
                         self.grok_model,
                         self.embedding_provider.embedding_model,
                         request_usage.prompt_tokens,
                         request_usage.completion_tokens,
                         self.in_price,
                         self.out_price)
 
        self.db.append_chat_history(user_id, 
                                 query, 
                                 response_text, 
                                 self.grok_model, 
                                 self.embedding_provider.embedding_model)
        return response_text

    @classmethod
    def get_config_class(cls):
        return GrokConfig
