import anthropic
from chatbot.rag.base_rag import *
from chatbot.config import ClaudeConfig, get_api_key
from chatbot.embeddings.base_embedding import CohereEmbedding


class ClaudeRAG(BaseRAG):
    def _initialize_models(self):
        """Initialize Claude chat model and embedding model."""
        config = ClaudeConfig()
        api_key = get_api_key("CLAUDE")
        self.claude_client = anthropic.AsyncAnthropic(api_key=api_key)
        self.claude_model = self.model_name or next(iter(config.AVAILABLE_MODELS))
        self.in_price, self.out_price = config.AVAILABLE_MODELS[self.claude_model]
        self.embedding_provider = CohereEmbedding(get_api_key("COHERE"))

        logger.info(f"Using claude model: {self.claude_model}")
    
    def _create_messages(self, query: str, context: str, user_id: str) -> List[Dict[str, str]]:
        """Create messages formatted for Claude API."""        
        messages = []

        history = self.db.get_chat_history(user_id) # returns "No previous conversations." for new users
        if "No previous conversations." not in history:
            for hist in history:
                messages.extend([
                    {"role": "user", "content": hist["question"]},
                    {"role": "assistant", "content": hist["answer"]}
                ])
        
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})
        
        return messages
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_response(self, query, user_id):
        context = self._find_relevant_context(query, top_k=3)
        messages = self._create_messages(query, context, user_id)
        system_prompt = self._generate_system_prompt(query, user_id, context, 
                                                     include_context=False,
                                                     include_prev_conv=False,
                                                     include_query=False)
        
        response = await self.claude_client.messages.create(
                model=self.claude_model,
                max_tokens=RAGConfig.MAX_OUT_TOKENS,
                messages=messages,
                system=system_prompt,
                temperature=RAGConfig.TEMPERATURE
            )
                
        response_text = response.content[0].text
        request_usage = response.usage

        self.db.append_cost(user_id,
                         self.claude_model,
                         self.embedding_provider.embedding_model,
                         request_usage.input_tokens,
                         request_usage.output_tokens,
                         self.in_price,
                         self.out_price)
        
        self.db.append_chat_history(user_id, 
                                 query, 
                                 response_text, 
                                 self.claude_model, 
                                 self.embedding_provider.embedding_model)

        return response_text
    
    @classmethod
    def get_config_class(cls):
        return ClaudeConfig
