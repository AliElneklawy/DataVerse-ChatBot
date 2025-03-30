Examples
========

This page provides practical examples of how to use DataVerse ChatBot for various use cases.

Basic Usage
-----------

Setting Up a Simple RAG Chatbot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio
   from chatbot.rag.openai_rag import OpenAIRAG
   from chatbot.utils.paths import WEB_CONTENT_DIR, INDEXES_DIR
   from uuid import uuid4

   # Path to your content
   content_path = WEB_CONTENT_DIR / "mycontent.txt"

   # Create a unique user ID
   user_id = str(uuid4())

   # Initialize the RAG system
   rag = OpenAIRAG(
       content_path=content_path,
       index_path=INDEXES_DIR,
       model_name="gpt-3.5-turbo-0125",
       chunking_type="recursive",
       rerank=True



   # Function to chat with the bot
   async def chat():
       while True:
           query = input("You: ")
           if query.lower() in ["exit", "quit"]:
               break
               
           response = await rag.get_response(query, user_id)
           print(f"Bot: {response}")

   # Run the chat loop
   if __name__ == "__main__":
       asyncio.run(chat())

   Web Crawling and Content Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio
   import tldextract
   from chatbot.crawler import Crawler
   from chatbot.utils.paths import WEB_CONTENT_DIR
   
   async def crawl_website(url, max_depth=2):
       # Extract domain name from URL
       domain_name = tldextract.extract(url).domain
       
       # Initialize crawler
       crawler = Crawler(
           base_url=url,
           domain_name=domain_name,
           client="crawl4ai"  # or "scrapegraph"


       
       # Extract content
       content_path = await crawler.extract_content(
           link=url,
           webpage_only=False,  # Crawl linked pages
           max_depth=max_depth  # Crawl depth limit


       
       print(f"Content extracted and saved to: {content_path}")
       return content_path
   
   if __name__ == "__main__":
       website_url = "https://example.com"
       asyncio.run(crawl_website(website_url))

   File Processing
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from chatbot.utils.file_loader import FileLoader
   from chatbot.utils.paths import WEB_CONTENT_DIR
   
   def process_file(file_path, output_name="extracted_content.txt"):
       # Create output path
       output_path = WEB_CONTENT_DIR / output_name
       
       # Initialize file loader
       loader = FileLoader(
           file_path=file_path,
           content_path=output_path,
           client="docling"  # or "langchain"


       
       # Extract content
       documents = loader.extract_from_file()
       
       if documents:
           print(f"Successfully extracted {len(documents)} documents")
           print(f"Content saved to: {output_path}")
           return output_path
       else:
           print("Failed to extract content")
           return None
   
   if __name__ == "__main__":
       # Process a PDF file
       pdf_path = "data/training_files/document.pdf"
       process_file(pdf_path, "pdf_content.txt")
       
       # Process a DOCX file
       docx_path = "data/training_files/document.docx"
       process_file(docx_path, "docx_content.txt")

   Advanced Usage
-----------------

   Using Voice Mode
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio
   from chatbot.voice_mode import VoiceMode
   from chatbot.rag.claude_rag import ClaudeRAG
   from chatbot.utils.paths import WEB_CONTENT_DIR, INDEXES_DIR
   from uuid import uuid4
   
   # Initialize voice mode
   voice = VoiceMode()
   
   # Initialize RAG
   rag = ClaudeRAG(
       content_path=WEB_CONTENT_DIR / "mycontent.txt",
       index_path=INDEXES_DIR


   
   # User ID for tracking chat history
   user_id = str(uuid4())
   
   async def voice_chat():
       print("Press Enter to start recording (5-second limit)...")
       input()
       
       # Record and transcribe
       wav_path = voice.start_recording()
       transcription = voice.transcribe(wav_path)
       
       print(f"You said: {transcription}")
       
       # Get response
       response = await rag.get_response(transcription, user_id)
       print(f"Bot: {response}")
       
       # Convert response to speech
       voice.text_to_speech(response)
   
   if __name__ == "__main__":
       asyncio.run(voice_chat())

   Custom Dataset Creation and Classifier Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from chatbot.utils.make_dataset import create_dataset
   from chatbot.utils.train_clf import train_classifier
   from chatbot.utils.paths import DATASETS_DIR, MODELS_DIR
   
   # Step 1: Create a dataset from RAG responses
   def prepare_dataset():
       # Create dataset with labels (1 for uncertain, 0 for certain)
       dataset = create_dataset(
           input_file=DATASETS_DIR / "raw_responses.csv",
           output_file=DATASETS_DIR / "labeled_responses.csv",
           model_name="all-MiniLM-L6-v2"


       
       print(f"Dataset created with {len(dataset)} samples")
       return dataset
   
   # Step 2: Train a classifier on the dataset
   def train_uncertainty_classifier(dataset_path):
       # Train the classifier
       metrics = train_classifier(
           dataset_path=dataset_path,
           model_type="xgboost",  # or "random_forest"
           output_path=MODELS_DIR / "clf.pkl",
           test_size=0.2,
           random_state=42


       
       print("Classifier trained successfully")
       print(f"Accuracy: {metrics['accuracy']:.4f}")
       print(f"Precision: {metrics['precision']:.4f}")
       print(f"Recall: {metrics['recall']:.4f}")
   
   if __name__ == "__main__":
       # Prepare dataset
       dataset = prepare_dataset()
       
       # Train classifier
       train_uncertainty_classifier(DATASETS_DIR / "labeled_responses.csv")

   Implementing a Custom RAG
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from chatbot.rag.base_rag import BaseRAG
   from chatbot.embeddings.base_embedding import HuggingFaceEmbedding
   from chatbot.config import Config
   
   class CustomRAG(BaseRAG):
       """Custom RAG implementation with a local HuggingFace model."""
       
       def _initialize_models(self):
           """Initialize models for the RAG system."""
           # Use a local embedding model
           self.embedding_provider = HuggingFaceEmbedding(
               embedding_model="sentence-transformers/all-mpnet-base-v2",
               device="cpu"


           
           # Custom model configuration
           self.model_name = "custom-model"
           self.in_price = 0.0  # Free local model
           self.out_price = 0.0  # Free local model
       
       async def get_response(self, query, user_id):
           """Generate a response using a custom approach."""
           # Find relevant context
           context = self._find_relevant_context(query, top_k=5)
           
           # Create a prompt with the context
           prompt = f"Context:\n{context}\n\nQuestion: {query}"
           
           # ... your custom logic to generate a response ...
           # This could use a local model, rule-based system, or external API
           
           # For this example, just return a placeholder
           response = f"This is a custom RAG response for: {query}"
           
           # Add to chat history
           self.db.append_chat_history(
               user_id=user_id,
               question=query,
               answer=response,
               model_used=self.model_name,
               embedding_model_used=self.embedding_provider.embedding_model


           
           return response
       
       @classmethod
       def get_config_class(cls):
           """Return the configuration class for this RAG."""
           return Config