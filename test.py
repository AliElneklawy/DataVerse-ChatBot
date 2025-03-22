# # from langchain_experimental.text_splitter import SemanticChunker
# # from langchain_text_splitters import CharacterTextSplitter

# # # CharacterTextSplitter().split_text
# # s = """"Overview
# # ScrapeGraphAI is an open-source Python library designed to revolutionize scraping tools. In today’s data-intensive digital landscape, this library stands out by integrating Large Language Models (LLMs) and modular graph-based pipelines to automate the scraping of data from various sources (e.g., websites, local files etc.).

# # Simply specify the information you need to extract, and ScrapeGraphAI handles the rest, providing a more flexible and low-maintenance solution compared to traditional scraping tools.

# # Why ScrapegraphAI?
# # Traditional web scraping tools often rely on fixed patterns or manual configuration to extract data from web pages. ScrapegraphAI, leveraging the power of LLMs, adapts to changes in website structures, reducing the need for constant developer intervention. This flexibility ensures that scrapers remain functional even when website layouts change.

# # We support many LLMs including GPT, Gemini, Groq, Azure, Hugging Face etc. as well as local models which can run on your machine using Ollama.

# # AI Models and Token Limits
# # ScrapGraphAI supports a wide range of AI models from various providers. Each model has a specific token limit, which is important to consider when designing your scraping pipelines. Here’s an overview of the supported models and their token limits:

# # OpenAI Models
# # GPT-3.5 Turbo (16,385 tokens)

# # GPT-4 (8,192 tokens)

# # GPT-4 Turbo Preview (128,000 tokens)

# # GPT-4o (128000 tokens)

# # GTP-4o-mini (128000 tokens)

# # Azure OpenAI Models
# # GPT-3.5 Turbo (16,385 tokens)

# # GPT-4 (8,192 tokens)

# # GPT-4 Turbo Preview (128,000 tokens)

# # GPT-4o (128000 tokens)

# # GTP-4o-mini (128000 tokens)

# # Google AI Models
# # Gemini Pro (128,000 tokens)

# # Gemini 1.5 Pro (128,000 tokens)

# # Anthropic Models
# # Claude Instant (100,000 tokens)

# # Claude 2 (200,000 tokens)

# # Claude 3 (200,000 tokens)

# # Mistral AI Models
# # Mistral Large (128,000 tokens)

# # Open Mistral 7B (32,000 tokens)

# # Open Mixtral 8x7B (32,000 tokens)

# # For a complete list of supported models and their token limits, please refer to the API documentation.

# # Understanding token limits is crucial for optimizing your scraping tasks. Larger token limits allow for processing more text in a single API call, which can be beneficial for scraping lengthy web pages or documents."""

# # import cohere
# # class CohereEmbedding():
# #     def __init__(self, api_key, 
# #                  embedding_model: str = "embed-multilingual-v3.0"):
# #         self.client = cohere.Client(api_key)
# #         self.embedding_model = embedding_model

    
# #     def embed_documents(self, texts: list, is_query: bool = False) -> list:
# #         input_type = "search_query" if is_query else "search_document"
# #         return self.client.embed(
# #             texts=texts,
# #             model=self.embedding_model,
# #             input_type=input_type,
# #         ).embeddings
    
# # text_splitter = CharacterTextSplitter(chunk_size=500,
# #                 chunk_overlap=150,
# #                 length_function=len,
# #                 separators=["\n\n", "\n", " ", ""])

# # docs = text_splitter.split_text(s)

# # print(len(docs))

# from docling.document_converter import DocumentConverter

# source = r"D:\Studying\LECS\AWS\Module 2.pdf" # document per local path or URL
# converter = DocumentConverter()
# result = converter.convert(source)

# # Print results to console
# # print(result.document.export_to_markdown())

# # print(result.document.export_to_dict()) # export to JSON

# # Save results to a text file
# with open(r'C:\Users\Metro\OneDrive\Desktop\conversion_results.txt', 'w', encoding='utf-8') as f:
#     # f.write("\n\nMarkdown Conversion:\n")
#     f.write(result.document.export_to_markdown())

# from langchain_community.document_loaders import DirectoryLoader
# from langchain_docling import DoclingLoader
# from langchain_docling.loader import ExportType

# loader = DirectoryLoader(r"C:\Users\Metro\OneDrive\Desktop\New folder", loader_cls=DoclingLoader,
#                          show_progress=True, use_multithreading=True)
# docs = loader.load()

# with open(r"C:\Users\Metro\OneDrive\Desktop\New Text Document (2).txt", "w", encoding="utf-8") as f:
#     for doc in docs:
#         f.write(doc.page_content + "\n")

# from docling.document_converter import DocumentConverter
# from pathlib import Path

# # Define the input folder and output file paths
# input_folder = Path(r"C:\Users\Metro\OneDrive\Desktop\New folder")
# output_file = Path(r"C:\Users\Metro\OneDrive\Desktop\New Text Document (3).txt")

# # Get all files in the folder (no explicit looping needed here)
# input_paths = list(input_folder.glob("*"))

# # Initialize Docling's DocumentConverter
# converter = DocumentConverter()

# # Convert all files in one go
# results = converter.convert_all(input_paths, raises_on_error=False)

# # Combine all Markdown outputs into a single string
# combined_markdown = "\n\n---\n\n".join(
#     result.document.export_to_markdown() for result in results
# )

# # Write the combined Markdown to the output file
# with output_file.open("w", encoding="utf-8") as fp:
#     fp.write(combined_markdown)

# print(f"Converted files from {input_folder} to Markdown and saved to {output_file}")




# import sqlite3
# from datetime import datetime, timedelta

# last_hr = datetime.now()

# # print(last_hr)


# with sqlite3.connect(r"C:\Users\Metro\OneDrive\Desktop\WebRAG\data\database\history_and_usage.db") as conn:
#     ts = conn.execute("select timestamp from chat_history")
#     t = ts.fetchall()

# print(t[0][0])
# t = datetime.strptime(t[0][0], "%Y-%m-%d %H:%M:%S.%f")
# now = datetime.now()

# diff = now - t

# print(diff, timedelta(hours=1) > diff)

# with sqlite3.connect(r"C:\Users\Metro\OneDrive\Desktop\WebRAG\data\database\history_and_usage.db") as conn:
#     # Calculate the timestamp for 1 hour ago
#     one_hour_ago = datetime.now() - timedelta(hours=1)
    
#     # Execute the query with the calculated timestamp
#     ts = conn.execute(
#         "SELECT question, answer FROM chat_history WHERE timestamp >= ?",
#         (one_hour_ago,)  # Pass the timestamp as a tuple
#     )
    
#     # Fetch all results
#     t = ts.fetchall()

# # Print or process the results
# for row in t:
#     print(f"Question: {row[0]}, Answer: {row[1]}")

# # import smtplib

# # sender_email = "melneklawy1966@gmail.com"
# # receiver_email = "ali.mostafa.elneklawy@gmail.com"
# # password = "puli lejn inll xvto"
# # with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
# #     server.login(sender_email, password)
# #     server.sendmail(sender_email, receiver_email, "this is a test from python!")


# import re
# email_validation_pattern = r"^[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$"
# print(bool(re.fullmatch(email_validation_pattern, "es-ali.elsayed2024@alexu.edu.eg")))

# import re
# import dns.resolver
# import time
# import random

# # Your fixed regex for syntax validation
# email_regex = r"^[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$"

# def validate_email_dns(email):
#     # Step 1: Check syntax
#     if not re.fullmatch(email_regex, email):
#         return False, "Invalid email syntax"

#     # Step 2: Extract domain and check MX records
#     try:
#         domain = email.split('@')[1]
#         mx_records = dns.resolver.resolve(domain, 'MX')
#         if mx_records:
#             mx_servers = [str(record.exchange) for record in mx_records]
#             return True, f"Valid: MX records found for {domain} - {', '.join(mx_servers)}"
#         else:
#             return False, f"No MX records found for {domain}"
#     except dns.resolver.NXDOMAIN:
#         return False, f"Domain {domain} does not exist"
#     except dns.resolver.NoAnswer:
#         return False, f"No MX records found for {domain}"
#     except dns.resolver.Timeout:
#         return False, f"DNS query timed out for {domain}"
#     except Exception as e:
#         return False, f"Error checking {domain}: {str(e)}"

# # Test cases with rate limiting
# test_emails = [
#     "es-ali.elsayed2050@alexu.edu.eg",
#     "test@gmail.com",
#     "invalid@nonexistentdomain12345.com",
#     "user@.com"
# ]

# for email in test_emails:
#     is_valid, message = validate_email_dns(email)
#     print(f"{email}: {message}")
#     time.sleep(random.uniform(0.5, 2.0))  # Random delay to avoid rate limits

# import sqlite3
# # user_id = None
# # select_clause = "SELECT user_id" if user_id else "SELECT *"
# from src.chatbot.utils.utils import DatabaseOps
# d = DatabaseOps()
# from collections import defaultdict
# import json

# # subs = d.get_bot_sub() #with id: [(7856659305,)], without id: [(7856659305, 'مركز البحرين لتقنية المعلومات والذكاء الاصطناعي', '2025-03-14 14:05:58.520897')]
# # print(subs)

# h = d.get_chat_history(full_history=True)
# print(h)


# with open("sample.json", "w", encoding='utf-8') as outfile:
#     json.dump(h, outfile, indent=4, ensure_ascii=False)

# import json
# from collections import defaultdict

# with sqlite3.connect(r"C:\Users\Metro\OneDrive\Desktop\WebRAG\data\database\history_and_usage.db") as conn:
#     cursor = conn.execute("""
#         SELECT *
#         FROM chat_history 
#         ORDER BY user_id, timestamp
#     """)
#     results = cursor.fetchall()

# # Group by user_id
# grouped_history = defaultdict(list)
# for user_id, ts, question, answer, llm, embedder in results:
#     grouped_history[user_id].append({
#         "timestamp": ts,
#         "user": question,
#         "assistant": answer,
#         "llm": llm,
#         "embedder": embedder
#     })

# # Convert to a list of user objects for clearer separation
# history = [{"user_id": user_id, "interactions": interactions} 
#           for user_id, interactions in grouped_history.items()]

# with open("sample.json", "w", encoding='utf-8') as outfile:
#     json.dump(history, outfile, indent=4, ensure_ascii=False)

# from src.chatbot.utils.utils import DatabaseOps
# e = EmailService()
# d = DatabaseOps()

# chat_history = d.get_chat_history(full_history=True)

# e.send_email(
#                 subject="conversations",
#                 json_data=chat_history,
#             )

# import asyncio
# from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
# from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
# from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy

# async def main():
#     # Configure a 2-level deep crawl
#     config = CrawlerRunConfig(
#         deep_crawl_strategy=BFSDeepCrawlStrategy(
#             max_depth=2, 
#             include_external=False,
#             max_pages=10
#         ),
#         verbose=True
#     )

#     async with AsyncWebCrawler() as crawler:
#         results = await crawler.arun("https://ai.pydantic.dev/", config=config)

#         print(f"Crawled {len(results)} pages in total")

#         # Access individual results
#         for result in results[:3]:  # Show first 3 results
#             print(f"URL: {result.url}")
#             print(f"Depth: {result.metadata.get('depth', 0)}")

# if __name__ == "__main__":
#     asyncio.run(main())


# import asyncio
# import aiohttp
# from bs4 import BeautifulSoup
# from urllib.parse import urljoin, urlparse, urlunparse
# import logging
# import re
# from typing import List, Set, Tuple, Optional

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class AsyncCrawler:
#     def __init__(self, base_url, session=None):
#         self.base_url = self._clean_url(base_url)
#         self.visited_urls = set()
#         self.urls_to_visit = [(self.base_url, 0)]
#         self.session = session
#         self.semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
        
#         # File extensions to skip
#         self.skip_extensions = {
#             # Documents
#             '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.txt', '.rtf', '.odt',
#             # Images
#             '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp', '.tiff', '.ico',
#             # Audio/Video
#             '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv', '.wav', '.ogg', '.webm',
#             # Archives
#             '.zip', '.rar', '.7z', '.tar', '.gz', '.bz2',
#             # Other
#             '.css', '.js', '.xml', '.json', '.csv', '.rss', '.atom',
#             # Executable
#             '.exe', '.dll', '.apk', '.dmg', '.iso',
#         }
    
#     async def extract_urls(self, max_pages=None, max_depth=None, output_file="crawled_urls.txt"):
#         """
#         Asynchronously extract URLs to crawl starting from the base URL,
#         respecting depth and page limits.

#         Args:
#             max_pages (int, optional): Maximum number of pages to crawl. Defaults to None (unlimited).
#             max_depth (int, optional): Maximum depth to crawl. Defaults to None (unlimited).
#             output_file (str, optional): File to store the crawled URLs. Defaults to "crawled_urls.txt".

#         Returns:
#             Set[str]: A set of visited URLs that were crawled.
#         """
#         if self.session is None:
#             async with aiohttp.ClientSession() as session:
#                 self.session = session
#                 await self._extract_urls_impl(max_pages, max_depth)
#         else:
#             await self._extract_urls_impl(max_pages, max_depth)

#         # Save the visited URLs to a file
#         try:
#             with open(output_file, "w") as f:
#                 for url in sorted(self.visited_urls):
#                     f.write(url + "\n")
#             logger.info(f"Visited URLs saved to {output_file}")
#         except Exception as e:
#             logger.error(f"Error writing URLs to file {output_file}: {e}")
        
#         return self.visited_urls
    
#     async def _extract_urls_impl(self, max_pages=None, max_depth=None):
#         pages_crawled = 0
        
#         while self.urls_to_visit and (max_pages is None or pages_crawled < max_pages):
#             batch_size = min(20, len(self.urls_to_visit))
#             batch = []

#             for _ in range(batch_size):
#                 if not self.urls_to_visit:
#                     break
                
#                 current_url, current_depth = self.urls_to_visit.pop(0)
                
#                 if current_url in self.visited_urls:
#                     continue
#                 if max_depth is not None and current_depth > max_depth:
#                     continue
                
#                 # Skip URLs with file extensions we don't want to crawl
#                 if self._should_skip_url(current_url):
#                     logger.info(f"Skipping file: {current_url}")
#                     self.visited_urls.add(current_url)  # Mark as visited so we don't try again
#                     continue
                
#                 batch.append((current_url, current_depth))
            
#             if not batch:
#                 continue
            
#             batch_tasks = [
#                 self._crawl_page(url, depth) 
#                 for url, depth in batch
#             ]
            
#             results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
#             for i, result in enumerate(results):
#                 url, depth = batch[i]
                
#                 self.visited_urls.add(url)
#                 pages_crawled += 1
                
#                 logger.info(f"Crawled {url} (depth: {depth}) - Progress: {pages_crawled}/{max_pages if max_pages else 'unlimited'}")
                
#                 if isinstance(result, Exception):
#                     logger.error(f"Error crawling {url}: {result}")
#                     continue
                
#                 for link, new_depth in result:
#                     clean_link = self._clean_url(link)
#                     # Skip files we don't want to crawl
#                     if self._should_skip_url(clean_link):
#                         continue
#                     if clean_link not in self.visited_urls and not any(clean_link == u for u, _ in self.urls_to_visit):
#                         self.urls_to_visit.append((clean_link, new_depth))
        
#         logger.info(f"Crawling finished. Total pages crawled: {len(self.visited_urls)}")
    
#     def _should_skip_url(self, url):
#         """
#         Check if a URL should be skipped based on its file extension.
        
#         Args:
#             url (str): The URL to check.
            
#         Returns:
#             bool: True if the URL should be skipped, False otherwise.
#         """
#         parsed_url = urlparse(url)
#         path = parsed_url.path.lower()
        
#         # Check if the URL has a file extension we want to skip
#         for ext in self.skip_extensions:
#             if path.endswith(ext):
#                 return True
                
#         # Additional check for query parameters that might indicate a file download
#         if parsed_url.query and re.search(r'(download|file|attachment|document)', parsed_url.query, re.I):
#             return True
            
#         return False
    
#     async def _crawl_page(self, url, depth):
#         """
#         Asynchronously crawl a single page to extract linked URLs within the same domain.

#         Args:
#             url (str): The URL of the page to crawl.
#             depth (int): The current depth of the crawl.

#         Returns:
#             List[tuple[str, int]]: A list of tuples containing (cleaned URL, depth) for links found on the page.
#         """
#         links = []

#         async with self.semaphore:
#             try:
#                 timeout = aiohttp.ClientTimeout(total=20)
                
#                 # Check content type before downloading entire content
#                 async with self.session.head(url, timeout=timeout, allow_redirects=True) as head_response:
#                     content_type = head_response.headers.get('Content-Type', '')
                    
#                     # Skip if it's not an HTML file
#                     if content_type and 'text/html' not in content_type.lower():
#                         logger.info(f"Skipping non-HTML content: {url} (Content-Type: {content_type})")
#                         return links
                
#                 async with self.session.get(url, timeout=timeout) as response:
#                     if response.status != 200:
#                         logger.warning(f"Non-200 status code for {url}: {response.status}")
#                         return links
                    
#                     # Double-check content type from actual response
#                     content_type = response.headers.get('Content-Type', '')
#                     if 'text/html' not in content_type.lower():
#                         logger.info(f"Skipping non-HTML content: {url} (Content-Type: {content_type})")
#                         return links
                    
#                     html = await response.text()
#                     soup = BeautifulSoup(html, "html.parser")
                    
#                     for link in soup.find_all("a", href=True):
#                         next_url = urljoin(url, link["href"])
#                         if not next_url.startswith(('http://', 'https://')):
#                             continue
#                         if not self._is_same_domain(next_url):
#                             continue
                        
#                         clean_url = self._clean_url(next_url)
#                         if clean_url != url:
#                             links.append((clean_url, depth + 1))
#             except Exception as e:
#                 logger.error(f"Error crawling {url}: {e}")
#                 raise
        
#         return links
    
#     def _is_same_domain(self, url):
#         from urllib.parse import urlparse
#         base_domain = urlparse(self.base_url).netloc
#         target_domain = urlparse(url).netloc
#         return base_domain == target_domain or target_domain.endswith('.' + base_domain)


#     def _clean_url(self, url):
#         from urllib.parse import urlparse, urlunparse
#         parsed = urlparse(url)
#         # Remove fragments and sort query parameters for consistency
#         query = '&'.join(sorted(parsed.query.split('&')))
#         cleaned = urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', query, ''))
#         return cleaned.rstrip('/')



# # Example usage
# async def main():
#     crawler = AsyncCrawler("https://qanoon.om/")
#     visited = await crawler.extract_urls(output_file="crawled_urls.txt")
#     print(f"Crawled {len(visited)} pages")
    

# if __name__ == "__main__":
#     asyncio.run(main())


from src.chatbot.utils.paths import WEB_CONTENT_DIR
from src.chatbot.utils.utils import create_folder

from pathlib import Path

print(WEB_CONTENT_DIR / create_folder("test"))