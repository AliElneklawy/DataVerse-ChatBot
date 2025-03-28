import os
import logging
import requests
import uuid
from pathlib import Path
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler
from .utils.utils import create_folder
from .utils.paths import WEB_CONTENT_DIR
from urllib.parse import urljoin, urlparse
from typing import Set, List, Optional, Iterable, Dict
from .utils.crawler_progress import CrawlerProgress


logger = logging.getLogger(__name__)

class Crawler:
    def __init__(self, 
                 base_url, 
                 domain_name,
                 client: str = "crawl4ai",
                 output_folder: Optional[str] = None,):
        """
        Initialize the Crawler instance.

        Args:
            base_url (str): The starting URL to begin crawling from.
            domain_name (str): The domain name extracted from the base URL, used for naming output files.
            output_folder (Optional[str]): Directory to save the crawled content. Defaults to WEB_CONTENT_DIR if None.
            client (str): The crawling client to use ("crawl4ai" or "scrapegraph"). Defaults to "crawl4ai".
        """
        output_folder = Path(output_folder) \
                            if output_folder \
                            else create_folder(WEB_CONTENT_DIR)
        self.output_file = output_folder / f"{domain_name}.txt"
        self.client = client
        self.session = requests.Session()
        self.visited_urls: Set[str] = set()
        self.urls_to_visit: List[tuple[str, int]] = [(base_url, 0)]
        self.base_domain = urlparse(base_url).netloc
        self.crawl_id = str(uuid.uuid4())

        if self.client not in ["crawl4ai", "scrapegraph"]:
            logger.warning(f"Invalid client type: {client}. Defaulting to Crawl4AI.")
            self.client = "crawl4ai"

        if self.client == "crawl4ai":
            from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
            from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
            self.browser_config = BrowserConfig()
            self.run_config = CrawlerRunConfig(markdown_generator=DefaultMarkdownGenerator())
            self.crawler = None
        elif self.client == "scrapegraph":
            from scrapegraph_py import Client
            self.sgai_client = Client(api_key=os.getenv("SGAI_API"))
            
        # Initialize progress tracking
        CrawlerProgress.init_progress(self.crawl_id, base_url)

    async def extract_content(self, 
                              link: str, 
                              webpage_only: bool = False, 
                              max_depth: int = None) -> str | Path:
        """
        Scrape content from a URL or list of URLs with fallback between clients.

        Attempts to scrape using the primary client (specified in self.client). If it fails, falls back to the other client.
        Saves the extracted content in markdown format to a file.

        Args:
            link (str): The initial URL to scrape or start crawling from.
            webpage_only (bool): If True, scrape only the provided link; if False, crawl linked pages. Defaults to False.
            max_depth (int, optional): Maximum depth to crawl if webpage_only is False. Defaults to None (unlimited).

        Returns:
            str | Path: The path to the output file containing the scraped content.
        """
        # Update progress: initializing
        CrawlerProgress.update_progress(self.crawl_id, status='initializing', current_url=link)
        
        # if self.output_file.exists():
        #     return self.output_file
        
        url_list = [link] if webpage_only else await self._extract_urls(max_depth=max_depth)
        
        # Update progress with total URLs to crawl
        CrawlerProgress.update_progress(
            self.crawl_id, 
            status='crawling',
            total_urls=len(url_list),
            crawled_urls=0
        )
        
        with open(self.output_file, "w", encoding="utf-8") as file:
            if self.client == "crawl4ai":
                success_content_dict = await self._batch_crawl4ai(url_list)
                
                crawled_count = 0
                for url, (success, content) in success_content_dict.items():
                    # Update progress for current URL
                    crawled_count += 1
                    CrawlerProgress.update_progress(
                        self.crawl_id,
                        current_url=url,
                        crawled_urls=crawled_count,
                        log_message=f"Processing {url}"
                    )
                    
                    if success:
                        file.write(content)
                        file.write("\n\n")
                    else:
                        logger.debug(f"Falling back to scrapegraph for {url}")
                        CrawlerProgress.update_progress(
                            self.crawl_id,
                            log_message=f"Falling back to scrapegraph for {url}"
                        )
                        success, content = self._sgai_crawler_client(url)
                        if success:
                            file.write(content)
                            file.write("\n\n")
                        else:
                            logger.error(f"Both crawlers failed for {url}")
                            CrawlerProgress.update_progress(
                                self.crawl_id,
                                log_message=f"Both crawlers failed for {url}"
                            )
            else:  # scrapegraph
                crawled_count = 0
                for url in url_list:
                    # Update progress for current URL
                    crawled_count += 1
                    CrawlerProgress.update_progress(
                        self.crawl_id,
                        current_url=url,
                        crawled_urls=crawled_count,
                        log_message=f"Processing {url} with scrapegraph"
                    )
                    
                    success, content = self._sgai_crawler_client(url)
                    if not success:
                        logger.debug(f"Falling back to crawl4ai for {url}")
                        CrawlerProgress.update_progress(
                            self.crawl_id,
                            log_message=f"Falling back to crawl4ai for {url}"
                        )
                        success, content = await self._crawl4ai_crawler_client(url)
                    
                    if success:
                        file.write(content)
                        file.write("\n\n")
                    else:
                        logger.error(f"Both crawlers failed for {url}")
                        CrawlerProgress.update_progress(
                            self.crawl_id,
                            log_message=f"Both crawlers failed for {url}"
                        )

        # Mark crawling as complete
        CrawlerProgress.complete_progress(self.crawl_id, success=True)
        return self.output_file

    async def _batch_crawl4ai(self, urls: Iterable[str]) -> Dict[str, tuple[bool, str]]:
        """
        Batch process multiple URLs using Crawl4AI to take advantage of caching.

        Args:
            urls (Iterable[str]): List of URLs to process.

        Returns:
            Dict[str, tuple[bool, str]]: Dictionary mapping URLs to tuples of (success, content).
        """
        if self.crawler is None:
            self.crawler = AsyncWebCrawler(config=self.browser_config)
            await self.crawler.start()

        result_dict = {}
        session_id = "batch_session"

        try:
            url_list = list(urls)
            batch_size = 10
            
            for i in range(0, len(url_list), batch_size):
                batch_urls = url_list[i:i+batch_size]
                
                for url in batch_urls:
                    try:
                        CrawlerProgress.update_progress(
                            self.crawl_id,
                            log_message=f"Processing {url} with crawl4ai"
                        )
                        
                        result = await self.crawler.arun(
                            url=url,
                            config=self.run_config,
                            session_id=session_id
                        )
                        if result.success:
                            result_dict[url] = (True, result.markdown.raw_markdown)
                        else:
                            logger.error(f"Crawl4ai failed for {url}: {result.error_message}")
                            CrawlerProgress.update_progress(
                                self.crawl_id,
                                log_message=f"Crawl4ai failed for {url}: {result.error_message}"
                            )
                            result_dict[url] = (False, "")
                    except Exception as e:
                        logger.error(f"Crawl4ai failed for {url}: {e}")
                        CrawlerProgress.update_progress(
                            self.crawl_id,
                            log_message=f"Crawl4ai failed for {url}: {str(e)}"
                        )
                        result_dict[url] = (False, "")
            
            return result_dict
        except Exception as e:
            logger.error(f"Batch crawling failed: {e}")
            CrawlerProgress.update_progress(
                self.crawl_id,
                error=f"Batch crawling failed: {str(e)}"
            )
            return {url: (False, "") for url in urls}
        
    def _firecrawl_crawler_client(self, url: str) -> tuple[bool, str]:
        pass

    def _sgai_crawler_client(self, url: str) -> tuple[bool, str]:
        """
        Scrape a URL using the Scrapegraph client synchronously.

        Args:
            url (str): The URL to scrape.

        Returns:
            tuple[bool, str]: A tuple containing a success flag (True if successful, False otherwise) and the scraped markdown content.
        """
        try:
            response = self.sgai_client.markdownify(website_url=url)
            if response and 'result' in response:
                return True, response['result']
            
            logger.error(f"Scrapegraph failed: {url} - Error: couldn't parse url.")
            return False, ""
        except Exception as e:
            logger.error(f"Scrapegraph failed: {url} - Error: {e}")
            return False, ""

    async def _crawl4ai_crawler_client(self, url: str) -> tuple[bool, str]:
        """
        Scrape a URL using the Crawl4AI client asynchronously.

        Args:
            url (str): The URL to scrape.

        Returns:
            tuple[bool, str]: A tuple containing a success flag (True if successful, False otherwise) and the scraped markdown content.
        """
        if self.crawler is None:
            self.crawler = AsyncWebCrawler(config=self.browser_config)
            await self.crawler.start()

        try:
            session_id = "session"
            result = await self.crawler.arun(
                url=url,
                config=self.run_config,
                session_id=session_id
            )
            if result.success:
                return True, result.markdown.raw_markdown
            
            logger.error(f"Crawl4ai failed: {url} - Error: {result.error_message}")
            return False, ""
        except Exception as e:
            print(f"Crawl4ai failed: {url} - Error: {e}")
            return False, ""

    async def _extract_urls(self, max_pages: int = None, max_depth: int = None):
        """
        Extract URLs to crawl starting from the base URL, respecting depth and page limits.

        Args:
            max_pages (int, optional): Maximum number of pages to crawl. Defaults to None (unlimited).
            max_depth (int, optional): Maximum depth to crawl. Defaults to None (unlimited).

        Returns:
            Set[str]: A set of visited URLs that were crawled.
        """
        CrawlerProgress.update_progress(
            self.crawl_id, 
            status='discovering',
            log_message="Discovering URLs to crawl"
        )
        
        pages_crawled = 0

        while self.urls_to_visit and (max_pages is None or pages_crawled < max_pages):
            current_url, current_depth = self.urls_to_visit.pop(0)

            if current_url in self.visited_urls:
                continue
            if max_depth is not None and current_depth >= max_depth:
                continue

            logger.info(f"Crawling: {current_url} (depth: {current_depth + 1})")
            CrawlerProgress.update_progress(
                self.crawl_id,
                current_url=current_url,
                log_message=f"Discovering links at {current_url} (depth: {current_depth + 1})"
            )

            new_links = self._crawl_page(current_url, current_depth + 1)
            self.visited_urls.add(current_url)
            pages_crawled += 1

            for link, depth in new_links:
                if link not in self.visited_urls and (link, depth) not in self.urls_to_visit:
                    self.urls_to_visit.append((link, depth))
                    
            # Update progress with discovered URLs
            CrawlerProgress.update_progress(
                self.crawl_id,
                total_urls=len(self.visited_urls) + len(self.urls_to_visit),
                crawled_urls=0,
                log_message=f"Discovered {len(self.visited_urls) + len(self.urls_to_visit)} URLs so far"
            )

        logger.info(f"Crawling finished. Total pages crawled: {len(self.visited_urls)}")
        return self.visited_urls
    
    def _crawl_page(self, url, depth) -> List[tuple[str, int]]:
        """
        Crawl a single page to extract linked URLs within the same domain.

        Args:
            url (str): The URL of the page to crawl.
            depth (int): The current depth of the crawl.

        Returns:
            List[tuple[str, int]]: A list of tuples containing (cleaned URL, depth) for links found on the page.
        """
        # response = self.session.get(url, timeout=None)
        try:
            response = self.session.get(url, timeout=20)
            soup = BeautifulSoup(response.content, "html.parser")
        except Exception as e:
            return []

        links = []
        for link in soup.find_all("a", href=True):
            next_url = urljoin(url, link["href"])

            if not next_url.startswith(('http://', 'https://')):
                continue
            if not self._is_same_domain(next_url):
                continue

            _clean_url = self._clean_url(next_url)
            if _clean_url not in self.visited_urls:
                links.append((_clean_url, depth))

        return links

    def _is_same_domain(self, url: str) -> bool:
        """
        Check if a URL belongs to the same domain as the base URL.

        Args:
            url (str): The URL to check.

        Returns:
            bool: True if the URL is in the same domain, False otherwise.
        """
        try:
            return urlparse(url).netloc == self.base_domain
        except:
            return False
        
    def _clean_url(self, url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        
    # Get crawl ID
    def get_crawl_id(self) -> str:
        """Get the unique ID for this crawl session"""
        return self.crawl_id
        
    async def close(self):
        """
        Close any open connections and resources.
        """
        if self.client == "crawl4ai" and self.crawler is not None:
            await self.crawler.close()
        elif self.client == "scrapegraph":
            self.sgai_client.close()