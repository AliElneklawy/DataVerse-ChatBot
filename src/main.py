import os
import asyncio
import tldextract
from uuid import uuid4
from chatbot.crawler import Crawler
from chatbot.voice_mode import VoiceMode
from chatbot.rag.grok_rag import GrokRAG
from chatbot.rag.cohere_rag import CohereRAG
from chatbot.rag.claude_rag import ClaudeRAG
from chatbot.rag.gemini_rag import GeminiRAG
from chatbot.rag.openai_rag import OpenAIRAG
from chatbot.rag.mistral_rag import MistralRAG
from chatbot.rag.deepseek_rag import DeepseekRAG
from chatbot.utils.file_loader import FileLoader
from chatbot.utils.paths import INDEXES_DIR, WEB_CONTENT_DIR

# =================================================================================
# OMP: Error #15: Initializing libomp140.x86_64.dll, but found libiomp5md.dll
# already initialized. This happened after integrating docling. Find and delete
# libiomp5md.dll!
# You should have used a venv!
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "FALSE" # THIS IS SO BAD! SET IT BACK TO FALSE!
# =================================================================================


def extract_domain_name(link):
    return tldextract.extract(link).domain


def fetch_content(
    link,
    domain_name,
    max_depth=None,
    file_path=None,
    webpage_only=True,
):
    if link:
        crawler = Crawler(link, domain_name, client="crawl4ai")  # or "scrapegraph"
        content_path = asyncio.run(
            crawler.extract_content(
                link, webpage_only=webpage_only, max_depth=max_depth
            )
        )

    if file_path:
        loader = FileLoader(file_path, content_path, client="docling")
        docs = loader.extract_from_file()
        if docs:
            print(f"Successfully extracted {len(docs)} documents")

    return content_path


async def run_text_mode(rag, user_id):
    while True:
        query = input("User: ").strip()
        if not query:
            continue
        if query.lower() == "quit":
            break

        try:
            response = await rag.get_response(query, user_id)
            print("Assistant:", response)
        except Exception as e:
            print(f"Error: {e}")


async def run_voice_mode(rag, oVoice: VoiceMode, user_id):
    while True:
        wav_path = oVoice.start_recording()
        query = oVoice.transcribe(wav_path)
        print(query)
        response = asyncio.run(rag.get_response(query, user_id))
        oVoice.text_to_speech(response)
        print("Assistant:", response)


def main():
    original_value = os.environ.get("KMP_DUPLICATE_LIB_OK")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    USER_ID = str(uuid4())
    link = "https://alielneklawy.wixsite.com/themlengineer"
    domain_name = extract_domain_name(link)
    content_path = fetch_content(link, domain_name, webpage_only=False)
    rag = CohereRAG(content_path, INDEXES_DIR, chunking_type="recursive", rerank=False)

    response_monitor = rag.resp_monitor
    hist_sender = rag.hist_sender

    try:
        mode = 1  # Assuming text mode for now
        if mode == 1:
            asyncio.run(run_text_mode(rag, USER_ID))
        else:
            oVoice = VoiceMode()
            asyncio.run(run_voice_mode(rag, oVoice, USER_ID))
    except (KeyboardInterrupt, EOFError):
        print("\nMain thread caught Ctrl+C, stopping background tasks...")
    finally:
        for monitor in [response_monitor, hist_sender]:
            if hasattr(monitor, "_running") and monitor._running:
                print(f"Stopping {monitor.__class__.__name__} thread...")
                monitor._stop_monitoring()
        if original_value is not None:
            os.environ["KMP_DUPLICATE_LIB_OK"] = original_value
        else:
            del os.environ["KMP_DUPLICATE_LIB_OK"]
        print("Program terminated cleanly.")


if __name__ == "__main__":
    main()
