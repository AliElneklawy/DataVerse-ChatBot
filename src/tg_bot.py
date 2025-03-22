import os
import re
import io
import logging
import asyncio
import tldextract
import dns.resolver
from telegram import Update
from enum import Enum, auto
from openai import AsyncOpenAI
from dotenv import load_dotenv
from chatbot.crawler import Crawler
from chatbot.config import get_api_key
from chatbot.utils.utils import DatabaseOps
from chatbot.rag.cohere_rag import CohereRAG
from chatbot.utils.file_loader import FileLoader
from concurrent.futures import ThreadPoolExecutor
from chatbot.utils.paths import INDEXES_DIR, WEB_CONTENT_DIR
from telegram.ext import (
    filters,
    CommandHandler,
    Application,
    MessageHandler,
    ContextTypes,
    ConversationHandler,
)

load_dotenv()
logger = logging.getLogger(__name__)

class ConversationState(Enum):
    QUESTION = auto()

class TelegramBot:
    EMAIL_REGEX = r"^[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$"
    ADMINS = [5859780703, 7298637025]
    
    def __init__(self, link):
        self.rag = None
        self.link = link
        self.application = Application.builder().token(os.getenv('BOT_TOKEN')).concurrent_updates(True).build()
        # daemon threads to ensure they exit when the main program exits
        # self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2, thread_name_prefix="rag_worker")  # Thread pool for handling RAG requests

        self.db = DatabaseOps()
        self.client = AsyncOpenAI(api_key=get_api_key("OPENAI"))
        # self.email_service = EmailService()
        # self.resp_monitor = ResponseMonitor(self.email_service)
        
        # signal handlers for graceful shutdown
        import signal
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle_exit)

        self._setup_handlers()

    def extract_domain_name(self, link):
        return tldextract.extract(link).domain

    async def fetch_content(self, link, domain_name, max_depth=None, file_path=None, webpage_only=True):
        if link:
            crawler = Crawler(link, domain_name, client="crawl4ai")
            content_path = await crawler.extract_content(link, webpage_only=webpage_only, max_depth=max_depth)

        if file_path:
            loader = FileLoader(file_path, content_path, client="docling")
            docs = loader.extract_from_file()
            if docs:
                logger.info(f"Successfully extracted {len(docs)} documents")
        
        return content_path

    async def _init_rag_system(self):
        # domain_name = self.extract_domain_name(self.link)
        # content_path = await self.fetch_content(self.link, domain_name, webpage_only=False)
        self.rag = CohereRAG(WEB_CONTENT_DIR / "bcaitech.txt", 
                             INDEXES_DIR, chunking_type="recursive")
        logger.info("RAG system initialized.")

    def _setup_handlers(self):
        self.question_handler = ConversationHandler(
            entry_points=[CommandHandler("start", self.start)],
            states={
                ConversationState.QUESTION: [
                    MessageHandler(filters.TEXT | filters.VOICE & ~filters.COMMAND, 
                                   self.handle_question),
                ],
            },
            fallbacks=[
                CommandHandler("start", self.start),
                CommandHandler("cancel", self.cancel_conversation),
            ],
            name="conversation_handler"
        )

        self.application.add_handler(CommandHandler("add_admin", self.add_admin))
        self.application.add_handler(CommandHandler("remove_admin", self.remove_admin))
        self.application.add_handler(CommandHandler("show_admins", self.get_admins))
        self.application.add_handler(CommandHandler("set_email", self.set_email))
        self.application.add_handler(CommandHandler("broadcast", self.broadcast))
        self.application.add_handler(CommandHandler("add_content", self.add_content))
        self.application.add_handler(self.question_handler)
        

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        user = update.effective_user
        user_id = user.id
        msg = "مرحبًا بك في بوت الدعم الخاص بشركة بيسايتك، وجهتك الأولى لعالم التقنية والذكاء الإصطناعي ودعم الأعمال.\n" \
              "كيف يمكنني مساعدتك اليوم؟"
        await update.message.reply_text(msg)

        if not self._user_exists(user_id):
            user_name = user.first_name
            self.db.append_bot_sub(user_id, user_name, 'Telegram')
            logger.info("Appended new telegram user.")

        return ConversationState.QUESTION

    async def transcribe(self, audio_buffer: io.BytesIO):
        transcription = await self.client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_buffer,
            response_format="text"
        )

        return transcription
    
    async def add_content(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        args: list = context.args
        new_content: str = "\n\n" + " ".join(args)
        msg = await update.message.reply_text("Please wait while I add the new content to my knowledge base...")
        self.rag._update_vectorstore(new_content)
        await msg.edit_text("I added your question successfully!")

    async def add_admin(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        current_user_id = update.effective_user.id
        if not context.args:
            await update.message.reply_text("Please provide a user ID")
            return

        user_id = int(context.args[0])
        if not self._is_admin(current_user_id):
            await update.message.reply_text("This command is only available for admins.")
            return
        
        if self._is_admin(user_id):
            await update.message.reply_text("User is already an admin.")
            return
        
        self.ADMINS.append(user_id)
        await update.message.reply_text("User is added as admin.")

    async def remove_admin(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        current_user_id = update.effective_user.id
        if not context.args:
            await update.message.reply_text("Please provide a user ID")
            return

        user_id = int(context.args[0])
        if not self._is_admin(current_user_id):
            await update.message.reply_text("This command is only available for admins.")
            return
        
        if not self._is_admin(user_id):
            await update.message.reply_text("User is not an admin.")
            return
        
        self.ADMINS.remove(user_id)
        await update.message.reply_text("User removed from admins.")

    async def get_admins(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        admins_str = "\n".join(str(admin) for admin in self.ADMINS) \
                                                        if self.ADMINS \
                                                        else "No admins set"
        await update.message.reply_text(f"Current admins:\n{admins_str}")

    def _is_admin(self, user_id) -> bool:
        return user_id in self.ADMINS

    def _user_exists(self, id):
        users = self.db.get_bot_sub(id)
        return True if users else False
    
    # runs the RAG query in a separate thread
    def _run_rag_query(self, question, user_id):
        try:
            return self.rag.get_response(question, user_id)
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return f"عذرًا، حدث خطأ أثناء معالجة سؤالك: {e}"

    async def _extract_question(self, msg, context: ContextTypes.DEFAULT_TYPE):
        if msg.text:
            return msg.text
        elif msg.voice:
            try:
                file = msg.voice
                file = await context.bot.get_file(file.file_id)
                file_bytes = await file.download_as_bytearray()
                with io.BytesIO(file_bytes) as audio_buffer:
                    audio_buffer.name = 'audio.ogg'
                    transcription = await self.transcribe(audio_buffer)
                    await msg.reply_text(f"سؤالك: {transcription}")
                    return transcription
            except Exception as e:
                logger.error(f"Failed to transcribe voice message: {e}")
                return None
        return None
    
    async def handle_question(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        user_id = str(update.effective_user.id)
        chat_id = update.effective_chat.id
        msg = update.message

        # if msg.text:
        #     question = update.message.text
        # elif msg.voice:
        #     file = msg.voice
        #     file = await context.bot.get_file(file.file_id)
        #     file_bytes = await file.download_as_bytearray()
        #     audio_buffer = io.BytesIO(file_bytes)
        #     audio_buffer.name = 'audio.ogg'
        #     question = await self.transcribe(audio_buffer)
        #     await update.message.reply_text(question)

        question = await self._extract_question(msg, context)
        if not question:
            await msg.reply_text("لم أتمكن من فهم سؤالك. من فضلك أعد المحاولة.")
            return ConversationState.QUESTION

        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        
        try:
            # Run the RAG query in a separate thread using the thread pool executor.
            # response = await asyncio.get_event_loop().run_in_executor(
            #     self.executor, 
            #     self._run_rag_query,
            #     question, 
            #     user_id
            # )
            response = await self.rag.get_response(question, user_id)
            await update.message.reply_text(response)
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            await update.message.reply_text("عذرًا، حدث خطأ أثناء معالجة سؤالك. حاول مرة أخرى لاحقًا.")

        return ConversationState.QUESTION

    async def cancel_conversation(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        await update.message.reply_text("تم إلغاء المحادثة. يمكنك البدء من جديد باستخدام /start.")
        return ConversationHandler.END

    async def set_email(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not self._is_admin(user_id):
            await update.message.reply_text("This command is only available for admins.")
            return
        
        if not context.args:
            self.rag.email_service.receiver_email = None
            await update.message.reply_text(f"Email unset. You won't receive updates about your bot's responses.")
            return

        new_email = context.args[0]
        if not self._is_valid_email(new_email):
            await update.message.reply_text("The email is invalid. Please double-check the email that you provided.")
            return
        
        self.rag.email_service.receiver_email = new_email
        await update.message.reply_text(f"Email set to {new_email}.")

    def _is_valid_email(self, new_email):
        if not re.fullmatch(self.EMAIL_REGEX, new_email):
            return False
        
        try:
            domain = new_email.split('@')[1]
            mx_records = dns.resolver.resolve(domain, 'MX')
            if mx_records:
                return True
            else:
                return False
        except:
            return False

    async def broadcast(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id

        if not self._is_admin(user_id):
            await update.message.reply_text("This command is only available for admins.")
            return
        
        subs = self.db.get_bot_sub()
        admin_msg = " ".join(context.args)

        try:
            for sub in subs:
                chat_id = sub[0]
                if chat_id != user_id: # don't send to yourself
                    await context.bot.send_message(chat_id=chat_id, text=admin_msg)
                    logger.info("Message broadcasted successfully.")
        except Exception as e:
            await context.bot.send_message(chat_id=user_id, text=f"Something went wrong ({chat_id}): {e}")
            logger.error(f"Something went wrong: {e}")
        else:
            await context.bot.send_message(chat_id=user_id, text="Messages broadcasted successfully.")

    async def run_async(self):
        """Run the bot asynchronously"""
        await self._init_rag_system()  # Initialize RAG system asynchronously
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        logger.info("========= Bot is running =========")

        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        finally:
            import signal
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            
            logger.info("Shutting down resources...")
            
            if self.rag and self.rag.resp_monitor._running:
                logger.info("Stopping response monitor...")
                self.rag.resp_monitor._stop_monitoring()
            
            logger.info("Stopping Telegram application...")
            await self.application.updater.stop()
            await self.application.stop()
            
            logger.info("Shutting down thread pool...")
            # self.executor.shutdown(wait=False)
            
            for task in asyncio.all_tasks():
                if task is not asyncio.current_task():
                    task.cancel()
            
            import sys
            logger.info("Exiting application...")
            sys.exit(0)

    def _handle_exit(self, signum, frame):
        """Handle exit signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        import threading
        threading.Timer(3.0, lambda: os._exit(1)).start()
        raise KeyboardInterrupt("Shutdown signal received")
    
    def run(self):
        """Run the bot using asyncio.run"""
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down")
            import sys
            sys.exit(0)

if __name__ == "__main__":
    bot = TelegramBot("https://bcaitech.bh/")
    bot.run()
