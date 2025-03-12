import os
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import BadRequest

from ..config import TELEGRAM_BOT_TOKEN, DOCUMENT_UPLOAD_PATH
from ..handlers.message import message_handler
from ..handlers.document import document_handler
from ..utils.language import detect_language
from ..database.mongodb import db

class TelegramBot:
    def __init__(self):
        self.app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        self.setup_handlers()
        self.processing_users = set()
        os.makedirs(DOCUMENT_UPLOAD_PATH, exist_ok=True)

    def setup_handlers(self):
        """Setup message and command handlers"""
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("docs", self.list_documents))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        self.app.add_handler(MessageHandler(filters.Document.ALL, self.handle_document))

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user_lang = detect_language(update.effective_user.language_code or 'en')
        if user_lang == 'tr':
            await update.message.reply_text(
                "Merhaba! Ben cok dilli bir AI asistanim. "
                "Benimle istediginiz dilde sohbet edebilirsiniz."
            )
        else:
            await update.message.reply_text(
                "Hello! I'm a multilingual AI assistant. "
                "You can chat with me in any language."
            )

    async def list_documents(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /docs command"""
        user_id = str(update.effective_user.id)
        docs = document_handler.get_user_documents(user_id)
        
        if not docs:
            await update.message.reply_text("You haven't uploaded any documents yet.")
            return
        
        response = "Your uploaded documents:\n\n"
        for doc in docs:
            status = "✅" if doc["status"] == "processed" else "❌"
            response += f"{status} {os.path.basename(doc['file_path'])} - {doc['upload_time'].strftime('%Y-%m-%d %H:%M')}\n"
        
        await update.message.reply_text(response)

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle document uploads"""
        user_id = str(update.effective_user.id)
        document = update.message.document
        
        # Download file
        file = await context.bot.get_file(document.file_id)
        file_path = os.path.join(DOCUMENT_UPLOAD_PATH, f"{user_id}_{document.file_name}")
        await file.download_to_drive(file_path)
        
        try:
            # Process document
            result = await document_handler.process_document(file_path, user_id)
            
            if result["status"] == "exists":
                await update.message.reply_text("This document has already been uploaded and processed.")
            else:
                await update.message.reply_text(
                    "Document uploaded and processed successfully! I will analyze it and provide a summary."
                )
                
                # Schedule message processing if not already processing for this user
                if user_id not in self.processing_users:
                    self.processing_users.add(user_id)
                    asyncio.create_task(self._process_messages(user_id))
                
        except Exception as e:
            await update.message.reply_text(f"Error processing document: {str(e)}")
        finally:
            # Clean up downloaded file
            if os.path.exists(file_path):
                os.remove(file_path)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages"""
        user_id = str(update.effective_user.id)
        message_text = update.message.text
        
        # Add message to queue using MongoDB instance directly
        db.add_message(user_id, message_text)
        
        # Schedule message processing if not already processing for this user
        if user_id not in self.processing_users:
            self.processing_users.add(user_id)
            asyncio.create_task(self._process_messages(user_id))

    async def _process_messages(self, user_id: str):
        """Process messages for a user"""
        try:
            response = await message_handler.process_message_queue(user_id)
            if response:
                # Split long messages
                await self._send_long_message(user_id, response)
        finally:
            self.processing_users.discard(user_id)
            
            # Check for more pending messages
            pending_messages = message_handler.db.message_queue.count_documents({
                "user_id": user_id,
                "is_processed": False
            })
            
            if pending_messages > 0:
                self.processing_users.add(user_id)
                asyncio.create_task(self._process_messages(user_id))

    async def _send_long_message(self, chat_id: str, text: str, max_length: int = 4000):
        """Split and send long messages"""
        # If message is short enough, send it directly
        if len(text) <= max_length:
            await self.app.bot.send_message(chat_id=chat_id, text=text)
            return

        # Split message into parts
        parts = []
        current_part = ""
        
        # Split by paragraphs first
        paragraphs = text.split("\n\n")
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed max_length
            if len(current_part) + len(paragraph) + 2 > max_length:
                # If current_part is not empty, add it to parts
                if current_part:
                    parts.append(current_part.strip())
                current_part = paragraph
            else:
                # Add paragraph to current_part
                if current_part:
                    current_part += "\n\n"
                current_part += paragraph
        
        # Add the last part if not empty
        if current_part:
            parts.append(current_part.strip())
        
        # Send each part with a part number
        total_parts = len(parts)
        for i, part in enumerate(parts, 1):
            if total_parts > 1:
                header = f"Part {i}/{total_parts}:\n\n"
                message = header + part
            else:
                message = part
            
            try:
                await self.app.bot.send_message(chat_id=chat_id, text=message)
                # Add a small delay between messages to maintain order
                if i < total_parts:
                    await asyncio.sleep(0.5)
            except Exception as e:
                print(f"Error sending message part {i}: {str(e)}")
                # If a part is still too long, split it further
                if isinstance(e, BadRequest) and "Message is too long" in str(e):
                    # Split into smaller parts
                    subparts = [message[i:i + max_length] for i in range(0, len(message), max_length)]
                    for subpart in subparts:
                        await self.app.bot.send_message(chat_id=chat_id, text=subpart)
                        await asyncio.sleep(0.5)

    async def start(self):
        """Start the bot"""
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()

    async def stop(self):
        """Stop the bot"""
        await self.app.stop()
        await self.app.shutdown()

# Create a singleton instance
bot = TelegramBot()
