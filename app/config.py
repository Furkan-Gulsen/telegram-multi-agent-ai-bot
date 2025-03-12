import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Bot Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB_NAME = "telegram_bot"
MONGODB_COLLECTIONS = {
    "messages": "message_queue",
    "documents": "documents"
}

# Vector Search Configuration
VECTOR_DIMENSIONS = 1536  # OpenAI embedding dimensions
VECTOR_SIMILARITY = "cosine"
VECTOR_INDEX_NAME = "default"

# Message Processing Configuration
WAIT_TIME = 15  # seconds to wait for additional messages
MAX_MESSAGES_PER_BATCH = 10  # maximum number of messages to process in one batch

# Document Processing Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DOCUMENT_UPLOAD_PATH = "uploads"

# Language Model Configuration
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.7

# Create necessary directories
os.makedirs(DOCUMENT_UPLOAD_PATH, exist_ok=True)
