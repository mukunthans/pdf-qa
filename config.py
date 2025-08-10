import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# App Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 3

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FILE = "logs/pdf_extraction.log"