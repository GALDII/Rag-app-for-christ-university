import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

def get_cohere_api_key():
    """Returns the Cohere API key from environment variables."""
    return os.getenv("COHERE_API_KEY")

def get_groq_api_key():
    """Returns the Groq API key from environment variables."""
    return os.getenv("GROQ_API_KEY")

def get_pinecone_api_key():
    """Returns the Pinecone API key from environment variables."""
    return os.getenv("PINECONE_API_KEY")

# --- App Configuration ---
PDF_PATH = "student_handbook.pdf"
PINECONE_INDEX_NAME = "student-handbook"
