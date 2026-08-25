import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class Config:
    # Azure Speech
    AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
    AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "eastus")
    AZURE_STT_LANGUAGE = os.getenv("AZURE_STT_LANGUAGE", "en-US")
    AZURE_TTS_VOICE = os.getenv("AZURE_TTS_VOICE", "en-US-JennyNeural")

    # LLM (OpenRouter / OpenAI)
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    _raw_endpoint = os.getenv("OPENROUTER_ENDPOINT_URL", "https://openrouter.ai/api/v1")
    # Normalize base URL if endpoint contains /chat/completions
    OPENROUTER_BASE_URL = _raw_endpoint.replace("/chat/completions", "").rstrip("/")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL") or os.getenv("OPENAI_MODEL", "openrouter/free")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 150))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))

    # Redis / Valkey
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", 3600))

    # Exotel
    EXOTEL_ACCOUNT_SID = os.getenv("EXOTEL_ACCOUNT_SID", "")
    EXOTEL_API_KEY = os.getenv("EXOTEL_API_KEY", "")
    EXOTEL_API_TOKEN = os.getenv("EXOTEL_API_TOKEN", "")
    EXOTEL_SUBDOMAIN = os.getenv("EXOTEL_SUBDOMAIN", "")
    EXOTEL_PHONE_NUMBER = os.getenv("EXOTEL_PHONE_NUMBER", "")
    EXOTEL_USE_STREAM = os.getenv("EXOTEL_USE_STREAM", "false").lower() == "true"

    # Server
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8080))
    KOYEB_APP_URL = os.getenv("KOYEB_APP_URL", "localhost:8080")

    # System Prompt
    SYSTEM_PROMPT = os.getenv(
        "SYSTEM_PROMPT",
        """You are a helpful and friendly customer support agent.
Your goal is to assist customers with their questions and resolve their issues.
Be concise, professional, and empathetic. If you don't know the answer,
offer to connect them with a human agent."""
    )

    @classmethod
    def validate(cls):
        required = ["AZURE_SPEECH_KEY", "OPENROUTER_API_KEY"]
        missing = [k for k in required if not getattr(cls, k) or getattr(cls, k).startswith("your_")]
        if missing:
            logger.warning(f"Note: Some environment variables are not configured or use placeholders: {missing}")
        return True
