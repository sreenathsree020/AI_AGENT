import os
import logging
import asyncio
from typing import Optional
import azure.cognitiveservices.speech as speechsdk
from openai import AsyncOpenAI

from config import Config

logger = logging.getLogger(__name__)


class VoiceAgent:
    def __init__(self):
        self.speech_key = Config.AZURE_SPEECH_KEY
        self.speech_region = Config.AZURE_SPEECH_REGION
        self.system_prompt = Config.SYSTEM_PROMPT

        # OpenRouter / OpenAI client initialization
        self.llm_client = None
        if Config.OPENROUTER_API_KEY and not Config.OPENROUTER_API_KEY.startswith("your_"):
            self.llm_client = AsyncOpenAI(
                base_url=Config.OPENROUTER_BASE_URL,
                api_key=Config.OPENROUTER_API_KEY,
                default_headers={
                    "HTTP-Referer": f"http://{Config.HOST}:{Config.PORT}",
                    "X-Title": "VoiceAgent-Exotel"
                }
            )
            logger.info(f"OpenRouter initialized with model: {Config.OPENROUTER_MODEL} at {Config.OPENROUTER_BASE_URL}")

        # Azure Speech configuration
        self.speech_config = None
        if self.speech_key and not self.speech_key.startswith("your_"):
            try:
                self.speech_config = speechsdk.SpeechConfig(
                    subscription=self.speech_key,
                    region=self.speech_region
                )
                self.speech_config.speech_recognition_language = Config.AZURE_STT_LANGUAGE
                self.speech_config.speech_synthesis_voice_name = Config.AZURE_TTS_VOICE
                logger.info(f"Azure Speech initialized: region={self.speech_region}, voice={Config.AZURE_TTS_VOICE}")
            except Exception as e:
                logger.error(f"Error configuring Azure Speech SDK: {e}")

    async def speech_to_text(self, audio_bytes: bytes) -> Optional[str]:
        """Convert audio (PCM, 16kHz, 16-bit, mono) to text using Azure STT."""
        if not self.speech_config:
            logger.warning("Azure Speech Config not initialized.")
            return None

        try:
            audio_stream = speechsdk.audio.PushAudioInputStream()
            audio_stream.write(audio_bytes)
            audio_stream.close()
            audio_config = speechsdk.audio.AudioConfig(stream=audio_stream)
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            result = await asyncio.to_thread(recognizer.recognize_once)
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return result.text
            elif result.reason == speechsdk.ResultReason.NoMatch:
                logger.debug("No speech recognized.")
                return None
            else:
                logger.warning(f"STT failed: {result.reason}")
                return None
        except Exception as e:
            logger.error(f"STT error: {e}")
            return None

    async def text_to_speech(self, text: str) -> bytes:
        """Convert text to audio (PCM, 16kHz, 16-bit, mono) using Azure TTS."""
        if not self.speech_config:
            logger.warning("Azure Speech Config not initialized.")
            return b""

        try:
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=None)
            result = await asyncio.to_thread(synthesizer.speak_text_async(text).get)
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                return result.audio_data
            else:
                logger.error(f"TTS failed: {result.reason}")
                return b""
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return b""

    async def generate_response(self, user_input: str, session_id: Optional[str] = None) -> str:
        """Generate AI response using OpenRouter / OpenAI."""
        if not self.llm_client:
            return "Thank you for your message. The LLM API key is not configured."

        try:
            response = await self.llm_client.chat.completions.create(
                model=Config.OPENROUTER_MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenRouter / LLM error: {e}", exc_info=True)
            return "I'm sorry, I'm having trouble processing your request. Please try again."

    async def generate_greeting(self) -> str:
        return "Hello! Thank you for calling our support line. How can I assist you today?"
