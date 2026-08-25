import os
import logging
import asyncio
import time
from typing import Optional, List, Dict
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

        # Azure Speech configurations
        self.speech_config_mulaw = None
        self.speech_config_pcm = None

        if self.speech_key and not self.speech_key.startswith("your_"):
            try:
                # 1. Telephony Config: 8kHz μ-law (PCMU)
                self.speech_config_mulaw = speechsdk.SpeechConfig(
                    subscription=self.speech_key,
                    region=self.speech_region
                )
                self.speech_config_mulaw.speech_recognition_language = Config.AZURE_STT_LANGUAGE
                self.speech_config_mulaw.speech_synthesis_voice_name = Config.AZURE_TTS_VOICE
                self.speech_config_mulaw.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Raw8Khz8BitMonoMULaw
                )

                # 2. Browser Config: 16kHz PCM
                self.speech_config_pcm = speechsdk.SpeechConfig(
                    subscription=self.speech_key,
                    region=self.speech_region
                )
                self.speech_config_pcm.speech_recognition_language = Config.AZURE_STT_LANGUAGE
                self.speech_config_pcm.speech_synthesis_voice_name = Config.AZURE_TTS_VOICE
                self.speech_config_pcm.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm
                )

                logger.info(f"Azure Speech initialized: region={self.speech_region}, voice={Config.AZURE_TTS_VOICE}")
            except Exception as e:
                logger.error(f"Error configuring Azure Speech SDK: {e}")

    async def speech_to_text(self, audio_bytes: bytes, is_mulaw: bool = True) -> Optional[str]:
        """
        Convert audio to text using Azure STT.
        If is_mulaw=True: audio is 8kHz 8-bit μ-law (Exotel telephony).
        If is_mulaw=False: audio is 16kHz 16-bit PCM (Browser).
        """
        config = self.speech_config_mulaw if is_mulaw else self.speech_config_pcm
        if not config:
            logger.warning("[STT] Azure Speech Config not initialized.")
            return None

        t0 = time.time()
        logger.info(f"[STT] Processing audio buffer ({len(audio_bytes)} bytes, format={'MULAW_8K' if is_mulaw else 'PCM_16K'})...")
        try:
            if is_mulaw:
                stream_format = speechsdk.audio.AudioStreamFormat(
                    samples_per_second=8000,
                    bits_per_sample=8,
                    channels=1,
                    wave_stream_format=speechsdk.AudioStreamWaveFormat.MULAW
                )
            else:
                stream_format = speechsdk.audio.AudioStreamFormat(
                    samples_per_second=16000,
                    bits_per_sample=16,
                    channels=1,
                    wave_stream_format=speechsdk.AudioStreamWaveFormat.PCM
                )

            audio_stream = speechsdk.audio.PushAudioInputStream(stream_format=stream_format)
            audio_stream.write(audio_bytes)
            audio_stream.close()

            audio_config = speechsdk.audio.AudioConfig(stream=audio_stream)
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=config,
                audio_config=audio_config
            )
            result = await asyncio.to_thread(recognizer.recognize_once)
            elapsed = (time.time() - t0) * 1000

            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                logger.info(f"[STT] Recognized text ({elapsed:.0f}ms): \"{result.text}\"")
                return result.text
            elif result.reason == speechsdk.ResultReason.NoMatch:
                logger.debug(f"[STT] No speech recognized ({elapsed:.0f}ms).")
                return None
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = speechsdk.CancellationDetails(result)
                logger.warning(f"[STT] Canceled: reason={cancellation_details.reason}, error_details={cancellation_details.error_details}")
                return None
            else:
                logger.warning(f"[STT] Recognition failed ({elapsed:.0f}ms): {result.reason}")
                return None
        except Exception as e:
            logger.error(f"[STT] Error during recognition: {e}", exc_info=True)
            return None

    async def text_to_speech(self, text: str, format_type: str = "mulaw") -> bytes:
        """
        Convert text to audio using Azure TTS.
        format_type='mulaw': Raw 8kHz 8-bit μ-law for Exotel telephony.
        format_type='pcm': Raw 16kHz 16-bit PCM for Browser testing.
        """
        config = self.speech_config_mulaw if format_type == "mulaw" else self.speech_config_pcm
        if not config:
            logger.warning("[TTS] Azure Speech Config not initialized.")
            return b""

        t0 = time.time()
        logger.info(f"[TTS] Synthesizing ({format_type}): \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
        try:
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=config, audio_config=None)
            result = await asyncio.to_thread(synthesizer.speak_text_async(text).get)
            elapsed = (time.time() - t0) * 1000

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.info(f"[TTS] Synthesized {len(result.audio_data)} bytes {format_type} ({elapsed:.0f}ms)")
                return result.audio_data
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = speechsdk.CancellationDetails(result)
                logger.error(f"[TTS] Canceled: reason={cancellation_details.reason}, error_details={cancellation_details.error_details}")
                return b""
            else:
                logger.error(f"[TTS] Synthesis failed ({elapsed:.0f}ms): {result.reason}")
                return b""
        except Exception as e:
            logger.error(f"[TTS] Error during synthesis: {e}", exc_info=True)
            return b""

    async def generate_response(
        self,
        user_input: str,
        session_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate AI response using OpenRouter / OpenAI with multi-turn conversation context."""
        if not self.llm_client:
            logger.warning("[LLM] OpenRouter client not initialized.")
            return "Thank you for calling. Our automated assistant is currently unavailable."

        t0 = time.time()
        messages = [{"role": "system", "content": self.system_prompt}]

        if conversation_history:
            for turn in conversation_history[-6:]:
                if turn.get("customer"):
                    messages.append({"role": "user", "content": turn["customer"]})
                if turn.get("agent"):
                    messages.append({"role": "assistant", "content": turn["agent"]})

        messages.append({"role": "user", "content": user_input})
        logger.info(f"[LLM] Calling model {Config.OPENROUTER_MODEL} (messages={len(messages)})...")

        try:
            response = await self.llm_client.chat.completions.create(
                model=Config.OPENROUTER_MODEL,
                messages=messages,
                max_tokens=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE
            )
            raw_reply = response.choices[0].message.content.strip()
            # Strip internal chain-of-thought (<think>...</think>) tags from reasoning models
            import re
            reply = re.sub(r"<think>.*?</think>", "", raw_reply, flags=re.DOTALL).strip()
            if not reply:
                reply = raw_reply

            elapsed = (time.time() - t0) * 1000
            logger.info(f"[LLM] Response ({elapsed:.0f}ms): \"{reply[:100]}{'...' if len(reply) > 100 else ''}\"")
            return reply
        except Exception as e:
            logger.error(f"[LLM] OpenRouter error: {e}", exc_info=True)
            return "I apologize, I'm having a brief issue retrieving that information. How else may I assist you?"

    async def generate_greeting(self) -> str:
        return "Hello! Thank you for calling our AI support line. How can I assist you today?"
