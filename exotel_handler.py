import os
import json
import base64
import logging
import audioop
import asyncio
import time
from typing import Optional
from voice_agent import VoiceAgent
from session_manager import SessionManager
from config import Config

logger = logging.getLogger(__name__)


class ExotelHandler:
    def __init__(self, voice_agent: VoiceAgent, session_manager: SessionManager):
        self.voice_agent = voice_agent
        self.session_manager = session_manager

    def _get_host(self) -> str:
        url = Config.KOYEB_APP_URL or "localhost:8000"
        return url.replace("https://", "").replace("http://", "").strip("/")

    # ---------- Gather Method (HTTP Webhook / Passthru) ----------
    def incoming_call_gather(self, call_sid: str) -> str:
        """Return Exotel ExoML for initial greeting and speech gather."""
        koyeb_url = self._get_host()
        greeting = "Hello! Welcome to our AI support. How can I help you today?"
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Response>\n'
            f'    <Say>{greeting}</Say>\n'
            f'    <Gather action="https://{koyeb_url}/api/exotel/gather-response?call_sid={call_sid}" method="POST" input="speech" timeout="5">\n'
            '    </Gather>\n'
            f'    <Redirect>https://{koyeb_url}/api/exotel/incoming?call_sid={call_sid}</Redirect>\n'
            '</Response>'
        )

    async def gather_response(self, call_sid: str, speech_result: str) -> str:
        """Process gathered speech from Exotel and return next ExoML response."""
        koyeb_url = self._get_host()
        session = self.session_manager.get_session(call_sid) or {}
        history = session.get("conversation", [])

        if speech_result and speech_result.strip():
            logger.info(f"[GATHER] Call {call_sid} - Customer speech: \"{speech_result}\"")
            ai_text = await self.voice_agent.generate_response(speech_result, call_sid, history)
            self.session_manager.add_conversation_turn(call_sid, speech_result, ai_text)
            say_text = ai_text
        else:
            say_text = "I didn't quite catch that. Could you please repeat your question?"

        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Response>\n'
            f'    <Say>{say_text}</Say>\n'
            f'    <Gather action="https://{koyeb_url}/api/exotel/gather-response?call_sid={call_sid}" method="POST" input="speech" timeout="5">\n'
            '    </Gather>\n'
            f'    <Redirect>https://{koyeb_url}/api/exotel/incoming?call_sid={call_sid}</Redirect>\n'
            '</Response>'
        )

    # ---------- Media Streams Method (WebSocket Real-Time Voice) ----------
    def incoming_call_stream(self, call_sid: str) -> str:
        """Return Exotel ExoML that connects to a WebSocket media stream."""
        koyeb_url = self._get_host()
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Response>\n'
            f'    <Stream url="wss://{koyeb_url}/ws/exotel-stream?callSid={call_sid}" />\n'
            '</Response>'
        )

    async def send_audio_chunks(self, websocket, stream_sid: str, pcm_16k_audio: bytes):
        """Convert 16kHz PCM audio to 8kHz μ-law and stream in 20ms chunks to Exotel."""
        if not pcm_16k_audio:
            return

        try:
            # 1. Resample 16kHz PCM -> 8kHz PCM
            pcm_8k = self._resample_16k_to_8k(pcm_16k_audio)
            # 2. Convert 8kHz PCM -> μ-law
            mulaw_audio = audioop.lin2ulaw(pcm_8k, 2)

            # 3. Stream in 20ms chunks (160 bytes per 20ms at 8000Hz 8-bit mono)
            chunk_size = 160
            total_chunks = (len(mulaw_audio) + chunk_size - 1) // chunk_size
            logger.info(f"[MEDIA_OUT] Streaming {len(mulaw_audio)} bytes audio ({total_chunks} chunks of 20ms) to stream {stream_sid}")

            for i in range(0, len(mulaw_audio), chunk_size):
                chunk = mulaw_audio[i:i + chunk_size]
                payload = base64.b64encode(chunk).decode("utf-8")
                msg = json.dumps({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {
                        "payload": payload
                    }
                })
                await websocket.send_text(msg)
                # Pacing: ~18ms sleep to keep playback buffer smooth
                await asyncio.sleep(0.018)

            logger.info(f"[MEDIA_OUT] Completed streaming audio response to stream {stream_sid}")
        except Exception as e:
            logger.error(f"[MEDIA_OUT] Error sending audio chunks: {e}", exc_info=True)

    async def process_media_stream(self, websocket, call_sid: str):
        """Handle full bi-directional real-time media stream with Exotel VoiceBot."""
        await websocket.accept()
        logger.info(f"============================================================")
        logger.info(f"[WS] WebSocket connected for call SID: {call_sid}")
        logger.info(f"============================================================")

        stream_sid = None
        audio_buffer = b""
        speech_started = False
        silence_frames = 0
        SILENCE_THRESHOLD = 40  # ~800ms of silence at 20ms frames
        ENERGY_THRESHOLD = 450   # RMS energy threshold for speech activity
        greeting_sent = False

        try:
            while True:
                message = await websocket.receive_text()
                data = json.loads(message)
                event = data.get("event")

                if event == "connected":
                    protocol = data.get("protocol", "Call")
                    logger.info(f"[WS] Exotel media stream connected (protocol={protocol})")

                elif event == "start":
                    start_data = data.get("start", {})
                    stream_sid = data.get("streamSid") or start_data.get("streamSid") or "unknown_stream"
                    extracted_call_sid = data.get("callSid") or start_data.get("callSid")
                    if extracted_call_sid and extracted_call_sid != "unknown":
                        call_sid = extracted_call_sid

                    logger.info(f"[WS] Stream started: streamSid={stream_sid}, callSid={call_sid}")
                    if not self.session_manager.get_session(call_sid):
                        self.session_manager.create_session(call_sid)

                    # Send initial AI voice greeting upon connection
                    if not greeting_sent:
                        greeting_sent = True
                        greeting_text = await self.voice_agent.generate_greeting()
                        logger.info(f"[AI_GREETING] Synthesizing initial greeting: \"{greeting_text}\"")
                        audio_greeting = await self.voice_agent.text_to_speech(greeting_text)
                        if audio_greeting:
                            asyncio.create_task(self.send_audio_chunks(websocket, stream_sid, audio_greeting))

                elif event == "media":
                    media_obj = data.get("media", {})
                    payload = media_obj.get("payload", "")
                    if not payload:
                        continue

                    if not stream_sid:
                        stream_sid = data.get("streamSid")

                    # Decode Exotel μ-law audio
                    try:
                        mulaw_chunk = base64.b64decode(payload)
                        pcm_8k = audioop.ulaw2lin(mulaw_chunk, 2)
                        pcm_16k = self._resample_8k_to_16k(pcm_8k)
                    except Exception as dec_err:
                        logger.debug(f"[AUDIO] Decode error: {dec_err}")
                        continue

                    # Calculate energy for Voice Activity Detection
                    energy = self._audio_energy(pcm_16k)

                    if energy > ENERGY_THRESHOLD:
                        if not speech_started:
                            speech_started = True
                            audio_buffer = pcm_16k
                            logger.info(f"[VAD] Caller started speaking (energy={energy:.0f})...")
                        else:
                            audio_buffer += pcm_16k
                        silence_frames = 0
                    else:
                        if speech_started:
                            audio_buffer += pcm_16k
                            silence_frames += 1

                            if silence_frames >= SILENCE_THRESHOLD:
                                # End of caller utterance detected
                                speech_started = False
                                silence_frames = 0
                                logger.info(f"[VAD] Speech ended. Total buffer: {len(audio_buffer)} bytes. Processing...")

                                if len(audio_buffer) > 3200:  # at least ~100ms
                                    utterance_audio = audio_buffer
                                    audio_buffer = b""

                                    # Run STT -> LLM -> TTS pipeline
                                    asyncio.create_task(self._handle_user_utterance(
                                        websocket, stream_sid, call_sid, utterance_audio
                                    ))
                                else:
                                    audio_buffer = b""

                elif event == "mark":
                    mark_name = data.get("mark", {}).get("name")
                    logger.debug(f"[WS] Received mark: {mark_name}")

                elif event == "dtmf":
                    digit = data.get("dtmf", {}).get("digit")
                    logger.info(f"[WS] DTMF received: {digit}")

                elif event == "stop":
                    logger.info(f"[WS] Exotel stream stopped: {stream_sid}")
                    break

        except Exception as e:
            logger.error(f"[WS] Error in media stream session: {e}", exc_info=True)
        finally:
            logger.info(f"[WS] Closing session for call: {call_sid}")
            self.session_manager.end_session(call_sid)
            try:
                await websocket.close()
            except Exception:
                pass

    async def _handle_user_utterance(self, websocket, stream_sid: str, call_sid: str, audio_bytes: bytes):
        """Asynchronous processing of recognized speech to AI response and playback."""
        try:
            # 1. Speech-to-Text
            text = await self.voice_agent.speech_to_text(audio_bytes)
            if not text or not text.strip():
                logger.info("[STT] No clear words recognized.")
                return

            logger.info(f"🗣️ [CALLER]: \"{text}\"")

            # 2. Get history & Generate LLM response
            session = self.session_manager.get_session(call_sid) or {}
            history = session.get("conversation", [])
            ai_text = await self.voice_agent.generate_response(text, call_sid, history)
            self.session_manager.add_conversation_turn(call_sid, text, ai_text)
            logger.info(f"🤖 [AI AGENT]: \"{ai_text}\"")

            # 3. Text-to-Speech
            audio_resp = await self.voice_agent.text_to_speech(ai_text)
            if audio_resp and stream_sid:
                await self.send_audio_chunks(websocket, stream_sid, audio_resp)
        except Exception as e:
            logger.error(f"[PIPELINE] Error handling utterance: {e}", exc_info=True)

    def _resample_8k_to_16k(self, pcm_8k: bytes) -> bytes:
        """Resample PCM from 8kHz to 16kHz."""
        try:
            pcm_16k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)
            return pcm_16k
        except Exception:
            import numpy as np
            arr_8k = np.frombuffer(pcm_8k, dtype=np.int16)
            duration = len(arr_8k) / 8000
            new_len = int(duration * 16000)
            x_old = np.linspace(0, duration, len(arr_8k))
            x_new = np.linspace(0, duration, new_len)
            arr_16k = np.interp(x_new, x_old, arr_8k).astype(np.int16)
            return arr_16k.tobytes()

    def _resample_16k_to_8k(self, pcm_16k: bytes) -> bytes:
        """Resample PCM from 16kHz to 8kHz."""
        try:
            pcm_8k, _ = audioop.ratecv(pcm_16k, 2, 1, 16000, 8000, None)
            return pcm_8k
        except Exception:
            import numpy as np
            arr_16k = np.frombuffer(pcm_16k, dtype=np.int16)
            duration = len(arr_16k) / 16000
            new_len = int(duration * 8000)
            x_old = np.linspace(0, duration, len(arr_16k))
            x_new = np.linspace(0, duration, new_len)
            arr_8k = np.interp(x_new, x_old, arr_16k).astype(np.int16)
            return arr_8k.tobytes()

    def _audio_energy(self, pcm_bytes: bytes) -> float:
        try:
            return float(audioop.rms(pcm_bytes, 2))
        except Exception:
            import numpy as np
            arr = np.frombuffer(pcm_bytes, dtype=np.int16)
            if len(arr) == 0:
                return 0.0
            return float(np.sqrt(np.mean(arr.astype(np.float64)**2)))