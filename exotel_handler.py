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
        self._is_speaking = False

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

    async def send_audio_chunks(self, websocket, stream_sid: str, mulaw_audio: bytes):
        """Stream native 8kHz μ-law audio chunks directly to Exotel."""
        if not mulaw_audio:
            return

        self._is_speaking = True
        try:
            chunk_size = 320  # 320 bytes = 40ms at 8000Hz 8-bit mono
            total_chunks = (len(mulaw_audio) + chunk_size - 1) // chunk_size
            logger.info(f"[MEDIA_OUT] Streaming {len(mulaw_audio)} bytes native μ-law ({total_chunks} chunks) to stream: {stream_sid}")

            for i in range(0, len(mulaw_audio), chunk_size):
                chunk = mulaw_audio[i:i + chunk_size]
                payload = base64.b64encode(chunk).decode("utf-8")
                msg = {
                    "event": "media",
                    "streamSid": stream_sid or "",
                    "media": {
                        "payload": payload
                    }
                }
                await websocket.send_text(json.dumps(msg))
                await asyncio.sleep(0.038)  # 40ms pacing

            logger.info(f"[MEDIA_OUT] Finished playing audio to caller (stream: {stream_sid})")
        except Exception as e:
            logger.error(f"[MEDIA_OUT] Error sending audio chunks: {e}", exc_info=True)
        finally:
            await asyncio.sleep(0.2)
            self._is_speaking = False

    async def process_media_stream(self, websocket, call_sid: str):
        """Handle full bi-directional real-time media stream with Exotel VoiceBot."""
        await websocket.accept()
        logger.info(f"============================================================")
        logger.info(f"[WS] WebSocket connected for call SID: {call_sid}")
        logger.info(f"============================================================")

        stream_sid = None
        mulaw_audio_buffer = b""
        speech_started = False
        silence_frames = 0
        SILENCE_THRESHOLD = 25   # ~500ms of silence at 20ms frames
        ENERGY_THRESHOLD = 250    # Sensitive speech activity threshold
        greeting_sent = False
        self._is_speaking = False

        async def send_greeting_if_needed():
            nonlocal greeting_sent, stream_sid
            if not greeting_sent:
                greeting_sent = True
                greeting_text = await self.voice_agent.generate_greeting()
                logger.info(f"[AI_GREETING] Synthesizing greeting: \"{greeting_text}\"")
                audio_greeting = await self.voice_agent.text_to_speech(greeting_text, format_type="mulaw")
                if audio_greeting:
                    asyncio.create_task(self.send_audio_chunks(websocket, stream_sid or "", audio_greeting))

        try:
            while True:
                message = await websocket.receive_text()
                data = json.loads(message)
                event = data.get("event")

                # Extract streamSid and callSid from any event
                incoming_sid = data.get("streamSid") or data.get("stream_sid") or data.get("start", {}).get("streamSid")
                if incoming_sid:
                    stream_sid = incoming_sid

                incoming_call_sid = data.get("callSid") or data.get("start", {}).get("callSid")
                if incoming_call_sid and incoming_call_sid != "unknown":
                    call_sid = incoming_call_sid

                if event == "connected":
                    protocol = data.get("protocol", "Call")
                    logger.info(f"[WS] Exotel connected (protocol={protocol}) | {data}")
                    # Trigger greeting immediately on connection
                    await send_greeting_if_needed()

                elif event == "start":
                    logger.info(f"[WS] Stream started: streamSid={stream_sid}, callSid={call_sid} | {data}")
                    if not self.session_manager.get_session(call_sid):
                        self.session_manager.create_session(call_sid)
                    await send_greeting_if_needed()

                elif event == "media":
                    media_obj = data.get("media", {})
                    payload = media_obj.get("payload", "")
                    if not payload:
                        continue

                    # Trigger greeting if not yet sent
                    if not greeting_sent:
                        await send_greeting_if_needed()

                    # If bot is currently speaking, discard echo
                    if self._is_speaking:
                        mulaw_audio_buffer = b""
                        speech_started = False
                        silence_frames = 0
                        continue

                    # Decode base64 μ-law chunk
                    try:
                        mulaw_chunk = base64.b64decode(payload)
                        pcm_chunk = audioop.ulaw2lin(mulaw_chunk, 2)
                    except Exception:
                        continue

                    # Energy calculation for Voice Activity Detection
                    energy = self._audio_energy(pcm_chunk)

                    if energy > ENERGY_THRESHOLD:
                        if not speech_started:
                            speech_started = True
                            mulaw_audio_buffer = mulaw_chunk
                            logger.info(f"[VAD] Caller speaking (energy={energy:.0f})...")
                        else:
                            mulaw_audio_buffer += mulaw_chunk
                        silence_frames = 0
                    else:
                        if speech_started:
                            mulaw_audio_buffer += mulaw_chunk
                            silence_frames += 1

                            if silence_frames >= SILENCE_THRESHOLD:
                                speech_started = False
                                silence_frames = 0
                                logger.info(f"[VAD] Speech ended. Buffer: {len(mulaw_audio_buffer)} bytes. Processing STT...")

                                if len(mulaw_audio_buffer) >= 1600:  # at least ~200ms of speech
                                    utterance = mulaw_audio_buffer
                                    mulaw_audio_buffer = b""
                                    asyncio.create_task(self._handle_user_utterance(
                                        websocket, stream_sid or "", call_sid, utterance
                                    ))
                                else:
                                    mulaw_audio_buffer = b""

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
            self._is_speaking = False
            try:
                await websocket.close()
            except Exception:
                pass

    async def _handle_user_utterance(self, websocket, stream_sid: str, call_sid: str, mulaw_bytes: bytes):
        """Process recognized speech to AI response and play back in native 8kHz μ-law."""
        try:
            # 1. Native 8kHz μ-law STT
            text = await self.voice_agent.speech_to_text(mulaw_bytes, is_mulaw=True)
            if not text or not text.strip():
                logger.info("[STT] No clear words recognized.")
                return

            logger.info(f"🗣️ [CALLER]: \"{text}\"")

            # 2. LLM response with session history
            session = self.session_manager.get_session(call_sid) or {}
            history = session.get("conversation", [])
            ai_text = await self.voice_agent.generate_response(text, call_sid, history)
            self.session_manager.add_conversation_turn(call_sid, text, ai_text)
            logger.info(f"🤖 [AI AGENT]: \"{ai_text}\"")

            # 3. Native 8kHz μ-law TTS
            audio_resp = await self.voice_agent.text_to_speech(ai_text, format_type="mulaw")
            if audio_resp:
                await self.send_audio_chunks(websocket, stream_sid, audio_resp)
        except Exception as e:
            logger.error(f"[PIPELINE] Error handling utterance: {e}", exc_info=True)

    def _audio_energy(self, pcm_bytes: bytes) -> float:
        try:
            return float(audioop.rms(pcm_bytes, 2))
        except Exception:
            import numpy as np
            arr = np.frombuffer(pcm_bytes, dtype=np.int16)
            if len(arr) == 0:
                return 0.0
            return float(np.sqrt(np.mean(arr.astype(np.float64)**2)))