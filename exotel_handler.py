import os
import json
import base64
import logging
import audioop
import asyncio
from typing import Optional
from voice_agent import VoiceAgent
from session_manager import SessionManager
from config import Config

logger = logging.getLogger(__name__)


class ExotelHandler:
    def __init__(self, voice_agent: VoiceAgent, session_manager: SessionManager):
        self.voice_agent = voice_agent
        self.session_manager = session_manager

    # ---------- Gather Method (simpler, no WebSocket) ----------
    def incoming_call_gather(self, call_sid: str) -> str:
        """Return Exotel XML for initial greeting and gather."""
        koyeb_url = Config.KOYEB_APP_URL
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Response>\n'
            '    <Say>Hello! Welcome to AI support. How can I help you?</Say>\n'
            f'    <Gather action="https://{koyeb_url}/api/exotel/gather-response?call_sid={call_sid}" method="POST" input="speech" timeout="5">\n'
            '    </Gather>\n'
            f'    <Redirect>https://{koyeb_url}/api/exotel/incoming?call_sid={call_sid}</Redirect>\n'
            '</Response>'
        )

    async def gather_response(self, call_sid: str, speech_result: str) -> str:
        """Process gathered speech and return next Exotel XML."""
        koyeb_url = Config.KOYEB_APP_URL
        if speech_result:
            ai_text = await self.voice_agent.generate_response(speech_result, call_sid)
            self.session_manager.add_conversation_turn(call_sid, speech_result, ai_text)
            say_text = ai_text
        else:
            say_text = "I didn't catch that. Let's try again."

        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Response>\n'
            f'    <Say>{say_text}</Say>\n'
            f'    <Gather action="https://{koyeb_url}/api/exotel/gather-response?call_sid={call_sid}" method="POST" input="speech" timeout="5">\n'
            '    </Gather>\n'
            f'    <Redirect>https://{koyeb_url}/api/exotel/incoming?call_sid={call_sid}</Redirect>\n'
            '</Response>'
        )

    # ---------- Media Streams Method (real-time, lower latency) ----------
    def incoming_call_stream(self, call_sid: str) -> str:
        """Return Exotel XML that connects to a WebSocket stream."""
        koyeb_url = Config.KOYEB_APP_URL
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Response>\n'
            f'    <Stream url="wss://{koyeb_url}/ws/exotel-stream?callSid={call_sid}" />\n'
            '</Response>'
        )

    async def process_media_stream(self, websocket, call_sid: str):
        """Handle bi-directional media stream with Exotel."""
        await websocket.accept()
        logger.info(f"Media stream opened for call {call_sid}")

        stream_sid = None
        audio_buffer = b""
        speech_started = False
        silence_frames = 0
        SILENCE_THRESHOLD = 30  # ~300ms at 100ms frames

        try:
            while True:
                message = await websocket.receive_text()
                data = json.loads(message)

                if data.get("event") == "start":
                    stream_sid = data.get("streamSid") or data.get("stream_sid")
                    logger.info(f"Stream started: {stream_sid}")

                elif data.get("event") == "media":
                    payload = data.get("media", {}).get("payload", "")
                    if not payload:
                        continue
                    # Exotel sends μ-law encoded audio
                    mulaw_audio = base64.b64decode(payload)
                    # Convert μ-law to PCM
                    pcm_audio = audioop.ulaw2lin(mulaw_audio, 2)
                    # Resample from 8kHz to 16kHz (required by Azure)
                    pcm_16k = self._resample_8k_to_16k(pcm_audio)

                    # Simple VAD: check energy
                    energy = self._audio_energy(pcm_16k)
                    if energy > 500:
                        if not speech_started:
                            speech_started = True
                            audio_buffer = pcm_16k
                        else:
                            audio_buffer += pcm_16k
                        silence_frames = 0
                    else:
                        if speech_started:
                            silence_frames += 1
                            if silence_frames >= SILENCE_THRESHOLD:
                                # Process complete utterance
                                if audio_buffer:
                                    text = await self.voice_agent.speech_to_text(audio_buffer)
                                    if text:
                                        logger.info(f"Recognized: {text}")
                                        ai_text = await self.voice_agent.generate_response(text, call_sid)
                                        self.session_manager.add_conversation_turn(call_sid, text, ai_text)
                                        audio_resp = await self.voice_agent.text_to_speech(ai_text)
                                        if audio_resp:
                                            # Convert PCM 16k to μ-law 8k for Exotel
                                            pcm_8k = self._resample_16k_to_8k(audio_resp)
                                            mulaw_resp = audioop.lin2ulaw(pcm_8k, 2)
                                            resp_payload = base64.b64encode(mulaw_resp).decode('utf-8')
                                            await websocket.send_text(json.dumps({
                                                "event": "media",
                                                "streamSid": stream_sid,
                                                "media": {"payload": resp_payload}
                                            }))
                                # Reset for next utterance
                                audio_buffer = b""
                                speech_started = False
                                silence_frames = 0

                elif data.get("event") == "stop":
                    logger.info(f"Stream stopped: {stream_sid}")
                    break

        except Exception as e:
            logger.error(f"Error in media stream: {e}")
        finally:
            self.session_manager.end_session(call_sid)
            try:
                await websocket.close()
            except Exception:
                pass

    def _resample_8k_to_16k(self, pcm_8k: bytes) -> bytes:
        """Resample PCM from 8kHz to 16kHz using audioop or numpy."""
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
        """Resample PCM from 16kHz to 8kHz using audioop or numpy."""
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