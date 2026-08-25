import os
import logging
import uuid
import asyncio
import json
import audioop
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import Config
from voice_agent import VoiceAgent
from session_manager import SessionManager
from exotel_handler import ExotelHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("voice_app")

# Validate environment
Config.validate()

app = FastAPI(title="AI Voice Agent Backend (Exotel + Azure + OpenRouter)")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

voice_agent = VoiceAgent()
session_manager = SessionManager()
exotel_handler = ExotelHandler(voice_agent, session_manager)


async def extract_params(request: Request) -> dict:
    """Extract parameters from Query Params, Form Data, or JSON body."""
    params = dict(request.query_params)
    try:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
            if isinstance(body, dict):
                params.update(body)
        elif "form" in content_type:
            form = await request.form()
            params.update(dict(form))
    except Exception:
        pass
    return params


# ----------------------- Health & Info -----------------------
@app.get("/")
async def root():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception:
        return {"status": "online", "service": "AI Voice Support (Exotel)"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "active_sessions": session_manager.active_count(),
        "exotel_configured": bool(Config.EXOTEL_ACCOUNT_SID and Config.EXOTEL_API_KEY),
        "azure_configured": bool(Config.AZURE_SPEECH_KEY and not Config.AZURE_SPEECH_KEY.startswith("your_")),
        "openrouter_configured": bool(Config.OPENROUTER_API_KEY and not Config.OPENROUTER_API_KEY.startswith("your_")),
        "openrouter_model": Config.OPENROUTER_MODEL,
        "streaming_mode": Config.EXOTEL_USE_STREAM
    }


# ----------------------- Exotel HTTP Endpoints (GET + POST) -----------------------
@app.api_route("/api/exotel/incoming", methods=["GET", "POST"])
@app.api_route("/api/exotel/passthru", methods=["GET", "POST"])
async def exotel_incoming(request: Request):
    """
    Handle incoming call from Exotel Passthru, Dynamic URL, or Flow Builder.
    Supports both GET and POST requests.
    """
    params = await extract_params(request)
    call_sid = params.get("CallSid") or params.get("CallUUID") or params.get("Sid") or str(uuid.uuid4())
    from_number = params.get("From") or params.get("Caller") or params.get("CallFrom") or "unknown"
    to_number = params.get("To") or params.get("CallTo") or "unknown"

    logger.info(f"📞 [INCOMING CALL] Method={request.method} | From={from_number} | To={to_number} | CallSid={call_sid}")

    if not session_manager.get_session(call_sid):
        session_manager.create_session(call_sid, {"from": from_number, "to": to_number})

    if Config.EXOTEL_USE_STREAM:
        exoml = exotel_handler.incoming_call_stream(call_sid)
    else:
        exoml = exotel_handler.incoming_call_gather(call_sid)

    logger.info(f"📄 [EXOML RESPONSE] Returning XML to Exotel:\n{exoml}")
    return Response(content=exoml, media_type="application/xml")


@app.api_route("/api/exotel/gather-response", methods=["GET", "POST"])
async def exotel_gather_response(request: Request):
    """
    Handle Exotel Gather speech recognition result.
    Processes speech with LLM and returns next ExoML.
    """
    params = await extract_params(request)
    call_sid = params.get("call_sid") or params.get("CallSid") or params.get("CallUUID") or "default"
    speech_result = params.get("SpeechResult") or params.get("Digits") or ""

    logger.info(f"🗣️ [GATHER RESULT] CallSid={call_sid} | Speech=\"{speech_result}\"")
    exoml = await exotel_handler.gather_response(call_sid, speech_result)
    return Response(content=exoml, media_type="application/xml")


@app.api_route("/api/exotel/status", methods=["GET", "POST"])
async def exotel_status_callback(request: Request):
    """Receive call status updates from Exotel (hangup, completed, failed)."""
    params = await extract_params(request)
    call_sid = params.get("CallSid") or params.get("CallUUID")
    status = params.get("Status") or params.get("CallStatus") or "unknown"
    logger.info(f"ℹ️ [STATUS CALLBACK] CallSid={call_sid} | Status={status}")

    if status in ["completed", "failed", "busy", "no-answer", "canceled"] and call_sid:
        session_manager.end_session(call_sid)

    return JSONResponse({"status": "received"})


# ----------------------- Exotel WebSocket Media Streaming -----------------------
@app.websocket("/ws/exotel-stream")
@app.websocket("/ws/media")
@app.websocket("/ws/audio")
@app.websocket("/ws/stream")
async def exotel_media_stream_ws(websocket: WebSocket):
    """
    WebSocket endpoint for Exotel VoiceBot and real-time audio media streams.
    Handles bi-directional 8kHz μ-law audio.
    """
    call_sid = websocket.query_params.get("callSid") or websocket.query_params.get("call_sid") or "unknown"
    await exotel_handler.process_media_stream(websocket, call_sid)


# ----------------------- Browser WebSocket Endpoint -----------------------
@app.websocket("/ws/browser")
async def browser_voice(websocket: WebSocket):
    """WebSocket for browser audio testing UI."""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    session_manager.create_session(session_id)
    logger.info(f"🌐 [BROWSER] Connected: {session_id}")

    try:
        greeting = await voice_agent.generate_greeting()
        await websocket.send_json({"type": "greeting", "text": greeting})
        audio_greeting = await voice_agent.text_to_speech(greeting, format_type="pcm")
        if audio_greeting:
            await websocket.send_bytes(audio_greeting)

        pcm_audio_buffer = b""
        speech_started = False
        silence_chunks = 0
        energy_threshold = 500
        silence_chunk_limit = 3
        min_audio_bytes = 16000

        while True:
            data = await websocket.receive()
            if data.get("type") == "websocket.disconnect":
                break

            if "bytes" in data and data["bytes"]:
                pcm_chunk = data["bytes"]

                try:
                    energy = audioop.rms(pcm_chunk, 2)
                except Exception:
                    continue

                if energy > energy_threshold:
                    if not speech_started:
                        logger.info(f"🌐 [BROWSER VAD] Speech started (energy={energy})")
                        speech_started = True
                        pcm_audio_buffer = pcm_chunk
                    else:
                        pcm_audio_buffer += pcm_chunk
                    silence_chunks = 0
                    continue

                if speech_started:
                    pcm_audio_buffer += pcm_chunk
                    silence_chunks += 1

                    if silence_chunks < silence_chunk_limit:
                        continue

                    utterance = pcm_audio_buffer
                    pcm_audio_buffer = b""
                    speech_started = False
                    silence_chunks = 0

                    if len(utterance) < min_audio_bytes:
                        continue

                    logger.info(f"🌐 [BROWSER VAD] Speech ended. Buffer: {len(utterance)} bytes. Processing STT...")
                    text = await voice_agent.speech_to_text(utterance, is_mulaw=False)
                    if not text:
                        continue

                    await websocket.send_json({"type": "transcript", "speaker": "user", "text": text})
                    session = session_manager.get_session(session_id) or {}
                    history = session.get("conversation", [])
                    ai_text = await voice_agent.generate_response(text, session_id, history)
                    session_manager.add_conversation_turn(session_id, text, ai_text)
                    await websocket.send_json({"type": "transcript", "speaker": "ai", "text": ai_text})
                    audio_resp = await voice_agent.text_to_speech(ai_text, format_type="pcm")
                    if audio_resp:
                        await websocket.send_bytes(audio_resp)

            elif "text" in data and data["text"]:
                try:
                    msg = json.loads(data["text"])
                    if msg.get("type") == "end":
                        break
                    if msg.get("type") == "text_message":
                        text = str(msg.get("message", "")).strip()
                        if not text:
                            continue

                        session = session_manager.get_session(session_id) or {}
                        history = session.get("conversation", [])
                        ai_text = await voice_agent.generate_response(text, session_id, history)
                        session_manager.add_conversation_turn(session_id, text, ai_text)
                        await websocket.send_json({"type": "transcript", "speaker": "ai", "text": ai_text})
                        audio_resp = await voice_agent.text_to_speech(ai_text, format_type="pcm")
                        if audio_resp:
                            await websocket.send_bytes(audio_resp)
                except Exception:
                    pass

    except WebSocketDisconnect:
        logger.info(f"🌐 [BROWSER] Disconnected: {session_id}")
    except Exception as e:
        logger.error(f"🌐 [BROWSER] Error: {e}", exc_info=True)
    finally:
        session_manager.end_session(session_id)


# ----------------------- Session Management APIs -----------------------
@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.post("/api/session/{session_id}/end")
async def end_session_api(session_id: str):
    session_manager.end_session(session_id)
    return {"status": "ended"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=Config.HOST, port=Config.PORT, reload=True)
