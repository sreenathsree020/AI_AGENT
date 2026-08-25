import os
import logging
import uuid
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import Config
from voice_agent import VoiceAgent
from session_manager import SessionManager
from exotel_handler import ExotelHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate environment
Config.validate()

app = FastAPI(title="AI Voice Customer Support (Exotel + OpenRouter)")
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
        "openrouter_model": Config.OPENROUTER_MODEL
    }


# ----------------------- Exotel Endpoints -----------------------
@app.post("/api/exotel/incoming")
async def exotel_incoming(request: Request):
    """Handle incoming call from Exotel. Supports Gather or Media Streams."""
    form = await request.form()
    call_sid = form.get("CallSid") or form.get("CallUUID") or str(uuid.uuid4())
    from_number = form.get("From", "unknown")
    logger.info(f"Incoming call from {from_number}, SID: {call_sid}")

    if not session_manager.get_session(call_sid):
        session_manager.create_session(call_sid, {"from": from_number})

    if Config.EXOTEL_USE_STREAM:
        twiml = exotel_handler.incoming_call_stream(call_sid)
    else:
        twiml = exotel_handler.incoming_call_gather(call_sid)

    return Response(content=twiml, media_type="application/xml")


@app.post("/api/exotel/gather-response")
async def exotel_gather_response(request: Request):
    """Handle Exotel Gather speech recognition result."""
    form = await request.form()
    call_sid = request.query_params.get("call_sid") or form.get("CallSid") or form.get("CallUUID") or "default"
    speech_result = form.get("SpeechResult", "")
    logger.info(f"Gather response from {call_sid}: {speech_result}")

    twiml = await exotel_handler.gather_response(call_sid, speech_result)
    return Response(content=twiml, media_type="application/xml")


@app.websocket("/ws/exotel-stream")
async def exotel_media_stream(websocket: WebSocket):
    """WebSocket endpoint for Exotel Media Streams."""
    call_sid = websocket.query_params.get("callSid", "unknown")
    await exotel_handler.process_media_stream(websocket, call_sid)


# ----------------------- Browser WebSocket Endpoint -----------------------
@app.websocket("/ws/browser")
async def browser_voice(websocket: WebSocket):
    """WebSocket for browser audio testing."""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    session_manager.create_session(session_id)
    logger.info(f"Browser client connected: {session_id}")

    try:
        greeting = await voice_agent.generate_greeting()
        await websocket.send_json({"type": "greeting", "text": greeting})
        audio_greeting = await voice_agent.text_to_speech(greeting)
        if audio_greeting:
            await websocket.send_bytes(audio_greeting)

        while True:
            data = await websocket.receive()
            if data.get("type") == "websocket.disconnect":
                break

            if "bytes" in data and data["bytes"]:
                pcm_audio = data["bytes"]
                text = await voice_agent.speech_to_text(pcm_audio)
                if text:
                    await websocket.send_json({"type": "transcript", "speaker": "user", "text": text})
                    ai_text = await voice_agent.generate_response(text, session_id)
                    session_manager.add_conversation_turn(session_id, text, ai_text)
                    await websocket.send_json({"type": "transcript", "speaker": "ai", "text": ai_text})
                    audio_resp = await voice_agent.text_to_speech(ai_text)
                    if audio_resp:
                        await websocket.send_bytes(audio_resp)

            elif "text" in data and data["text"]:
                try:
                    msg = json.loads(data["text"])
                    if msg.get("type") == "end":
                        break
                except Exception:
                    pass

    except WebSocketDisconnect:
        logger.info(f"Browser client disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Error in browser websocket: {e}")
    finally:
        session_manager.end_session(session_id)


# ----------------------- Session API -----------------------
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