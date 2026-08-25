import redis
import os
import json
from datetime import datetime
import logging
from typing import Optional, Dict, Any

from config import Config

logger = logging.getLogger(__name__)


class SessionManager:
    def __init__(self):
        self.redis_url = Config.REDIS_URL
        self.timeout = Config.SESSION_TIMEOUT
        self._memory_store: Dict[str, Dict[str, Any]] = {}
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=2.0
            )
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}. In-memory fallback will be used.")
            self.redis_client = None

    def create_session(self, session_id: str, data: Optional[dict] = None) -> dict:
        session = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "conversation": [],
            "status": "active",
            "data": data or {}
        }
        self.save(session_id, session)
        return session

    def get_session(self, session_id: str) -> Optional[dict]:
        if self.redis_client:
            try:
                raw = self.redis_client.get(f"session:{session_id}")
                if raw:
                    return json.loads(raw)
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
        return self._memory_store.get(session_id)

    def save(self, session_id: str, session: dict) -> None:
        session["last_activity"] = datetime.now().isoformat()
        self._memory_store[session_id] = session
        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"session:{session_id}",
                    self.timeout,
                    json.dumps(session)
                )
            except Exception as e:
                logger.warning(f"Redis save error: {e}")

    def add_conversation_turn(self, session_id: str, customer_text: str, agent_response: str) -> None:
        session = self.get_session(session_id) or self.create_session(session_id)
        session.setdefault("conversation", []).append({
            "timestamp": datetime.now().isoformat(),
            "customer": customer_text,
            "agent": agent_response
        })
        # Keep last 50 turns
        session["conversation"] = session["conversation"][-50:]
        self.save(session_id, session)

    def end_session(self, session_id: str) -> None:
        session = self.get_session(session_id)
        if session:
            session["status"] = "ended"
            session["ended_at"] = datetime.now().isoformat()
            self.save(session_id, session)
            if self.redis_client:
                try:
                    self.redis_client.delete(f"session:{session_id}")
                except Exception as e:
                    logger.warning(f"Redis delete error: {e}")
            self._memory_store.pop(session_id, None)

    def active_count(self) -> int:
        """Fast count using in-memory store (always synced with Redis)."""
        return sum(1 for s in self._memory_store.values() if s.get("status") == "active")
