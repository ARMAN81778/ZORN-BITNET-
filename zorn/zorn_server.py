"""OpenAI-compatible HTTP server for ZORN.

Built on Flask (bundled with this environment) rather than FastAPI/uvicorn,
which are not available. Flask's built-in dev server is single-process and
fine for local/small deployments; for real production traffic front it with
gunicorn/waitress + a reverse proxy, same as any Flask app.

Fixes vs. the previous version:
  - Multi-session support: each request carries a `session_id` (or one is
    generated) and gets its own ZornChat instance instead of every request
    reading/writing ONE shared `chat` object. The previous version had a
    real concurrency bug: two simultaneous users would stomp on each
    other's `chat.messages` history.
  - Idle sessions are evicted after `session_ttl_seconds` so long-running
    servers don't leak memory across abandoned conversations.
  - Real token-by-token streaming (Server-Sent-Events style chunks) using
    the existing `ZornSession.stream_ids` / `ZornChat.stream` generator,
    instead of buffering the whole response.
  - Structured JSON error responses (400/404/500) instead of unhandled
    exceptions turning into raw tracebacks or connection resets.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any, Optional

from zorn import ZornChat, ZornError


class _SessionStore:
    """Thread-safe registry of one ZornChat per conversation/session id."""

    def __init__(self, model_dir: str, cpu_cores: Optional[int], system_prompt: Optional[str],
                 ttl_seconds: float = 3600.0):
        self.model_dir = model_dir
        self.cpu_cores = cpu_cores
        self.system_prompt = system_prompt
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._sessions: dict[str, tuple[ZornChat, float]] = {}

    def get_or_create(self, session_id: str) -> ZornChat:
        with self._lock:
            self._evict_expired_locked()
            entry = self._sessions.get(session_id)
            if entry is None:
                chat = ZornChat(
                    self.model_dir, cpu_cores=self.cpu_cores,
                    system_prompt=self.system_prompt,
                )
                self._sessions[session_id] = (chat, time.time())
                return chat
            chat, _ = entry
            self._sessions[session_id] = (chat, time.time())
            return chat

    def reset(self, session_id: str) -> bool:
        with self._lock:
            entry = self._sessions.pop(session_id, None)
            if entry is None:
                return False
            entry[0].reset()
            return True

    def _evict_expired_locked(self):
        now = time.time()
        stale = [sid for sid, (_, ts) in self._sessions.items() if now - ts > self.ttl_seconds]
        for sid in stale:
            self._sessions.pop(sid, None)

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)


def _error_response(message: str, status: int):
    from flask import jsonify
    resp = jsonify({"error": {"message": message, "type": "invalid_request_error" if status < 500 else "server_error"}})
    resp.status_code = status
    return resp


def create_app(model_dir: str, cpu_cores: Optional[int] = None,
               system_prompt: Optional[str] = None, session_ttl_seconds: float = 3600.0):
    try:
        from flask import Flask, request, jsonify, Response, stream_with_context
    except ImportError as e:
        raise ImportError("Install server dependencies: pip install flask") from e

    app = Flask("zorn_server")

    # Load once up front so a bad model directory fails fast at startup,
    # not on the first request.
    try:
        probe = ZornChat(model_dir, cpu_cores=cpu_cores, system_prompt=system_prompt)
        model_name = probe.session.info.get("model_name", "zorn-model") if not probe.gguf else "zorn-gguf-model"
    except (ZornError, ImportError) as e:
        raise RuntimeError(f"Could not start ZORN server: {e}") from e

    sessions = _SessionStore(model_dir, cpu_cores, system_prompt, ttl_seconds=session_ttl_seconds)

    def _extract_request(body: dict[str, Any]):
        messages = body.get("messages") or []
        if not messages:
            raise ValueError("'messages' is required and must be non-empty")
        if not isinstance(messages, list):
            raise ValueError("'messages' must be a list")
        turns = [m for m in messages if m.get("role") != "system"]
        if not turns:
            raise ValueError("'messages' must include at least one non-system message")
        last = turns[-1]
        if last.get("role") != "user":
            raise ValueError("the last message must have role 'user'")
        text = last.get("content", "")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("the last user message must have non-empty text content")

        params = {
            "max_new_tokens": int(body.get("max_tokens", 256)),
            "temperature": float(body.get("temperature", 0.7)),
            "top_p": float(body.get("top_p", 0.9)),
            "top_k": int(body.get("top_k", 40)),
            "repetition_penalty": float(body.get("repetition_penalty", 1.05)),
        }
        if params["max_new_tokens"] <= 0:
            raise ValueError("'max_tokens' must be a positive integer")
        return text, params

    @app.get("/v1/models")
    def models():
        return jsonify({"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "zorn"}]})

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "model": model_name, "active_sessions": sessions.count()})

    @app.post("/v1/chat/completions")
    def completions():
        body = request.get_json(silent=True)
        if body is None:
            return _error_response("Request body must be valid JSON", 400)

        session_id = request.headers.get("X-Session-Id") or body.get("session_id") or uuid.uuid4().hex

        try:
            text, params = _extract_request(body)
        except ValueError as e:
            return _error_response(str(e), 400)

        try:
            chat = sessions.get_or_create(session_id)
        except (ZornError, ImportError) as e:
            return _error_response(f"Could not initialize session: {e}", 500)

        stream = bool(body.get("stream", False))

        if not stream:
            try:
                answer = chat.ask(text, **params)
            except ZornError as e:
                return _error_response(f"Generation failed: {e}", 500)
            except Exception as e:  # keep the process alive; report, don't crash
                return _error_response(f"Unexpected server error: {e}", 500)

            return jsonify({
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "session_id": session_id,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
            })

        # Streaming path: Server-Sent-Events style, OpenAI-compatible chunk shape.
        def generate_chunks():
            completion_id = "chatcmpl-" + uuid.uuid4().hex
            try:
                for piece in chat.stream(text, **params):
                    chunk = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": int(time.time()), "model": model_name,
                        "session_id": session_id,
                        "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                final = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": model_name,
                    "session_id": session_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(final)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                err = {"error": {"message": f"Generation failed: {e}", "type": "server_error"}}
                yield f"data: {json.dumps(err)}\n\n"

        return Response(stream_with_context(generate_chunks()), mimetype="text/event-stream")

    @app.post("/v1/sessions/<session_id>/reset")
    def reset_session(session_id: str):
        existed = sessions.reset(session_id)
        return jsonify({"session_id": session_id, "reset": existed})

    @app.errorhandler(404)
    def not_found(_e):
        return _error_response("Not found", 404)

    @app.errorhandler(500)
    def server_error(_e):
        return _error_response("Internal server error", 500)

    return app


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Run the ZORN OpenAI-compatible server")
    ap.add_argument("model_dir")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--cpu-cores", type=int, default=None)
    ap.add_argument("--system-prompt", default=None)
    args = ap.parse_args()

    app = create_app(args.model_dir, cpu_cores=args.cpu_cores, system_prompt=args.system_prompt)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
