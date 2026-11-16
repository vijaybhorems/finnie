"""Messages the worker sends to the caller's page during a call.

Pure (no LiveKit), so CI tests it. LiveKit already sends the page the live
transcript and the agent's state; these add what only the worker knows:

- "answer": the full written answer, as saved (figures, disclaimer), with the
  agent that wrote it. Sent when the graph finishes, even if the user cut the
  spoken answer short (design doc, Page changes item 3).
- "spoken": what was actually said for a turn, and whether the user
  interrupted it (shown beneath the answer, marked "interrupted").
- "notice": a line for the call bar, such as both voices failing.

Sent as LiveKit text streams on TOPIC, to the caller's identity only.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from src.voice.turns import SORRY

TOPIC = "finnie.voice"
SPEECH_DOWN = "Speech isn't working right now; the answer is on screen."


def answer_event(turn: int, agent_used: Optional[str], final_response: Optional[str]) -> dict[str, Any]:
    """The written answer for `turn`. A failed turn shows the apology, never its error text."""
    agent = agent_used or "error"
    answer = SORRY if agent == "error" or not final_response else final_response
    return {"type": "answer", "turn": turn, "agent": agent, "answer": answer}


def spoken_event(turn: int, spoken: str, interrupted: bool) -> dict[str, Any]:
    """What was said for `turn` (0: the session-start disclaimer)."""
    return {"type": "spoken", "turn": turn, "spoken": " ".join((spoken or "").split()),
            "interrupted": bool(interrupted)}


def notice_event(text: str) -> dict[str, Any]:
    return {"type": "notice", "text": text}


def encode(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False)
