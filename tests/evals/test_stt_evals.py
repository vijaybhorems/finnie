"""Speech-to-text accuracy on finance phrases (design doc, Evaluation layer 3).

Paid, and only worth running when the Deepgram model or the key terms change
(src/voice/stt.py):

    FINNIE_EVAL_AUDIO=1 pytest tests/evals/test_stt_evals.py -v -s

Each phrase in data/stt_phrases.jsonl is synthesised with macOS `say` in four
accents, plus a fast and a noisy take (pink noise mixed in by ffmpeg), and sent
to Deepgram with the worker's own options (`deepgram_options()`). Scored on word
error rate, key terms ("Roth", "401(k)", "RSI", tickers) and numbers.

Costs a few cents: about 430 clips of 2–3 s. Audio and transcripts are cached
in tests/evals/.stt_cache/ (git-ignored), keyed by the audio, model and key
terms, so a re-run is free until one of them changes.

Synthetic speech is cleaner than people. A phrase can point at a real recording
instead (`"audio": "path.wav"` in the manifest); the doc plans 50–100 of them,
kept in a bucket rather than the repo.

The tests at the top run offline, in CI: they check the scoring and the manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from dotenv import dotenv_values

from tests.evals.stt_metrics import has_number, has_term, normalize, wer

_HERE = Path(__file__).resolve().parent
MANIFEST = _HERE / "data" / "stt_phrases.jsonl"
CACHE = _HERE / ".stt_cache"
AUDIO_MODE = os.environ.get("FINNIE_EVAL_AUDIO", "0") == "1"

# (variant, macOS voice, words per minute, add noise)
VARIANTS = [
    ("us", "Samantha", 175, False),
    ("uk", "Daniel", 175, False),
    ("au", "Karen", 175, False),
    ("in", "Rishi", 175, False),
    ("us_fast", "Samantha", 230, False),
    ("us_noisy", "Samantha", 175, True),
]
# A little below the run of 2026-11-15 (nova-3, KEYTERMS as committed, with
# clean_transcript): WER 0.027, key terms 0.991, numbers 0.968 on 426 clips.
# A change to the model or key terms should not fall below them.
MAX_WER = 0.05
MIN_KEY_TERMS = 0.95
MIN_NUMBERS = 0.92


def _phrases() -> list[dict[str, Any]]:
    return [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]


# ── Offline: the scoring and the manifest ─────────────────────────────────────

class TestScoring:
    def test_formatting_differences_are_not_errors(self):
        assert wer("My 401(k) match is 4%.", "my 401k match is 4 percent") == (0, 6)
        assert wer("I put $6,500 in my Roth.", "I put 6500 dollars in my Roth") == (1, 6)  # "dollars" inserted
        assert wer("How is the S&P 500 doing?", "how is the s and p 500 doing") == (0, 8)

    def test_misheard_words_count(self):
        assert wer("How does a Roth IRA work?", "how does a wrath ira work") == (1, 6)

    def test_key_terms(self):
        assert has_term("roth ira", "Should I open a Roth IRA?")
        assert has_term("etf", "Are index ETFs cheaper?")
        assert has_term("401k", "my 401(k) match")
        assert not has_term("roth", "my wrath ira")
        assert not has_term("rsi", "versions")

    def test_numbers(self):
        assert has_number("6500", "I put $6,500 in")
        assert has_number("1.5", "retire with $1,500,000")
        assert has_number("1500000", "1.5 million dollars")
        assert has_number("7", "seven percent")
        assert not has_number("2028", "in 2018")


class TestManifest:
    def test_ids_are_unique(self):
        ids = [p["id"] for p in _phrases()]
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize("phrase", _phrases(), ids=lambda p: p["id"])
    def test_key_terms_and_numbers_are_in_the_reference(self, phrase):
        for term in phrase["key_terms"]:
            assert has_term(term, phrase["text"]), (term, normalize(phrase["text"]))
        for number in phrase["numbers"]:
            assert has_number(number, phrase["text"]), (number, normalize(phrase["text"]))

    def test_the_worker_key_terms_are_all_exercised(self):
        from src.voice.stt import KEYTERMS

        text = " ".join(p["text"] for p in _phrases())
        missing = [k for k in KEYTERMS if not has_term(k, text)]
        assert not missing


# ── Paid: Deepgram on synthesised audio ───────────────────────────────────────

def _deepgram_key() -> str:
    return dotenv_values(_HERE.parents[1] / ".env").get("DEEPGRAM_API_KEY") or ""


def _synthesise(phrase: dict[str, Any], voice: str, rate: int, noisy: bool, out: Path) -> Path:
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    clean = out.with_suffix(".clean.wav")
    subprocess.run(
        ["say", "-v", voice, "-r", str(rate), "-o", str(clean),
         "--file-format=WAVE", "--data-format=LEI16@16000", phrase.get("say", phrase["text"])],
        check=True,
    )
    if not noisy:
        clean.rename(out)
        return out
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(clean),
         "-f", "lavfi", "-i", "anoisesrc=color=pink:amplitude=0.03:sample_rate=16000",
         "-filter_complex", "amix=inputs=2:duration=first:normalize=0", "-ar", "16000", str(out)],
        check=True,
    )
    clean.unlink()
    return out


def _transcribe(audio: Path, options: dict[str, Any], key: str, cache: dict[str, Any]) -> dict[str, Any]:
    import httpx

    data = audio.read_bytes()
    cache_key = "|".join([
        hashlib.sha256(data).hexdigest()[:20], options["model"],
        hashlib.sha256(json.dumps(options["keyterm"]).encode()).hexdigest()[:10],
    ])
    if cache_key in cache:
        return cache[cache_key]
    params: list[tuple[str, str]] = [
        ("model", options["model"]), ("language", options["language"]),
        ("smart_format", str(options["smart_format"]).lower()),
        ("mip_opt_out", str(options["mip_opt_out"]).lower()),
        *[("keyterm", term) for term in options["keyterm"]],
    ]
    response = httpx.post(
        "https://api.deepgram.com/v1/listen", params=params, content=data, timeout=60,
        headers={"Authorization": f"Token {key}", "Content-Type": "audio/wav"},
    )
    response.raise_for_status()
    best = response.json()["results"]["channels"][0]["alternatives"][0]
    cache[cache_key] = {"transcript": best["transcript"], "confidence": best["confidence"]}
    return cache[cache_key]


@pytest.mark.skipif(not AUDIO_MODE, reason="Set FINNIE_EVAL_AUDIO=1 (paid: Deepgram)")
def test_deepgram_on_finance_phrases():
    if not _deepgram_key():
        pytest.skip("DEEPGRAM_API_KEY not in .env")
    if not shutil.which("say") or not shutil.which("ffmpeg"):
        pytest.skip("needs macOS `say` and ffmpeg")
    from src.core.config import get_settings
    from src.voice.stt import clean_transcript, deepgram_options

    voice = get_settings().voice
    options = deepgram_options(voice.stt_model)
    cache_file = CACHE / "transcripts.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}

    jobs = []
    for phrase in _phrases():
        for variant, macos_voice, rate, noisy in VARIANTS:
            if "audio" in phrase:  # a real recording: one clip, no variants
                if variant != VARIANTS[0][0]:
                    continue
                audio = Path(phrase["audio"])
            else:
                audio = _synthesise(phrase, macos_voice, rate, noisy, CACHE / "audio" / f"{phrase['id']}-{variant}.wav")
            jobs.append((phrase, variant, audio))

    key = _deepgram_key()
    with ThreadPoolExecutor(max_workers=8) as pool:
        heard = list(pool.map(lambda job: _transcribe(job[2], options, key, cache), jobs))
    cache_file.write_text(json.dumps(cache, indent=1))

    rows = []
    for (phrase, variant, _), result in zip(jobs, heard):
        transcript = clean_transcript(result["transcript"])  # what the graph receives
        edits, words = wer(phrase["text"], transcript)
        rows.append({
            "id": phrase["id"], "variant": variant, "text": phrase["text"], "heard": transcript,
            "confidence": result["confidence"], "edits": edits, "words": words,
            "terms": [(t, has_term(t, transcript)) for t in phrase["key_terms"]],
            "numbers": [(n, has_number(n, transcript)) for n in phrase["numbers"]],
        })
    report = _report(rows, voice.memory_min_transcript_confidence)
    (CACHE / "report.json").write_text(json.dumps({"summary": report, "rows": rows}, indent=1))

    overall = report["all"]
    assert overall["wer"] <= MAX_WER, overall
    assert overall["key_terms"] >= MIN_KEY_TERMS, overall
    assert overall["numbers"] >= MIN_NUMBERS, overall


def _score(rows: list[dict[str, Any]], low_confidence: float) -> dict[str, float]:
    terms = [ok for r in rows for _, ok in r["terms"]]
    numbers = [ok for r in rows for _, ok in r["numbers"]]
    return {
        "clips": len(rows),
        "wer": round(sum(r["edits"] for r in rows) / max(1, sum(r["words"] for r in rows)), 3),
        "key_terms": round(sum(terms) / max(1, len(terms)), 3),
        "numbers": round(sum(numbers) / max(1, len(numbers)), 3),
        # Turns the worker would not learn memories from (voice.memory_min_transcript_confidence).
        "low_confidence": round(sum(r["confidence"] < low_confidence for r in rows) / max(1, len(rows)), 3),
    }


def _report(rows: list[dict[str, Any]], low_confidence: float) -> dict[str, Any]:
    report = {"all": _score(rows, low_confidence)}
    for variant, *_ in VARIANTS:
        report[variant] = _score([r for r in rows if r["variant"] == variant], low_confidence)
    print(f"\n{'variant':10} {'clips':>5} {'WER':>6} {'terms':>6} {'numbers':>8} {'low conf':>9}")
    for name, s in report.items():
        print(f"{name:10} {s['clips']:>5} {s['wer']:>6.3f} {s['key_terms']:>6.3f} {s['numbers']:>8.3f} {s['low_confidence']:>9.3f}")
    missed: dict[str, int] = {}
    for r in rows:
        for term, ok in r["terms"] + r["numbers"]:
            if not ok:
                missed[term] = missed.get(term, 0) + 1
    print("most missed:", sorted(missed.items(), key=lambda kv: -kv[1])[:12])
    worst = sorted(rows, key=lambda r: -r["edits"] / max(1, r["words"]))[:8]
    for r in worst:
        print(f"  {r['variant']:8} {r['text']!r} -> {r['heard']!r}")
    return report
