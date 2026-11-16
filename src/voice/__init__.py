"""Voice calls: a LiveKit worker that speaks Finnie's answers (design doc: Finnie Voice Agent).

`access` and `speech` are pure and import no LiveKit code, so the main app and
CI can use them. `worker` and `prefetch` need the voice dependencies
(requirements-voice.txt) and run only in the worker process.
"""
