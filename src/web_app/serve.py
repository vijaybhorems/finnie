"""Container entry point: start the warm-up, then run Streamlit in this process.

``streamlit run`` gives no hook to run code at server start, so the first page
render would be the earliest the warm-up could begin. Launching Streamlit from
here instead starts it while the server boots.

With ``FINNIE_WARM_BEFORE_SERVE=1`` (set by ``docker/entrypoint.sh``) the port
isn't opened until the warm-up has finished. Cloud Run's startup probe is a TCP
check on that port, so an instance only takes traffic once LangGraph and the
embedding model are loaded — and the load runs under startup CPU boost instead
of the throttled CPU an instance gets once it is marked ready. The wait is
capped by ``FINNIE_WARMUP_TIMEOUT`` (seconds, default 540, under Cloud Run's
600s startup limit); past it Streamlit starts anyway and pages wait for the
warm-up themselves, as they do without the flag.

Usage: ``python -m src.web_app.serve [streamlit run options...]``
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_APP = Path(__file__).with_name("app.py")


def _warm_before_serve() -> bool:
    return os.environ.get("FINNIE_WARM_BEFORE_SERVE", "").strip().lower() in {"1", "true", "yes"}


def _warmup_timeout() -> float:
    try:
        return float(os.environ.get("FINNIE_WARMUP_TIMEOUT", "540"))
    except ValueError:
        return 540.0


def main() -> None:
    from src.web_app import warmup

    warmup.start()

    if _warm_before_serve():
        from src.utils.logger import get_logger

        logger = get_logger(__name__)
        started = time.monotonic()
        ready = warmup.wait(_warmup_timeout())
        logger.info(
            "serve_after_warmup" if ready else "serve_warmup_timed_out",
            waited_seconds=round(time.monotonic() - started, 1),
        )

    from streamlit.web import cli

    sys.argv = ["streamlit", "run", str(_APP), *sys.argv[1:]]
    sys.exit(cli.main())


if __name__ == "__main__":
    main()
