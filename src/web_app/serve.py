"""Container entry point: start the warm-up, then run Streamlit in this process.

``streamlit run`` gives no hook to run code at server start, so the first page
render would be the earliest the warm-up could begin. Launching Streamlit from
here instead starts it while the server boots, and by the time a visitor has
signed in with Google the workflow and embedding model are usually loaded.

Usage: ``python -m src.web_app.serve [streamlit run options...]``
"""
from __future__ import annotations

import sys
from pathlib import Path

_APP = Path(__file__).with_name("app.py")


def main() -> None:
    from src.web_app import warmup

    warmup.start()

    from streamlit.web import cli

    sys.argv = ["streamlit", "run", str(_APP), *sys.argv[1:]]
    sys.exit(cli.main())


if __name__ == "__main__":
    main()
