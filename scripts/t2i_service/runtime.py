from __future__ import annotations

import faulthandler
import logging
import signal
import sys


logger = logging.getLogger("t2i_service.server")


def install_runtime_hooks() -> None:
    faulthandler.enable(all_threads=True, file=sys.stderr)

    def _handle_signal(signum: int, _frame: object) -> None:
        name = signal.Signals(signum).name
        logger.warning("Received %s, process will shut down", name)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, _handle_signal)
