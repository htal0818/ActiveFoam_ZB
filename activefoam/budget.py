"""Wall-clock limit for a single simulation run.

Tangled low-adhesion configurations can slow a run to a crawl; the sweep
drivers use this to give such a run up and record it as invalid.
"""
import signal


class BudgetExceeded(BaseException):
    """Not an Exception subclass, so the model's revert guards cannot swallow it."""


def call_with_budget(seconds, fn, *args, **kwargs):
    def _handler(signum, frame):
        raise BudgetExceeded(f"run exceeded {seconds} s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
