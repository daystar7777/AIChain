"""Errors shared by the chain log and its users (kept tiny to avoid import cycles)."""


class ChainHalted(Exception):
    """Writing the chain log failed. The network stops acting until it is restarted, because
    in-memory state may now be ahead of what is recorded (fail closed)."""
