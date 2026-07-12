"""
LangGraph checkpoint manager — persists agent state to SQLite.

DESIGN DECISION (v0.3.0): Checkpointing is DISABLED by default.

Why: AsyncSqliteSaver.from_conn_string() is a @contextmanager — it must be
entered with `async with` and kept open for the process lifetime. The sync
SqliteSaver doesn't support LangGraph's async `ainvoke()`. Getting this right
requires an async lifespan wrapper that complicates the FastAPI startup.

For a hackathon, jobs complete in seconds — crash recovery is not critical.
The graph compiles WITHOUT a checkpointer; jobs run normally, they just don't
persist across API restarts. This is the pragmatic trade-off.

To re-enable: wrap the AsyncSqliteSaver context in an async startup/shutdown
handler and pass it to workflow.compile(checkpointer=saver).
"""
from __future__ import annotations

from loguru import logger


def get_checkpointer():
    """Return None — checkpointing disabled in v0.3.0.

    The graph compiles without a checkpointer. Jobs run normally but do not
    persist across API restarts. See module docstring for rationale.
    """
    logger.info("Checkpointing disabled (v0.3.0) — jobs run without persistence")
    return None


async def close_checkpointer():
    """No-op — checkpointing disabled."""
    pass
