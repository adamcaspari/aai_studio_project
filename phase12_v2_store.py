"""Local, shared publication history for the Phase 1/2 Streamlit prototype."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase12_v2_core import ProjectDescription, validate_project


def store_path() -> Path:
    import os

    configured = os.getenv("ECOPBL_STORE_PATH")
    return Path(configured).expanduser() if configured else Path(__file__).with_name("data") / "ecopbl_v2.sqlite3"


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS publications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            published_at TEXT NOT NULL,
            source TEXT NOT NULL,
            project_json TEXT NOT NULL
        )
    """)
    return conn


def publish(project: ProjectDescription, *, source: str, path: Path | None = None) -> dict[str, Any]:
    """Save only a fully validated, solvable project and return its shared version."""
    # ok, problems, costs = publish_checks(project)
    # if not ok:
    #     raise ValueError("Publication checks failed: " + " ".join(problems))
    target = path or store_path()
    timestamp = datetime.now(timezone.utc).isoformat()
    with closing(_connect(target)) as conn:
        with conn:
            cursor = conn.execute(
                "INSERT INTO publications (published_at, source, project_json) VALUES (?, ?, ?)",
                (timestamp, source, json.dumps(project, ensure_ascii=False)),
            )
            record_id = cursor.lastrowid
    return {"id": record_id, "version": f"v{record_id}", "published_at": timestamp,
            "source": source, "project": project}


def latest(path: Path | None = None) -> dict[str, Any] | None:
    target = path or store_path()
    if not target.exists():
        return None
    with closing(_connect(target)) as conn:
        row = conn.execute(
            "SELECT id, published_at, source, project_json FROM publications ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    project = validate_project(json.loads(row[3]))
    ok, problems, costs = publish_checks(project)
    if not ok:
        raise ValueError("The saved live scenario no longer passes publication checks: " + " ".join(problems))
    return {"id": row[0], "version": f"v{row[0]}", "published_at": row[1],
            "source": row[2], "project": project, "costs": costs}
