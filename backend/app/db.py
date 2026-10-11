"""SQLite persistence. One file, no server, safe for a single-depot deployment."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date, datetime
from pathlib import Path

from .models import AuditEntry, BrandingContract, Certificate, Fleet, JobCard, Plan, Train
from .seed import generate_fleet

SCHEMA = """
CREATE TABLE IF NOT EXISTS trains (
    id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, commissioned TEXT NOT NULL,
    total_km INTEGER NOT NULL, km_since_maintenance INTEGER NOT NULL,
    last_deep_clean TEXT NOT NULL, cleaning_flagged INTEGER NOT NULL DEFAULT 0,
    stabling_track TEXT NOT NULL, stabling_position INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS certificates (
    id INTEGER PRIMARY KEY, train_id INTEGER NOT NULL REFERENCES trains(id),
    department TEXT NOT NULL, issued TEXT NOT NULL, expires TEXT NOT NULL,
    UNIQUE(train_id, department)
);
CREATE TABLE IF NOT EXISTS job_cards (
    id INTEGER PRIMARY KEY, train_id INTEGER NOT NULL REFERENCES trains(id),
    system TEXT NOT NULL, description TEXT NOT NULL, priority TEXT NOT NULL,
    status TEXT NOT NULL, opened TEXT NOT NULL, closed TEXT
);
CREATE TABLE IF NOT EXISTS branding (
    id INTEGER PRIMARY KEY, train_id INTEGER NOT NULL REFERENCES trains(id),
    advertiser TEXT NOT NULL, target_hours INTEGER NOT NULL, delivered_hours REAL NOT NULL,
    period_start TEXT NOT NULL, period_end TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, actor TEXT NOT NULL,
    via TEXT NOT NULL, action TEXT NOT NULL, detail TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS approved_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT, plan_date TEXT NOT NULL, approved_at TEXT NOT NULL,
    approved_by TEXT NOT NULL, plan_json TEXT NOT NULL
);
"""


class Database:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        if self._conn.execute("SELECT COUNT(*) FROM trains").fetchone()[0] == 0:
            self.reset()

    # ------------------------------------------------------------ fleet I/O
    def load_fleet(self) -> Fleet:
        with self._lock:
            q = self._conn.execute
            return Fleet(
                trains=[Train(**dict(r)) for r in q("SELECT * FROM trains ORDER BY id")],
                certificates=[Certificate(**dict(r)) for r in q("SELECT * FROM certificates ORDER BY id")],
                job_cards=[JobCard(**dict(r)) for r in q("SELECT * FROM job_cards ORDER BY id")],
                branding=[BrandingContract(**dict(r)) for r in q("SELECT * FROM branding ORDER BY id")],
            )

    def save_fleet(self, fleet: Fleet) -> None:
        """Replace all fleet tables with `fleet` in one transaction."""
        with self._lock, self._conn:
            c = self._conn
            for table in ("branding", "job_cards", "certificates", "trains"):
                c.execute(f"DELETE FROM {table}")
            for table, rows in (
                ("trains", fleet.trains), ("certificates", fleet.certificates),
                ("job_cards", fleet.job_cards), ("branding", fleet.branding),
            ):
                for row in rows:
                    data = row.model_dump(mode="json")
                    cols = ", ".join(data)
                    marks = ", ".join("?" for _ in data)
                    c.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(data.values()))

    def reset(self, today: date | None = None) -> None:
        self.save_fleet(generate_fleet(today))
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM audit_log")
            self._conn.execute("DELETE FROM approved_plans")
        self.audit("system", "system", "seed", "Fleet database initialised with demo data")

    # ---------------------------------------------------------------- audit
    def audit(self, actor: str, via: str, action: str, detail: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO audit_log (ts, actor, via, action, detail) VALUES (?, ?, ?, ?, ?)",
                (datetime.now().isoformat(timespec="seconds"), actor, via, action, detail),
            )

    def audit_entries(self, limit: int = 100) -> list[AuditEntry]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))
            return [AuditEntry(**dict(r)) for r in rows]

    # ---------------------------------------------------------------- plans
    def approve_plan(self, plan: Plan, approved_by: str) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO approved_plans (plan_date, approved_at, approved_by, plan_json) VALUES (?, ?, ?, ?)",
                (plan.plan_date.isoformat(), datetime.now().isoformat(timespec="seconds"), approved_by, plan.model_dump_json()),
            )
            return int(cur.lastrowid)

    def approved_plans(self, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, plan_date, approved_at, approved_by, plan_json FROM approved_plans ORDER BY id DESC LIMIT ?", (limit,)
            )
            out = []
            for r in rows:
                plan = json.loads(r["plan_json"])
                out.append({
                    "id": r["id"], "plan_date": r["plan_date"], "approved_at": r["approved_at"],
                    "approved_by": r["approved_by"], "kpis": plan["kpis"],
                })
            return out
