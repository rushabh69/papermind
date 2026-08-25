"""SQLite persistence: documents, chunks, chat history, sources, quizzes, scores.

Everything the app must remember across restarts lives here. Chunk *text* is
stored here too (so BM25 + citation snippets work); the FAISS index only holds
the embedding vectors keyed by chunk id.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from . import config


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    filename     TEXT NOT NULL,
    upload_date  TEXT NOT NULL,
    total_pages  INTEGER NOT NULL,
    sections     TEXT,                    -- JSON list of detected section titles
    n_chunks     INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,
    page_number   INTEGER NOT NULL,
    section_title TEXT,
    chunk_text    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);

CREATE TABLE IF NOT EXISTS conversations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    question     TEXT NOT NULL,
    answer       TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conv_doc ON conversations(document_id);

CREATE TABLE IF NOT EXISTS sources (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    page_number      INTEGER,
    section          TEXT,
    snippet_text     TEXT
);

CREATE TABLE IF NOT EXISTS quiz_sets (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    difficulty   TEXT NOT NULL,
    qtype        TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quiz_questions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    quiz_set_id    INTEGER NOT NULL REFERENCES quiz_sets(id) ON DELETE CASCADE,
    document_id    INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    difficulty     TEXT NOT NULL,
    qtype          TEXT NOT NULL,
    question       TEXT NOT NULL,
    options        TEXT,                  -- JSON list for MCQ, null for written
    correct_answer TEXT,
    explanation    TEXT,
    source_page    INTEGER
);

CREATE TABLE IF NOT EXISTS practice_sessions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    quiz_set_id  INTEGER REFERENCES quiz_sets(id) ON DELETE SET NULL,
    difficulty   TEXT,
    qtype        TEXT,
    score        REAL,
    total        INTEGER,
    created_at   TEXT NOT NULL
);
"""


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# --- documents ---------------------------------------------------------
def add_document(filename: str, total_pages: int, sections: list[str]) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO documents (filename, upload_date, total_pages, sections) "
            "VALUES (?, ?, ?, ?)",
            (filename, _now(), total_pages, json.dumps(sections)),
        )
        return cur.lastrowid


def list_documents() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def get_document(document_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    return dict(row) if row else None


def delete_document(document_id: int) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))


# --- chunks ------------------------------------------------------------
def add_chunks(document_id: int, chunks: list[dict]) -> list[int]:
    """Insert chunk dicts (chunk_index, page_number, section_title, chunk_text).

    Returns the assigned chunk ids in the same order (used to align FAISS rows).
    """
    ids = []
    with get_conn() as conn:
        for c in chunks:
            cur = conn.execute(
                "INSERT INTO chunks (document_id, chunk_index, page_number, "
                "section_title, chunk_text) VALUES (?, ?, ?, ?, ?)",
                (document_id, c["chunk_index"], c["page_number"],
                 c.get("section_title"), c["chunk_text"]),
            )
            ids.append(cur.lastrowid)
        conn.execute("UPDATE documents SET n_chunks = ? WHERE id = ?",
                     (len(chunks), document_id))
    return ids


def get_chunks(document_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (document_id,)).fetchall()
    return [dict(r) for r in rows]


def get_chunks_by_ids(chunk_ids: list[int]) -> dict[int, dict]:
    if not chunk_ids:
        return {}
    q = "SELECT * FROM chunks WHERE id IN (%s)" % ",".join("?" * len(chunk_ids))
    with get_conn() as conn:
        rows = conn.execute(q, chunk_ids).fetchall()
    return {r["id"]: dict(r) for r in rows}


# --- chat history ------------------------------------------------------
def add_conversation(document_id: int, question: str, answer: str,
                     sources: list[dict]) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO conversations (document_id, question, answer, created_at) "
            "VALUES (?, ?, ?, ?)",
            (document_id, question, answer, _now()),
        )
        conv_id = cur.lastrowid
        for s in sources:
            conn.execute(
                "INSERT INTO sources (conversation_id, page_number, section, "
                "snippet_text) VALUES (?, ?, ?, ?)",
                (conv_id, s.get("page"), s.get("section"), s.get("snippet")),
            )
        return conv_id


def get_conversations(document_id: int) -> list[dict]:
    with get_conn() as conn:
        convs = conn.execute(
            "SELECT * FROM conversations WHERE document_id = ? ORDER BY id",
            (document_id,)).fetchall()
        out = []
        for c in convs:
            srcs = conn.execute(
                "SELECT page_number, section, snippet_text FROM sources "
                "WHERE conversation_id = ?", (c["id"],)).fetchall()
            d = dict(c)
            d["sources"] = [dict(s) for s in srcs]
            out.append(d)
    return out


# --- quizzes -----------------------------------------------------------
def add_quiz_set(document_id: int, difficulty: str, qtype: str,
                 questions: list[dict]) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO quiz_sets (document_id, difficulty, qtype, created_at) "
            "VALUES (?, ?, ?, ?)", (document_id, difficulty, qtype, _now()))
        set_id = cur.lastrowid
        for q in questions:
            conn.execute(
                "INSERT INTO quiz_questions (quiz_set_id, document_id, difficulty, "
                "qtype, question, options, correct_answer, explanation, source_page) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (set_id, document_id, q.get("difficulty", difficulty),
                 q.get("type", qtype), q["question"],
                 json.dumps(q.get("options")) if q.get("options") else None,
                 q.get("correct_answer"), q.get("explanation"),
                 q.get("source_page")),
            )
        return set_id


def get_quiz_questions(quiz_set_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM quiz_questions WHERE quiz_set_id = ? ORDER BY id",
            (quiz_set_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["options"] = json.loads(d["options"]) if d["options"] else None
        out.append(d)
    return out


def list_quiz_sets(document_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT qs.*, COUNT(q.id) AS n_questions FROM quiz_sets qs "
            "LEFT JOIN quiz_questions q ON q.quiz_set_id = qs.id "
            "WHERE qs.document_id = ? GROUP BY qs.id ORDER BY qs.id DESC",
            (document_id,)).fetchall()
    return [dict(r) for r in rows]


def add_practice_session(document_id: int, quiz_set_id: int, difficulty: str,
                         qtype: str, score: float, total: int) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO practice_sessions (document_id, quiz_set_id, difficulty, "
            "qtype, score, total, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (document_id, quiz_set_id, difficulty, qtype, score, total, _now()))
        return cur.lastrowid


def get_practice_sessions(document_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM practice_sessions WHERE document_id = ? ORDER BY id DESC",
            (document_id,)).fetchall()
    return [dict(r) for r in rows]