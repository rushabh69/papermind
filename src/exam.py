"""Practice-mode exam generation + written-answer grading (via Groq).

For generation we deliberately sample chunks spread ACROSS the document (not the
top-matched ones) so questions cover the whole thing. Difficulty is steered by a
per-level instruction in the prompt.
"""
from __future__ import annotations

import numpy as np

from . import config, db, llm


DIFFICULTY_GUIDE = {
    "Easy": "direct fact recall - definitions, stated numbers, dates, or facts "
            "written explicitly in the text.",
    "Medium": "require the student to connect two or more pieces of information "
              "from different parts of the text.",
    "Hard": "require inference, comparison, or analysis that goes beyond what is "
            "explicitly stated (but still grounded in the text).",
}


def _sample_chunks(document_id: int, k: int) -> list[dict]:
    """Evenly spread chunks across the document for good coverage."""
    chunks = db.get_chunks(document_id)
    if len(chunks) <= k:
        return chunks
    idxs = sorted(set(np.linspace(0, len(chunks) - 1, k).round().astype(int).tolist()))
    return [chunks[i] for i in idxs]


def generate_quiz(document_id: int, difficulty: str, qtype: str, n: int) -> tuple[int, list[dict]]:
    """Generate + persist a quiz set. Returns (quiz_set_id, questions)."""
    n_context = max(6, min(2 * n, 14))
    sampled = _sample_chunks(document_id, n_context)
    context = "\n\n".join(
        f"[page {c['page_number']}] {c['chunk_text']}" for c in sampled)

    if qtype == "MCQ":
        type_instr = ('Every question must be multiple choice with exactly 4 options. '
                      'Set "type" to "MCQ", "options" to the 4 choices, and '
                      '"correct_answer" to the exact text of the correct option.')
    elif qtype == "Written":
        type_instr = ('Every question must be open-ended/written. Set "type" to '
                      '"Written", "options" to null, and "correct_answer" to a concise '
                      'model answer.')
    else:  # Mixed
        type_instr = ('Mix MCQ and Written questions. For MCQ set "type"="MCQ" with 4 '
                      '"options" and "correct_answer" = the correct option text. For '
                      'Written set "type"="Written", "options"=null, "correct_answer" = '
                      'a concise model answer.')

    system = (
        "You are an exam writer. Create quiz questions grounded ONLY in the provided "
        "text. Never ask about things not in the text. Every question must include the "
        "source page it comes from."
    )
    user = (
        f"Source text (with page numbers):\n{context}\n\n"
        f"Write {n} {difficulty} questions. Difficulty means: {DIFFICULTY_GUIDE[difficulty]}\n"
        f"{type_instr}\n"
        'Respond as JSON: {"questions": [{"type": "MCQ"|"Written", "question": str, '
        '"options": [str, str, str, str] or null, "correct_answer": str, '
        '"explanation": str, "source_page": int}]}'
    )
    data = llm.chat_json([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], max_tokens=3000)

    raw = data.get("questions", []) if isinstance(data, dict) else []
    questions = []
    for q in raw:
        if not q.get("question"):
            continue
        questions.append({
            "type": q.get("type", qtype if qtype != "Mixed" else "Written"),
            "difficulty": difficulty,
            "question": str(q["question"]).strip(),
            "options": q.get("options") if q.get("options") else None,
            "correct_answer": str(q.get("correct_answer", "")).strip(),
            "explanation": str(q.get("explanation", "")).strip(),
            "source_page": q.get("source_page"),
        })

    set_id = db.add_quiz_set(document_id, difficulty, qtype, questions)
    return set_id, questions


def grade_written(question: str, model_answer: str, user_answer: str) -> dict:
    """Grade a written answer against the model answer. Returns {score, verdict, feedback}."""
    if not user_answer.strip():
        return {"score": 0, "verdict": "No answer", "feedback": "You didn't write anything."}
    system = ("You are a fair but rigorous grader. Compare the student's answer to the "
              "model answer and grade how well it captures the key points.")
    user = (
        f"Question: {question}\n\nModel answer: {model_answer}\n\n"
        f"Student answer: {user_answer}\n\n"
        'Respond as JSON: {"score": int 0-100, "verdict": "Correct"|"Partially correct"'
        '|"Incorrect", "feedback": str explaining what was right or missing}'
    )
    data = llm.chat_json([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], model=config.GROQ_MODEL_FAST)
    return {
        "score": int(data.get("score", 0)),
        "verdict": str(data.get("verdict", "")),
        "feedback": str(data.get("feedback", "")),
    }