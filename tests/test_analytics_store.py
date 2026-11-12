"""The question log must survive the container.

The bug this guards against: logs lived only in /tmp, so a redeploy emptied
them and the admin endpoints answered 200 with zero items. Every question is
now also written to stdout as a `QA_LOG {json}` line, which Azure Container
Apps ships to Log Analytics, and `scripts/sync_preguntas.py` reads it back.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import sync_preguntas  # noqa: E402
from app.services import analytics_store  # noqa: E402
from app.services.analytics_store import QA_LOG_PREFIX, AnalyticsStore  # noqa: E402


def make_store(tmp_path, name: str = "s") -> AnalyticsStore:
    return AnalyticsStore(
        db_path=str(tmp_path / f"{name}.db"), json_path=str(tmp_path / f"{name}.json")
    )


def qa_lines(capsys) -> list[str]:
    return [l for l in capsys.readouterr().out.splitlines() if l.startswith(QA_LOG_PREFIX)]


def test_every_question_is_written_to_stdout_as_one_json_line(tmp_path, capsys):
    make_store(tmp_path).log_question(
        "1.2.3.4", "pytest", "¿Qué hace Domingo?", "Línea uno.\nLínea dos.", out_of_scope=False
    )
    lines = qa_lines(capsys)
    assert len(lines) == 1, "una pregunta, una línea física (los saltos van escapados)"
    record = json.loads(lines[0][len(QA_LOG_PREFIX):])
    assert record["question"] == "¿Qué hace Domingo?"
    assert record["answer"] == "Línea uno.\nLínea dos."
    assert record["client_ip"] == "1.2.3.4"
    assert record["out_of_scope"] is False
    assert record["timestamp"] and record["date"] and record["time"]


def test_the_stdout_line_does_not_depend_on_the_local_files(tmp_path, capsys, monkeypatch):
    """If /tmp is unwritable the durable copy must still go out."""
    store = make_store(tmp_path)

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(store, "_connect", broken)
    try:
        store.log_question("ip", None, "q", "a", out_of_scope=True)
    except OSError:
        pass
    assert len(qa_lines(capsys)) == 1


def test_a_very_long_answer_stays_under_the_console_line_limit(tmp_path, capsys):
    make_store(tmp_path).log_question("ip", None, "q", "x" * 50_000, out_of_scope=False)
    (line,) = qa_lines(capsys)
    assert len(line.encode("utf-8")) < 16 * 1024
    json.loads(line[len(QA_LOG_PREFIX):])


def test_emitting_never_raises(monkeypatch):
    monkeypatch.setattr(analytics_store.json, "dumps", lambda *a, **k: 1 / 0)
    analytics_store.emit_qa_log({"question": "q"})   # must not raise


def test_sync_parses_the_lines_log_analytics_returns(tmp_path, capsys):
    """Round trip: what the backend prints is what the sync script reads."""
    make_store(tmp_path).log_question("ip", "ua", "pregunta", "respuesta", out_of_scope=False)
    (line,) = qa_lines(capsys)
    rows = [
        {"Log_s": line},
        {"Log_s": "INFO uvicorn.access GET /health 200"},
        {"Log_s": QA_LOG_PREFIX + "{json roto"},
    ]
    items = sync_preguntas.parse_log_analytics_rows(rows)
    assert [(i["question"], i["answer"]) for i in items] == [("pregunta", "respuesta")]


def test_sync_merge_does_not_duplicate_an_entry_seen_in_both_sources(tmp_path, capsys):
    make_store(tmp_path).log_question("ip", "ua", "pregunta", "respuesta", out_of_scope=False)
    (line,) = qa_lines(capsys)
    from_logs = sync_preguntas.parse_log_analytics_rows([{"Log_s": line}])
    merged = sync_preguntas.merge_items(from_logs, from_logs)
    merged = sync_preguntas.merge_items(merged, from_logs)
    assert len(merged) == 1
