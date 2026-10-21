from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from app.services.agent import AgentLimits, ProfileAgent
from app.services.retrieval import Document, RetrievalIndex

# --- fakes -------------------------------------------------------------------


def _doc(doc_id, category, title, text):
    return Document(id=doc_id, category=category, title=title, text=text, embedding=None)


@pytest.fixture
def index():
    docs = [
        _doc("role:data-equity", "role", "Data Scientist at Data Equity",
             "Domingo Berbel has been Data Scientist at Data Equity since October 2025."),
        _doc("role:suministros-medina", "role", "International Sales Representative",
             "He was International Sales Representative at Suministros Medina until September 2025."),
        _doc("project:portfolio-chatbot", "project", "Portfolio RAG assistant",
             "A retrieval augmented assistant built with FastAPI and React on Azure."),
        _doc("languages:spoken", "languages", "Spoken languages",
             "Spanish native. English professional working proficiency."),
    ]
    return RetrievalIndex(docs, vocabulary=frozenset({"Data Equity", "Suministros Medina", "FastAPI", "React", "Azure"}))


def _tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls, role="assistant")


def _response(message):
    """A scripted turn as a Responses API result: function calls, or text."""
    output = [
        {
            "type": "function_call",
            "call_id": tc.id,
            "name": tc.function.name,
            "arguments": tc.function.arguments,
        }
        for tc in (message.tool_calls or [])
    ]
    if message.content:
        output.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": message.content}],
            }
        )
    return SimpleNamespace(output=output, output_text=message.content or "")


def _as_events(response, size=7):
    """A scripted turn as the event stream the real API sends: text a few
    characters at a time, then the completed response carrying every item."""
    text = response.output_text
    events = [
        SimpleNamespace(type="response.output_text.delta", delta=text[i : i + size])
        for i in range(0, len(text), size)
    ]
    events.append(SimpleNamespace(type="response.completed", response=response))
    return iter(events)


class FakeClient:
    """Scripted OpenAI client. Each scripted item is one assistant turn.

    When called without `tools` it always returns prose, because a real model
    cannot emit a tool call for a tool it was not offered. Without this the
    agent's final tools-off call would be answered with another tool call.
    """

    FORCED_TEXT = "(answered without tools)"

    def __init__(self, script):
        self._script = list(script)
        self.calls = []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        response = _response(self._respond(**kwargs))
        if kwargs.get("stream"):
            return _as_events(response)
        return response

    def _respond(self, **kwargs):
        self.calls.append(kwargs)
        if "tools" not in kwargs:
            nxt = self._script.pop(0) if self._script else None
            if nxt is not None and not getattr(nxt, "tool_calls", None):
                return nxt
            return _message(content=self.FORCED_TEXT)
        if not self._script:
            return _message(content="(script exhausted)")
        return self._script.pop(0)


def executed_tool_call_ids(client):
    """Unique call_ids that produced real output.

    The input list accumulates and is resent on every call, so summing tool
    outputs across calls counts each one many times over.
    """
    ids = set()
    for call in client.calls:
        for item in call["input"]:
            if item.get("type") != "function_call_output":
                continue
            if "budget exhausted" in item["output"]:
                continue
            ids.add(item["call_id"])
    return ids


def _agent(index, script, **limit_kwargs):
    limits = AgentLimits(**limit_kwargs) if limit_kwargs else AgentLimits()
    return ProfileAgent(
        client=FakeClient(script),
        model="test-model",
        index=index,
        embedder=lambda _text: None,
        limits=limits,
    )


# --- tool dispatch -----------------------------------------------------------


def test_a_search_tool_call_is_dispatched_and_answered(index):
    agent = _agent(
        index,
        [
            _message(tool_calls=[_tool_call("c1", "search_profile", {"query": "current job"})]),
            _message(content="He is a Data Scientist at Data Equity."),
        ],
    )
    result = agent.answer("Where does he work?", history=[], language="en")
    assert "Data Equity" in result.answer
    assert result.documents


def test_get_entity_returns_the_addressed_document(index):
    agent = _agent(
        index,
        [
            _message(tool_calls=[_tool_call("c1", "get_entity", {"id": "project:portfolio-chatbot"})]),
            _message(content="He built a portfolio RAG assistant with FastAPI."),
        ],
    )
    result = agent.answer("Tell me about the chatbot", history=[], language="en")
    assert any(d.id == "project:portfolio-chatbot" for d in result.documents)


def test_list_entities_is_available_to_the_model(index):
    agent = _agent(
        index,
        [
            _message(tool_calls=[_tool_call("c1", "list_entities", {})]),
            _message(content="He has several roles and projects."),
        ],
    )
    result = agent.answer("What do you know?", history=[], language="en")
    assert result.answer


def test_category_filter_is_passed_through(index):
    agent = _agent(
        index,
        [
            _message(tool_calls=[_tool_call("c1", "search_profile", {"query": "x", "category": "project"})]),
            _message(content="Only the portfolio project matched."),
        ],
    )
    agent.answer("projects?", history=[], language="en")
    # The prefetch searched every category, so check the filtered call's output.
    output = next(
        item["output"]
        for item in agent.client.calls[-1]["input"]
        if item.get("type") == "function_call_output" and item["call_id"] == "c1"
    )
    assert "[project:portfolio-chatbot]" in output
    assert "[role:" not in output


def test_multi_hop_issues_a_second_search(index):
    agent = _agent(
        index,
        [
            _message(tool_calls=[_tool_call("c1", "search_profile", {"query": "current role"})]),
            _message(tool_calls=[_tool_call("c2", "search_profile", {"query": "languages"})]),
            _message(content="He works at Data Equity and speaks Spanish and English."),
        ],
    )
    result = agent.answer("Job and languages?", history=[], language="en")
    ids = {d.id for d in result.documents}
    assert len(ids) >= 2


# --- robustness --------------------------------------------------------------


def test_malformed_tool_arguments_are_recoverable(index):
    broken = SimpleNamespace(
        id="c1",
        type="function",
        function=SimpleNamespace(name="search_profile", arguments="{not json"),
    )
    agent = _agent(
        index,
        [_message(tool_calls=[broken]), _message(content="Recovered and answered.")],
    )
    result = agent.answer("anything", history=[], language="en")
    assert "Recovered" in result.answer


def test_unknown_tool_name_is_recoverable(index):
    agent = _agent(
        index,
        [
            _message(tool_calls=[_tool_call("c1", "delete_everything", {})]),
            _message(content="I could not use that tool, but here is the answer."),
        ],
    )
    result = agent.answer("anything", history=[], language="en")
    assert result.answer


def test_unknown_entity_id_is_recoverable(index):
    agent = _agent(
        index,
        [
            _message(tool_calls=[_tool_call("c1", "get_entity", {"id": "role:nope"})]),
            _message(content="That entity does not exist; here is what does."),
        ],
    )
    result = agent.answer("anything", history=[], language="en")
    assert result.answer


def test_the_iteration_cap_terminates_a_looping_model(index):
    """A model that only ever calls tools must not spin forever."""
    forever = [
        _message(tool_calls=[_tool_call(f"c{i}", "search_profile", {"query": "x"})])
        for i in range(50)
    ]
    agent = _agent(index, forever, max_iterations=3)
    result = agent.answer("anything", history=[], language="en")
    assert result.answer  # some answer is produced rather than hanging
    assert agent.client.calls, "the model was called"
    assert len(agent.client.calls) <= 4


def test_the_tool_call_cap_is_enforced(index):
    forever = [
        _message(tool_calls=[_tool_call(f"c{i}", "search_profile", {"query": "x"})])
        for i in range(50)
    ]
    agent = _agent(index, forever, max_iterations=20, max_tool_calls=2)
    agent.answer("anything", history=[], language="en")
    assert len(executed_tool_call_ids(agent.client)) == 2


# --- scope and grounding -----------------------------------------------------


def test_out_of_scope_is_detected(index):
    agent = _agent(index, [_message(content="OUT_OF_SCOPE")])
    result = agent.answer("Hala Madrid", history=[], language="en")
    assert result.out_of_scope


def test_a_fabricated_entity_triggers_one_regeneration(index):
    agent = _agent(
        index,
        [
            _message(tool_calls=[_tool_call("c1", "search_profile", {"query": "stack"})]),
            _message(content="He uses MongoDB in production."),
            _message(content="He uses FastAPI and React on Azure."),
        ],
    )
    result = agent.answer("What does he use?", history=[], language="en")
    assert result.regenerated
    assert "MongoDB" not in result.answer


def test_a_grounded_answer_is_not_regenerated(index):
    agent = _agent(
        index,
        [
            _message(tool_calls=[_tool_call("c1", "search_profile", {"query": "stack"})]),
            _message(content="He uses FastAPI and React on Azure."),
        ],
    )
    result = agent.answer("What does he use?", history=[], language="en")
    assert not result.regenerated


def test_history_is_forwarded_to_the_model(index):
    agent = _agent(index, [_message(content="Yes, as I said.")])
    agent.answer(
        "and that?",
        history=[{"role": "user", "content": "does he know spark"},
                 {"role": "assistant", "content": "yes"}],
        language="en",
    )
    roles = [m.get("role") for m in agent.client.calls[0]["input"]]
    assert roles.count("user") >= 2


@pytest.mark.parametrize("language,order", [("es", "Answer in Spanish"), ("en", "Answer in English.")])
def test_the_resolved_language_is_an_order_not_a_hint(index, language, order):
    """Left to judge the language itself, the model answered 'Power BI' in
    Spanish on the English site in four runs out of four."""
    agent = _agent(index, [_message(content="ok")])
    agent.answer("PySpark", history=[], language=language)
    system = agent.client.calls[0]["input"][0]["content"]
    assert order in system


def test_reasoning_is_requested_for_reasoning_models(index):
    agent = ProfileAgent(
        client=FakeClient([_message(content="ok")]),
        model="gpt-5.6-luna",
        index=index,
        embedder=lambda _text: None,
        reasoning_effort="low",
    )
    agent.answer("anything", history=[], language="en")
    call = agent.client.calls[0]
    assert call["reasoning"] == {"effort": "low"}
    assert "temperature" not in call
    # Stateless calls must carry reasoning back with the next turn's input.
    assert call["store"] is False
    assert call["include"] == ["reasoning.encrypted_content"]


def test_the_system_prompt_does_not_hardcode_grades(index):
    """Facts belong in the corpus. The old prompt hardcoded exact scores."""
    agent = _agent(index, [_message(content="ok")])
    agent.answer("grades?", history=[], language="en")
    system = agent.client.calls[0]["input"][0]["content"]
    for leak in ("9.75", "9.20", "Matrícula", "Apache Spark 9"):
        assert leak not in system


# --- prefetch ------------------------------------------------------------------


def test_the_question_is_searched_before_the_first_call(index):
    """Deciding to search for the question used to cost a full round trip."""
    agent = _agent(index, [_message(content="He is a Data Scientist at Data Equity.")])
    result = agent.answer("Where does he work now?", history=[], language="en")
    assert len(agent.client.calls) == 1
    first = agent.client.calls[0]["input"]
    assert first[-1]["type"] == "function_call_output"
    assert first[-2]["name"] == "search_profile"
    assert result.documents


def test_the_prefetch_counts_against_the_tool_call_budget(index):
    forever = [
        _message(tool_calls=[_tool_call(f"c{i}", "search_profile", {"query": "x"})])
        for i in range(50)
    ]
    agent = _agent(index, forever, max_iterations=20, max_tool_calls=1)
    agent.answer("anything", history=[], language="en")
    assert executed_tool_call_ids(agent.client) == {"call_prefetch"}


# --- streaming -----------------------------------------------------------------


def _stream(agent, question, language="en"):
    items = list(agent.answer_stream(question, history=[], language=language))
    *pieces, result = items
    return pieces, result


def test_the_answer_is_streamed_in_pieces(index):
    answer = "He is a Data Scientist at Data Equity. He works with FastAPI and React."
    agent = _agent(index, [_message(content=answer)])
    pieces, result = _stream(agent, "Where does he work?")
    assert len(pieces) >= 2, "text must leave before the whole answer exists"
    assert "".join(pieces) == answer
    assert result.answer == answer
    assert agent.client.calls[0]["stream"] is True


def test_a_streamed_tool_call_is_executed(index):
    agent = _agent(
        index,
        [
            _message(tool_calls=[_tool_call("c1", "get_entity", {"id": "languages:spoken"})]),
            _message(content="He is a Spanish native with professional English."),
        ],
    )
    pieces, result = _stream(agent, "Languages?")
    assert any(d.id == "languages:spoken" for d in result.documents)
    assert "".join(pieces) == "He is a Spanish native with professional English."


def test_a_fabricated_sentence_is_never_streamed(index):
    agent = _agent(
        index,
        [_message(content="He works at Data Equity. He also uses MongoDB daily. He builds with FastAPI.")],
    )
    pieces, result = _stream(agent, "What does he use?")
    streamed = "".join(pieces)
    assert "MongoDB" not in streamed
    assert "Data Equity" in streamed and "FastAPI" in streamed
    assert len(agent.client.calls) == 1, "dropping the sentence needs no second call"


def test_a_fully_fabricated_answer_is_regenerated(index):
    agent = _agent(
        index,
        [
            _message(content="He uses MongoDB."),
            _message(content="He uses FastAPI and React on Azure."),
        ],
    )
    pieces, result = _stream(agent, "What does he use?")
    assert "".join(pieces) == "He uses FastAPI and React on Azure."
    assert result.regenerated


def test_out_of_scope_is_never_streamed(index):
    agent = _agent(index, [_message(content="OUT_OF_SCOPE\n")])
    pieces, result = _stream(agent, "Hala Madrid")
    assert pieces == []
    assert result.out_of_scope


def test_the_streamed_loop_respects_the_iteration_cap(index):
    forever = [
        _message(tool_calls=[_tool_call(f"c{i}", "search_profile", {"query": "x"})])
        for i in range(50)
    ]
    agent = _agent(index, forever, max_iterations=3)
    pieces, result = _stream(agent, "anything")
    assert result.answer
    assert len(agent.client.calls) <= 4


def test_the_refusal_follows_the_language_the_model_read(index):
    """'Hala Madrid' has no marker the heuristic can use; the model's does."""
    agent = _agent(index, [_message(content="OUT_OF_SCOPE es")])
    result = agent.answer("Hala Madrid", history=[], language="en")
    assert result.out_of_scope and result.language == "es"

    agent = _agent(index, [_message(content="OUT_OF_SCOPE es")])
    pieces, streamed = _stream(agent, "Hala Madrid")
    assert pieces == [] and streamed.out_of_scope and streamed.language == "es"


def test_every_document_shown_to_the_model_counts_as_support():
    """Grounding used to stop collecting at eight documents while the model
    kept reading, so a correct sentence built from the ninth was dropped."""
    docs = [
        _doc(f"project:p{i}", "project", f"Project {i}", f"Built with Tool{i} in 2025.")
        for i in range(12)
    ]
    big = RetrievalIndex(docs, vocabulary=frozenset({"Data Equity"}))
    agent = _agent(
        big,
        [
            _message(tool_calls=[_tool_call("c1", "list_entities", {})]),
            _message(tool_calls=[_tool_call(f"g{i}", "get_entity", {"id": f"project:p{i}"}) for i in range(12)]),
            _message(content="He used Tool11 on one project."),
        ],
        max_tool_calls=20,
    )
    pieces, result = _stream(agent, "What tools?")
    assert "".join(pieces) == "He used Tool11 on one project."
