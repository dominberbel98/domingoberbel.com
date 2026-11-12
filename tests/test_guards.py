"""Rate limiting, the token budget and the visitor's address.

Three defects found in production on 2026-11-12, each guarded here:

* every logged question carried a 100.100.x.x address — the Container Apps
  ingress — so the per-IP rate limit was one bucket for the whole internet;
* the daily token budget counted `len(question) // 3`, about 100 per question,
  while a real answer bills thousands;
* the pre-LLM content filter matched substrings and refused ordinary
  professional questions ("artículo" contains "culo").
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.config import settings
from app.services.agent import AgentLimits, ProfileAgent
from app.services.analytics_store import AnalyticsStore
from app.services.client_ip import client_ip
from app.services.guards import RequestGuards
from app.services.rag_service import (
    is_contact_request,
    is_greeting,
    is_inappropriate,
)
from app.services.retrieval import Document, RetrievalIndex

# ── the visitor's address ───────────────────────────────────────────────────

INGRESS = "100.100.0.12"


def request(forwarded: str | None, peer: str = INGRESS):
    headers = {"x-forwarded-for": forwarded} if forwarded is not None else {}
    return SimpleNamespace(client=SimpleNamespace(host=peer), headers=headers)


def test_the_visitor_is_the_address_the_ingress_appended():
    assert client_ip(request("203.0.113.7")) == "203.0.113.7"


def test_two_visitors_behind_the_same_ingress_are_told_apart():
    assert client_ip(request("203.0.113.7")) != client_ip(request("198.51.100.9"))


def test_a_forged_leftmost_entry_does_not_buy_a_new_identity():
    """The client writes whatever it likes at the front of the header. If that
    were trusted, a new fake address per request would defeat the rate limit."""
    assert client_ip(request("1.2.3.4, 203.0.113.7")) == "203.0.113.7"
    assert client_ip(request("9.9.9.9, 8.8.8.8, 203.0.113.7")) == "203.0.113.7"


@pytest.mark.parametrize(
    "entry,expected",
    [
        ("203.0.113.7:51234", "203.0.113.7"),
        ("[2001:db8::1]:443", "2001:db8::1"),
        ("2001:db8::1", "2001:db8::1"),
        (" 203.0.113.7 ", "203.0.113.7"),
    ],
)
def test_ports_and_ipv6_brackets_are_stripped(entry, expected):
    assert client_ip(request(entry)) == expected


@pytest.mark.parametrize("forwarded", [None, "", "not-an-ip", "unknown"])
def test_without_a_usable_header_the_socket_address_is_used(forwarded):
    assert client_ip(request(forwarded, peer="127.0.0.1")) == "127.0.0.1"


def test_the_header_is_ignored_when_no_proxy_is_trusted(monkeypatch):
    monkeypatch.setattr(settings, "trusted_proxy_hops", 0)
    assert client_ip(request("203.0.113.7", peer="127.0.0.1")) == "127.0.0.1"


def test_one_visitor_exhausting_the_limit_does_not_block_another():
    guards = RequestGuards(per_minute_limit=3, daily_token_limit=10)
    noisy, quiet = client_ip(request("203.0.113.7")), client_ip(request("198.51.100.9"))
    for _ in range(3):
        assert guards.enforce_rate_limit(noisy).allowed
    assert not guards.enforce_rate_limit(noisy).allowed
    assert guards.enforce_rate_limit(quiet).allowed


# ── the token budget ────────────────────────────────────────────────────────


def test_the_budget_is_refused_once_real_spend_reaches_the_limit():
    guards = RequestGuards(per_minute_limit=20, daily_token_limit=10_000)
    assert guards.enforce_token_budget(9_999).allowed
    assert not guards.enforce_token_budget(10_000).allowed
    assert not guards.enforce_token_budget(25_000).allowed


def test_usage_accumulates_and_no_increment_is_lost(tmp_path):
    store = AnalyticsStore(db_path=str(tmp_path / "a.db"), json_path=str(tmp_path / "a.json"))
    assert store.get_today_usage().tokens_used == 0
    store.increment_today_usage(4_200)
    assert store.increment_today_usage(3_100).tokens_used == 7_300
    assert store.get_today_usage().tokens_used == 7_300


def test_the_question_log_carries_the_tokens_it_cost(tmp_path, capsys):
    store = AnalyticsStore(db_path=str(tmp_path / "a.db"), json_path=str(tmp_path / "a.json"))
    store.log_question("203.0.113.7", "ua", "q", "a", out_of_scope=False, tokens=4321)
    line = next(l for l in capsys.readouterr().out.splitlines() if l.startswith("QA_LOG "))
    record = json.loads(line[len("QA_LOG "):])
    assert record["tokens"] == 4321
    assert record["client_ip"] == "203.0.113.7"


def _turn(text=None, call=None, tokens=0):
    output = []
    if call:
        output.append({"type": "function_call", "call_id": "c1", "name": call[0],
                       "arguments": json.dumps(call[1])})
    if text:
        output.append({"type": "message", "role": "assistant",
                       "content": [{"type": "output_text", "text": text}]})
    return SimpleNamespace(output=output, output_text=text or "",
                           usage=SimpleNamespace(total_tokens=tokens))


class BillingClient:
    """Replays scripted turns, each with the usage the API would report."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        response = self._turns.pop(0)
        if kwargs.get("stream"):
            events = [SimpleNamespace(type="response.output_text.delta", delta=response.output_text)]
            events.append(SimpleNamespace(type="response.completed", response=response))
            return iter(events)
        return response


def _agent(turns):
    index = RetrievalIndex(
        [Document(id="role:data-equity", category="role", title="Data Scientist at Data Equity",
                  text="Domingo Berbel has been Data Scientist at Data Equity since October 2025.",
                  embedding=None)],
        vocabulary=frozenset({"Data Equity"}),
    )
    return ProfileAgent(client=BillingClient(turns), model="gpt-4o-mini", index=index,
                        embedder=lambda text: None, limits=AgentLimits())


ANSWER = "He is Data Scientist at Data Equity."


def test_a_question_is_billed_for_every_turn_of_the_loop():
    """A tool call and the answer are two completions; both are paid for."""
    agent = _agent([
        _turn(call=("search_profile", {"query": "role"}), tokens=1_500),
        _turn(text=ANSWER, tokens=2_300),
    ])
    result = agent.answer("Where does he work?", language="en")
    assert result.tokens == 3_800
    assert agent.tokens_used == 3_800


def test_a_streamed_question_is_billed_the_same():
    agent = _agent([
        _turn(call=("search_profile", {"query": "role"}), tokens=1_500),
        _turn(text=ANSWER, tokens=2_300),
    ])
    *_, result = agent.answer_stream("Where does he work?", language="en")
    assert result.tokens == 3_800


def test_an_out_of_scope_question_is_still_billed():
    agent = _agent([_turn(text="OUT_OF_SCOPE", tokens=900)])
    result = agent.answer("Hala Madrid", language="es")
    assert result.out_of_scope and result.tokens == 900


# ── the pre-LLM content filter ──────────────────────────────────────────────

# Professional questions the substring filter refused without the model ever
# seeing them. Each contains one of its terms inside an unrelated word.
PROFESSIONAL_QUESTIONS_THE_FILTER_USED_TO_REFUSE = [
    "¿Ha escrito algún artículo o publicación?",            # culo
    "¿Sabe hacer cálculo de métricas de negocio?",           # culo
    "¿Qué reputación tiene entre sus compañeros?",           # puta
    "What is his computational background?",                 # puta
    "¿Ha participado en alguna disputa técnica?",            # puta
    "¿Tiene orientación a resultados?",                      # orientaci
    "¿Qué le hace atractivo para un equipo de datos?",       # atractivo
    "¿Ha trabajado en pair programming o en pareja?",        # pareja
    "¿Conoce el sector farmacéutico o el de la droguería?",  # droga
    "Has he worked on vehicle or particle data?",            # culo (vehículo/partícula)
    "¿Ha trabajado con datos de vehículos?",
]


@pytest.mark.parametrize("question", PROFESSIONAL_QUESTIONS_THE_FILTER_USED_TO_REFUSE)
def test_a_professional_question_is_not_refused_for_a_substring(question):
    assert not is_inappropriate(question), f"refused: {question!r}"


# Every message in the production log that was abusive or personal. Verbatim.
LOGGED_ABUSE = [
    "Hijo puta",
    "domingo es un hijoputa",
    "gilipollas",
    "Cuantas novias ha tenido domingo?",
    "tiene novia?",
    "es gay domingo?",
    "Es guapo?",
    "Eso es que es feo",
    "le gustan las mujeres?",
    "Crees que domingo encajaria bien en una pelicula XXX?",
]


@pytest.mark.parametrize("question", LOGGED_ABUSE)
def test_logged_abuse_is_still_filtered_before_the_model(question):
    assert is_inappropriate(question), f"not filtered: {question!r}"


@pytest.mark.parametrize(
    "question", ["¿Cuál es su orientación sexual?", "es IMBÉCIL", "¿Es racista?", "tiene pareja?"]
)
def test_accents_case_and_stems_do_not_let_abuse_through(question):
    assert is_inappropriate(question)


# Real questions that an earlier version answered with the out-of-scope refusal
# although they are exactly what the assistant exists for. No fast path may
# claim them; whether the model then answers is checked against production by
# scripts/eval_live.py, which uses this same list.
LOGGED_FALSE_REFUSALS = [
    "ha que se dedica domingo?",
    "que sabe de azure?",
    "cuales son los puntos fuertes de domingo",
    "le gusta el futbol?",
]


@pytest.mark.parametrize("question", LOGGED_FALSE_REFUSALS)
def test_a_question_once_refused_by_mistake_reaches_the_model(question):
    assert not is_inappropriate(question)
    assert not is_greeting(question)
    assert not is_contact_request(question)
