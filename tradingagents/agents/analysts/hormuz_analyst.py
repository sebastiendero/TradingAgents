"""Hormuz Analyst: reads Hormuz Watch shipping data and reports its market relevance.

Hormuz Watch (a separate project) tracks the Strait of Hormuz during the 2026
crisis: AIS transits, anchored queues and waiting-time forecasts, vessels that
switch off AIS, Sentinel-1 radar hull counts, port calls at bypass terminals
(Fujairah, Yanbu), sourced incidents, and how far its sources agree.

The data is fetched before the model is called (no tools), from
``HORMUZ_WATCH_DATA``: a local path or an http(s) URL to Hormuz Watch's
``data.json``. Default: the local dashboard at http://localhost:8765/data.json.
"""

from __future__ import annotations

import json
import os
import urllib.request

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction

TOOLS = ()  # data is pre-fetched into the prompt
DEFAULT_SOURCE = "http://localhost:8765/data.json"


def load_hormuz_data(source: str | None = None) -> dict | None:
    """Hormuz Watch's data.json, or None when it cannot be read."""
    source = source or os.environ.get("HORMUZ_WATCH_DATA") or DEFAULT_SOURCE
    try:
        if source.startswith(("http://", "https://")):
            req = urllib.request.Request(source, headers={"User-Agent": "tradingagents-hormuz-analyst"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)
        with open(os.path.expanduser(source)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _num(x, digits=1):
    return "n/a" if x is None else f"{x:,.{digits}f}"


def build_digest(d: dict) -> str:
    """A compact, factual digest of the shipping situation for the model to reason over."""
    st = d.get("status", {})
    sc = {s["key"]: s for s in d.get("throughput", {}).get("scenarios", [])}
    fleet = d.get("fleet", {})
    bo = d.get("beacon_off", {})
    lines = [
        f"Data generated at {d.get('generated_at')} (UTC).",
        f"Strait status: {st.get('label')} — visible AIS traffic {st.get('pct_of_normal')}% of normal; "
        f"{_num(st.get('visible_transits_per_day_7d'))} transits/day over 7 days vs {_num(st.get('baseline_per_day'), 0)} pre-war.",
    ]
    if sc:
        lines.append(
            "Estimated transits/day including AIS-dark vessels: central "
            f"{_num(sc.get('central', {}).get('transits_per_day'))}, range {_num(sc.get('bas', {}).get('transits_per_day'))}"
            f"–{_num(sc.get('haut', {}).get('transits_per_day'))}.")
        lines.append(
            "Forecast wait for a vessel arriving now (Little's law on the declared queue): "
            f"into the Gulf {_num(sc.get('central', {}).get('wait_in_days'))} days "
            f"(range {_num(sc.get('haut', {}).get('wait_in_days'))}–{_num(sc.get('bas', {}).get('wait_in_days'))}); "
            f"out of the Gulf {_num(sc.get('central', {}).get('wait_out_days'))} days. Declared queues: "
            f"{len(d.get('queue_in', []))} inbound, {len(d.get('queue_out', []))} outbound (a lower bound).")
    lines.append(f"Stationary tankers and cargo ships in the area: {fleet.get('stationary')} "
                 f"(of which {fleet.get('tankers')} tankers).")
    zones = sorted(d.get("zones", []), key=lambda z: -z.get("count", 0))[:5]
    if zones:
        lines.append("Largest waiting areas: " + "; ".join(
            f"{z['name']} {z['count']} ships ({z['tankers']} tankers)" for z in zones) + ".")
    likely = sum(1 for s in bo.get("silent", []) if s.get("likely_switch_off"))
    dark = sum(1 for r in bo.get("reappeared", []) if r.get("verdict") == "Passage sans balise déduit")
    lines.append(f"AIS switch-offs: {len(bo.get('silent', []))} vessels silent >2h, {likely} likely deliberate; "
                 f"{len(bo.get('reappeared', []))} reappearances, {dark} inferred dark transits; "
                 f"{bo.get('lost', 0)} lost from view.")
    for p in d.get("ports", []):
        flag = " [DATA FLAG: near-zero vs pre-war, unverified]" if p.get("flag") else ""
        lines.append(f"Port {p['name']} ({p['role']}): tanker calls/day {_num(p.get('tanker_calls_7d'))} "
                     f"(pre-war {_num(p.get('tanker_calls_pre'))}); tanker exports {_num((p.get('export_t_7d') or 0) / 1000, 0)} kt/day "
                     f"(pre-war {_num((p.get('export_t_pre') or 0) / 1000, 0)}){flag}.")
    passes = d.get("radar", {}).get("summary") or []
    if passes:
        last = passes[-1]
        hulls = sum(z.get("count") or 0 for z in last.get("zones", {}).values())
        lines.append(f"Latest Sentinel-1 radar pass {last.get('datetime')}: {hulls} hulls detected in the three watched zones.")
    inc = d.get("incidents", [])[:8]
    if inc:
        lines.append("Recent incidents (source in brackets; state media and military claims are not independent confirmation):")
        lines += [f"- {i.get('time')} [{i.get('source')}] {i.get('title')}" for i in inc]
    s = d.get("straits") or {}
    if s.get("brent_usd"):
        lines.append(f"Brent {s.get('brent_usd')} USD (straits.live, {s.get('as_of')}); war-risk insurance multiple: {s.get('insurance_multiple') or 'n/a'}.")
    checks = [c for c in d.get("checks", []) if c.get("state") == "ecart"]
    if checks:
        lines.append("Source disagreements to treat with caution: " + "; ".join(
            f"{c['what']} (ours {c.get('ours')}, {c.get('source')} {c.get('theirs')})" for c in checks) + ".")
    gaps = d.get("collector_gaps") or []
    if gaps:
        lines.append(f"Collection gaps in the last 3 days: {len(gaps)} (durations and switch-offs overlapping them are uncertain).")
    return "\n".join(lines)


def create_hormuz_analyst(llm):
    def hormuz_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)
        data = load_hormuz_data()
        if not data:
            report = ""  # read downstream as "no Hormuz report" rather than a blank finding
            return {"messages": [AIMessage(content="Hormuz Watch data unavailable.")], "hormuz_report": report}

        system_message = (
            "You are a shipping and energy supply-chain analyst covering the Strait of Hormuz. "
            "Using ONLY the Hormuz Watch data below, explain what the current shipping situation implies for the "
            "instrument under analysis: crude and product supply, bypass flows (Fujairah, Yanbu), tanker availability "
            "and freight, war-risk, and exposure of the company or asset. Separate measured facts from estimates, "
            "state the uncertainty and data flags explicitly, and do not invent figures that are not in the data. "
            "End with a Markdown table of the key indicators and their direction of impact on the instrument."
            + get_language_instruction()
            + "\n\nHormuz Watch data:\n" + build_digest(data)
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful AI assistant, collaborating with other assistants."
             " Report what the data supports; another agent decides the trade."
             " Today's date is {current_date}; treat it as 'now'. {instrument_context}\n{system_message}"),
            MessagesPlaceholder(variable_name="messages"),
        ])
        prompt = prompt.partial(system_message=system_message, current_date=current_date,
                                instrument_context=instrument_context)
        result = (prompt | llm).invoke(state["messages"])
        return {"messages": [result], "hormuz_report": result.content}

    return hormuz_analyst_node
