"""Daily Hormuz research notes: TradingAgents with the Hormuz Analyst, on local Ollama models.

    .venv/bin/python hormuz_run.py TTE FRO        # tickers to analyse (default: TTE)

Each run writes the full report tree (HTML + Markdown per agent) under
results/reports/<TICKER>_<stamp>/ and a short summary to results/hormuz_notes.json,
which Hormuz Watch can display. Research only: not financial or investment advice.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

ANALYSTS = ("market", "news", "hormuz")


def config() -> dict:
    cfg = DEFAULT_CONFIG.copy()
    cfg.update({
        "llm_provider": os.environ.get("TRADINGAGENTS_LLM_PROVIDER", "ollama"),
        "quick_think_llm": os.environ.get("TRADINGAGENTS_QUICK_THINK_LLM", "qwen3:latest"),
        "deep_think_llm": os.environ.get("TRADINGAGENTS_DEEP_THINK_LLM", "gpt-oss:latest"),
        "output_language": os.environ.get("TRADINGAGENTS_OUTPUT_LANGUAGE", "French"),
        "max_debate_rounds": 1,
    })
    return cfg


def run(ticker: str, date: str) -> dict:
    ta = TradingAgentsGraph(selected_analysts=ANALYSTS, config=config())
    started = time.time()
    state, decision = ta.propagate(ticker, date)
    path = ta.save_reports(state, ticker)
    return {
        "ticker": ticker, "date": date, "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.time() - started), "rating": state.get("final_rating"), "decision": decision,
        "hormuz_report": state.get("hormuz_report", ""), "report_dir": str(path),
        "models": {"quick": ta.config["quick_think_llm"], "deep": ta.config["deep_think_llm"]},
        "disclaimer": "Recherche automatisée par modèles d'IA ; pas un conseil en investissement.",
    }


def main() -> int:
    tickers = sys.argv[1:] or ["TTE"]
    date = dt.date.today().isoformat()
    out = Path(DEFAULT_CONFIG["results_dir"]) / "hormuz_notes.json"
    notes = json.loads(out.read_text()) if out.exists() else []
    for t in tickers:
        note = run(t, date)
        notes = [n for n in notes if not (n["ticker"] == t and n["date"] == date)] + [note]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(notes[-60:], ensure_ascii=False, indent=1))
        print(f"{t} : {note['rating']} ({note['seconds']} s) → {note['report_dir']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
