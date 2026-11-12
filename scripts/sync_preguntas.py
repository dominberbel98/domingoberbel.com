#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


def load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def fetch_remote_items(url: str, admin_key: str) -> list[dict]:
    req = urllib.request.Request(url)
    req.add_header("X-Admin-Key", admin_key)

    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    items = payload.get("items", [])
    return items if isinstance(items, list) else []


QA_LOG_PREFIX = "QA_LOG "


def parse_log_analytics_rows(rows: list[dict]) -> list[dict]:
    """Turn console-log rows into Q&A entries.

    The backend prints one `QA_LOG {json}` line per question (see
    backend/app/services/analytics_store.py). Any other console line, and any
    line whose JSON does not parse, is skipped.
    """
    items: list[dict] = []
    for row in rows:
        line = str(row.get("Log_s") or row.get("Log") or "")
        start = line.find(QA_LOG_PREFIX)
        if start < 0:
            continue
        try:
            entry = json.loads(line[start + len(QA_LOG_PREFIX):])
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and entry.get("question"):
            items.append(entry)
    return items


def fetch_log_analytics_items(resource_group: str, environment: str, days: int) -> list[dict]:
    """Read the QA_LOG lines Azure Container Apps shipped to Log Analytics.

    This is the durable source: the container's own files are wiped on every
    deploy or restart, the workspace keeps the console output for its
    retention period (30 days by default). Needs the `az` CLI, logged in.
    """
    az = shutil.which("az")
    if not az:
        raise RuntimeError("az CLI not found")

    workspace = subprocess.run(
        [az, "containerapp", "env", "show", "--name", environment,
         "--resource-group", resource_group, "--query",
         "properties.appLogsConfiguration.logAnalyticsConfiguration.customerId",
         "-o", "tsv"],
        capture_output=True, text=True, timeout=120, check=True,
    ).stdout.strip()
    if not workspace:
        raise RuntimeError("the Container Apps environment has no Log Analytics workspace")

    query = (
        "ContainerAppConsoleLogs_CL "
        f"| where Log_s has '{QA_LOG_PREFIX.strip()}' "
        "| project TimeGenerated, Log_s "
        "| order by TimeGenerated asc"
    )
    result = subprocess.run(
        [az, "monitor", "log-analytics", "query", "--workspace", workspace,
         "--analytics-query", query, "--timespan", f"P{days}D", "-o", "json"],
        capture_output=True, text=True, timeout=180, check=True,
    )
    rows = json.loads(result.stdout or "[]")
    return parse_log_analytics_rows(rows if isinstance(rows, list) else [])


def unique_key(item: dict) -> str:
    ts = str(item.get("timestamp") or item.get("ts") or item.get("created_at") or "")
    q = str(item.get("question") or "")
    a = str(item.get("answer") or item.get("answer_preview") or "")
    return f"{ts}|{q}|{a}"


def merge_items(local_items: list[dict], remote_items: list[dict]) -> list[dict]:
    by_key: dict[str, dict] = {}

    for it in local_items:
        by_key[unique_key(it)] = it

    for it in remote_items:
        normalized = {
            "timestamp": it.get("timestamp") or it.get("ts") or it.get("created_at") or "",
            "date": it.get("date") or "",
            "time": it.get("time") or "",
            "question": it.get("question") or "",
            "answer": it.get("answer") or it.get("answer_preview") or "",
            "out_of_scope": bool(it.get("out_of_scope", False)),
            "client_ip": it.get("client_ip") or "",
            "user_agent": it.get("user_agent") or "",
        }
        by_key[unique_key(normalized)] = normalized

    merged = list(by_key.values())
    merged.sort(key=lambda x: str(x.get("timestamp", "")))
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync remote Q&A logs to local preguntas.json")
    parser.add_argument(
        "--url",
        default="https://api.domingoberbel.com/api/admin/questions-json",
        help="Admin endpoint URL",
    )
    parser.add_argument(
        "--output",
        default="preguntas.json",
        help="Local JSON file path",
    )
    parser.add_argument(
        "--admin-key",
        default=os.getenv("ADMIN_READ_KEY", ""),
        help="Admin API key (defaults to ADMIN_READ_KEY env var)",
    )
    parser.add_argument(
        "--resource-group",
        default=os.getenv("AZ_RESOURCE_GROUP", ""),
        help="Azure resource group (enables the Log Analytics source)",
    )
    parser.add_argument(
        "--environment",
        default=os.getenv("AZ_CONTAINERAPPS_ENV", ""),
        help="Container Apps environment name (enables the Log Analytics source)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="How far back to read Log Analytics (its default retention is 30 days)",
    )
    args = parser.parse_args()

    if not args.admin_key:
        print("ERROR: Missing admin key. Set ADMIN_READ_KEY or pass --admin-key.", file=sys.stderr)
        return 1

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        remote_items = fetch_remote_items(args.url, args.admin_key)
    except urllib.error.URLError as exc:
        print(f"ERROR: failed to fetch remote items: {exc}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"ERROR: invalid JSON from endpoint: {exc}", file=sys.stderr)
        return 3

    # The endpoint only knows what the current container has seen since it
    # last started. Log Analytics has everything from the last `--days` days.
    durable_items: list[dict] = []
    if args.resource_group and args.environment:
        try:
            durable_items = fetch_log_analytics_items(
                args.resource_group, args.environment, args.days
            )
            print(f"Log Analytics: {len(durable_items)} entries in the last {args.days} days")
        except (RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            detail = getattr(exc, "stderr", "") or exc
            print(f"WARNING: could not read Log Analytics: {str(detail).strip()[:300]}",
                  file=sys.stderr)
    else:
        print("NOTE: Log Analytics source skipped (set AZ_RESOURCE_GROUP and "
              "AZ_CONTAINERAPPS_ENV, or pass --resource-group/--environment).")

    local_items = load_json_list(output_path)
    merged = merge_items(local_items, remote_items + durable_items)

    output_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: wrote {len(merged)} entries to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
