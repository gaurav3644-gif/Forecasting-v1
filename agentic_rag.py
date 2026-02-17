from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from assistant_openai import AssistantLLMError, OpenAIConfig, _validate_no_invented_numbers

# `retriever` is optional in this repo; keep imports tolerant.
try:
    import retriever  # type: ignore
except Exception:  # pragma: no cover
    retriever = None


class AgenticRAGError(RuntimeError):
    pass


_SQL_BLOCKLIST = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|reindex|analyze|replace)\b",
    re.IGNORECASE,
)


def _load_glossary() -> dict[str, Any]:
    try:
        p = Path(__file__).with_name("ai_glossary.json")
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _summarize_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """
    Create a compact summary suitable for prompting.

    Important: avoid embedding large sample tables/rows in the prompt, since models
    tend to echo them instead of answering.
    """
    if not isinstance(packet, dict):
        return {}
    out: dict[str, Any] = {}
    try:
        grain = packet.get("grain") if isinstance(packet.get("grain"), dict) else {}
        out["grain"] = grain
    except Exception:
        pass

    def _keep_block(name: str) -> None:
        blk = packet.get(name)
        if not isinstance(blk, dict):
            return
        keep: dict[str, Any] = {}
        for k in ("available_columns", "date_col", "date_range", "metrics"):
            if k in blk:
                keep[k] = _jsonable(blk.get(k))
        # Keep any pre-aggregated rollups (small).
        if name == "raw_sales" and isinstance(blk.get("monthly_rollup"), list):
            keep["monthly_rollup"] = _jsonable(blk.get("monthly_rollup"))
        out[name] = keep

    _keep_block("raw_sales")
    _keep_block("forecast_output")
    _keep_block("supply_plan")

    # Risks/actions summary (small).
    try:
        sp = packet.get("supply_plan_and_risks")
        if isinstance(sp, dict):
            out["supply_plan_and_risks"] = {
                "planning_context": _jsonable(sp.get("planning_context")) if isinstance(sp.get("planning_context"), dict) else {},
                "risks": _jsonable(sp.get("risks")) if isinstance(sp.get("risks"), list) else [],
                "actions": _jsonable(sp.get("actions")) if isinstance(sp.get("actions"), list) else [],
            }
    except Exception:
        pass
    return out


def _jsonable(v: Any) -> Any:
    if v is None:
        return None
    # pandas / numpy scalars
    try:
        import numpy as np  # type: ignore

        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
    except Exception:
        pass
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    if isinstance(v, (pd.Timedelta,)):
        return str(v)
    if isinstance(v, (float, int, bool, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(val) for k, val in v.items()}
    return str(v)


def _parse_combo_key(combo_key: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not combo_key or not isinstance(combo_key, str):
        return None, None
    if "||" not in combo_key:
        return None, None
    left, right = combo_key.split("||", 1)
    return (left.strip() or None, right.strip() or None)


def _filter_df_to_combo(df: pd.DataFrame, *, sku: Optional[str], store: Optional[str]) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df
    if sku and "item" in out.columns:
        out = out[out["item"].astype(str) == str(sku)]
    if store and "store" in out.columns:
        out = out[out["store"].astype(str) == str(store)]
    return out


def _normalize_df_for_sqlite(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()
    # Normalize datetime-like columns to ISO strings for consistent SQL comparisons.
    for col in list(out.columns):
        if col.lower() in ("date", "period_start", "ds", "timestamp"):
            try:
                out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
            except Exception:
                pass
    # Ensure object-like columns are plain strings (avoids pandas StringDtype quirks across platforms).
    for c in out.columns:
        if str(out[c].dtype).lower().startswith("string"):
            try:
                out[c] = out[c].astype(str)
            except Exception:
                pass
    return out


def _sqlite_schema(conn: sqlite3.Connection, *, tables: list[str]) -> dict[str, Any]:
    schema: dict[str, Any] = {}
    cur = conn.cursor()
    for t in tables:
        try:
            rows = cur.execute(f"PRAGMA table_info({t})").fetchall()
            cols = [{"name": r[1], "type": r[2]} for r in rows]
            count = cur.execute(f"SELECT COUNT(1) FROM {t}").fetchone()[0]
            schema[t] = {"columns": cols, "row_count": int(count)}
        except Exception as e:
            schema[t] = {"error": str(e)}
    return schema


def _ensure_select_only(sql: str) -> str:
    q = (sql or "").strip()
    if not q:
        raise AgenticRAGError("Empty SQL query.")
    if ";" in q.strip().rstrip(";"):
        raise AgenticRAGError("Only a single SQL statement is allowed.")
    head = q.lstrip().lower()
    if not (head.startswith("select") or head.startswith("with")):
        raise AgenticRAGError("Only SELECT queries are allowed.")
    if _SQL_BLOCKLIST.search(q):
        raise AgenticRAGError("Unsafe SQL (write/PRAGMA) is not allowed.")
    # Add a LIMIT if missing (best-effort).
    if re.search(r"\blimit\s+\d+\b", q, re.IGNORECASE) is None:
        q = q.rstrip()
        q = f"{q} LIMIT 200"
    return q


def _run_sql(conn: sqlite3.Connection, sql: str) -> dict[str, Any]:
    q = _ensure_select_only(sql)
    cur = conn.cursor()
    t0 = time.time()
    cur.execute(q)
    cols = [d[0] for d in (cur.description or [])]
    rows = cur.fetchmany(200)
    out_rows = []
    for r in rows:
        rec = {cols[i]: _jsonable(r[i]) for i in range(len(cols))}
        out_rows.append(rec)
    elapsed_ms = int((time.time() - t0) * 1000)
    return {
        "query": q,
        "columns": cols,
        "rows": out_rows,
        "row_count": len(out_rows),
        "truncated": len(out_rows) >= 200,
        "elapsed_ms": elapsed_ms,
    }


@dataclass(frozen=True)
class AgenticRAGResult:
    answer: str
    tool_calls: int
    provider: str


async def answer_question_agentic(
    question: str,
    *,
    session: dict[str, Any],
    combo_key: Optional[str],
    context_packet: Optional[dict[str, Any]] = None,
    history: Optional[list[dict[str, Any]]] = None,
    config: Optional[OpenAIConfig] = None,
) -> AgenticRAGResult:
    """
    Agentic RAG (NL2SQL over per-run data).

    - Builds three canonical data blocks (raw sales, forecast, supply plan & risks) via `context_packet`.
    - Creates an in-memory SQLite database from the user's run dataframes.
    - Uses OpenAI tool-calling to decide when to run SQL to answer the question.
    - Validates that the final answer does not introduce numeric values not present in the provided tool outputs/context.
    """
    cfg = config or OpenAIConfig.from_env()
    q = (question or "").strip()
    if not q:
        return AgenticRAGResult(answer="Data not available", tool_calls=0, provider="openai")

    raw_df = session.get("df")
    forecast_df = session.get("forecast_df")
    supply_df = session.get("supply_plan_full_df")
    if not (isinstance(supply_df, pd.DataFrame) and not supply_df.empty):
        supply_df = session.get("supply_plan_df")

    if not (isinstance(raw_df, pd.DataFrame) and not raw_df.empty):
        return AgenticRAGResult(answer="Data not available", tool_calls=0, provider="openai")
    if not (isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty):
        # Forecast may not exist for purely raw questions, but keep consistent behavior.
        forecast_df = pd.DataFrame()
    if not (isinstance(supply_df, pd.DataFrame) and not supply_df.empty):
        supply_df = pd.DataFrame()

    sku, store = _parse_combo_key(combo_key)
    raw_df = _filter_df_to_combo(raw_df, sku=sku, store=store)
    if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
        forecast_df = _filter_df_to_combo(forecast_df, sku=sku, store=store)
    if isinstance(supply_df, pd.DataFrame) and not supply_df.empty:
        supply_df = _filter_df_to_combo(supply_df, sku=sku, store=store)

    raw_df = _normalize_df_for_sqlite(raw_df)
    forecast_df = _normalize_df_for_sqlite(forecast_df)
    supply_df = _normalize_df_for_sqlite(supply_df)

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        raw_df.to_sql("raw_sales", conn, if_exists="replace", index=False)
        if not forecast_df.empty:
            forecast_df.to_sql("forecast_output", conn, if_exists="replace", index=False)
        if not supply_df.empty:
            supply_df.to_sql("supply_plan", conn, if_exists="replace", index=False)
    except Exception as e:
        try:
            conn.close()
        except Exception:
            pass
        raise AgenticRAGError(f"Failed to build local SQL workspace: {e}") from e

    tables = ["raw_sales"]
    if not forecast_df.empty:
        tables.append("forecast_output")
    if not supply_df.empty:
        tables.append("supply_plan")
    schema = _sqlite_schema(conn, tables=tables)

    # Optional: pull retrieved docs from local RAG index (if enabled).
    retrieved_docs: list[dict[str, Any]] = []
    try:
        if retriever is not None and hasattr(retriever, "retrieve"):
            retrieved_docs = await retriever.retrieve(q, k=5)
    except Exception:
        retrieved_docs = []

    glossary = _load_glossary()
    packet = context_packet if isinstance(context_packet, dict) else {}
    try:
        driver_artifacts = session.get("driver_artifacts")
        if isinstance(driver_artifacts, dict) and driver_artifacts:
            packet = dict(packet)
            packet["xai"] = {
                "directional_view": driver_artifacts.get("directional_view"),
                "local_drivers": driver_artifacts.get("local_drivers"),
                "local_driver_meta": driver_artifacts.get("local_driver_meta"),
            }
        feature_importance = session.get("feature_importance")
        if isinstance(feature_importance, dict) and feature_importance:
            packet = dict(packet)
            packet["feature_importance"] = feature_importance
    except Exception:
        pass
    if retrieved_docs:
        packet = dict(packet)
        packet["retrieved_docs"] = retrieved_docs

    tool_calls = 0

    tools = [
        {
            "type": "function",
            "function": {
                "name": "sql_query",
                "description": "Run a read-only SQL SELECT query against the per-run tables (raw_sales, forecast_output, supply_plan). Use this to fetch exact numbers and facts. Only SELECT/CTE queries are allowed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "A single SELECT statement (WITH ... SELECT ... is allowed)."}
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_schema",
                "description": "Get the available tables, columns, and row counts.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]

    base_rules = (
        "You are PiTensor, a supply chain decision assistant.\n\n"
        "Rules:\n"
        "- Answer ONLY using the data provided in the tool outputs and data blocks.\n"
        "- Do NOT use outside knowledge.\n"
        '- If the data is insufficient, say exactly: \"Data not available\".\n'
        "- Do NOT invent numbers.\n"
        "- The safest way to include numbers is to fetch them via sql_query.\n"
        "- You MAY use SQL aggregates (SUM/AVG/MIN/MAX) instead of doing arithmetic in text.\n"
        "- Keep explanations concise and business-focused.\n"
        "- Do NOT dump raw JSON, schema JSON, or large data previews. Summarize.\n"
        "- Do NOT answer with dataset metadata (rows/columns/date range) unless the user asked for it.\n"
        "- First directly answer the user's question, then add brief reasoning.\n"
    )

    # Keep some context always visible to the model.
    schema_text = json.dumps(schema, ensure_ascii=False)
    glossary_text = json.dumps(glossary, ensure_ascii=False)
    packet_summary = _summarize_packet(packet)
    packet_text = json.dumps(packet_summary, ensure_ascii=False)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": base_rules},
        {
            "role": "system",
            "content": (
                "You have access to these tables in a read-only SQL workspace:\n"
                f"SCHEMA_JSON:\n{schema_text}\n\n"
                f"GLOSSARY_JSON:\n{glossary_text}\n\n"
                "When needed, call get_schema or sql_query."
            ),
        },
        {
            "role": "system",
            "content": (
                "CANONICAL_DATA_BLOCKS_JSON (do not assume anything beyond this):\n"
                f"{packet_text}"
            ),
        },
    ]

    # Optional short conversation history.
    try:
        if isinstance(history, list):
            for m in history[-8:]:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role") or "").strip().lower()
                content = str(m.get("content") or "").strip()
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})
    except Exception:
        pass

    messages.append({"role": "user", "content": q})

    async def _call_llm(*, tool_choice: str = "auto") -> dict[str, Any]:
        import httpx

        payload = {
            "model": cfg.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "max_tokens": int(cfg.max_tokens),
            "temperature": float(cfg.temperature),
        }
        url = f"{cfg.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=cfg.timeout_s) as client:
            res = await client.post(url, headers=headers, json=payload)
            try:
                res.raise_for_status()
            except httpx.HTTPStatusError as e:
                detail = (e.response.text or "").strip()
                raise AssistantLLMError(f"OpenAI request failed: HTTP {e.response.status_code}. {detail[:800]}") from e
        return res.json()

    # Tool-call loop (bounded).
    tool_results_for_validation: list[dict[str, Any]] = []
    final_answer: Optional[str] = None
    for _ in range(0, 6):
        data = await _call_llm(tool_choice="auto")
        msg = ((data.get("choices") or [{}])[0].get("message") or {})
        tool_calls_msg = msg.get("tool_calls") or []
        content = str(msg.get("content") or "").strip()
        if tool_calls_msg:
            messages.append({"role": "assistant", "content": content or "", "tool_calls": tool_calls_msg})
            for tc in tool_calls_msg:
                fn = (tc.get("function") or {}).get("name")
                args_raw = (tc.get("function") or {}).get("arguments") or "{}"
                tool_call_id = tc.get("id")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                except Exception:
                    args = {}

                if fn == "get_schema":
                    result = {"schema": schema, "tables": tables}
                elif fn == "sql_query":
                    try:
                        query = str(args.get("query") or "")
                        result = _run_sql(conn, query)
                    except Exception as e:
                        result = {"error": str(e)}
                else:
                    result = {"error": f"Unknown tool: {fn}"}

                tool_calls += 1
                tool_results_for_validation.append({"tool": fn, "result": result})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": fn,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
            continue

        if content:
            final_answer = content
            break

    try:
        conn.close()
    except Exception:
        pass

    if not final_answer:
        raise AgenticRAGError("LLM returned an empty response.")

    # Heuristic: if the model didn't use SQL and responded with a metadata dump for a "top/highest month" question,
    # compute a minimal SQL result and ask the model to narrate it (no new numbers).
    try:
        if tool_calls == 0:
            ql = q.lower()
            looks_like_meta = ("rows" in final_answer.lower() and "date range" in final_answer.lower()) or ("rows ×" in final_answer.lower())
            asks_top_month = ("month" in ql) and any(w in ql for w in ("highest", "top", "most", "max")) and any(w in ql for w in ("sale", "sales", "demand", "actual"))
            if looks_like_meta and asks_top_month:
                rs = packet_summary.get("raw_sales") if isinstance(packet_summary.get("raw_sales"), dict) else {}
                date_col = rs.get("date_col") if isinstance(rs.get("date_col"), str) else "date"
                sales_col = None
                metrics = rs.get("metrics") if isinstance(rs.get("metrics"), dict) else {}
                if isinstance(metrics.get("sales_col"), str):
                    sales_col = metrics.get("sales_col")
                sales_col = sales_col or "sales"
                sql = (
                    f"SELECT substr({date_col}, 1, 7) AS month, "
                    f"SUM(CAST({sales_col} AS REAL)) AS total_sales "
                    f"FROM raw_sales "
                    f"WHERE {date_col} IS NOT NULL "
                    f"GROUP BY month "
                    f"ORDER BY total_sales DESC "
                    f"LIMIT 3"
                )
                # Run on a temporary connection because `conn` is closed below.
                conn2 = sqlite3.connect(":memory:", check_same_thread=False)
                try:
                    raw_df.to_sql("raw_sales", conn2, if_exists="replace", index=False)
                    sql_res = _run_sql(conn2, sql)
                finally:
                    try:
                        conn2.close()
                    except Exception:
                        pass
                tool_results_for_validation.append({"tool": "sql_query", "result": sql_res})
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Computed SQL result for this question (use ONLY these numbers for the answer; do not add any other numbers):\n"
                            f"{json.dumps(sql_res, ensure_ascii=False)}\n\n"
                            "Now answer the user's question in a ChatGPT-like explanatory tone:\n"
                            "- Start with the direct answer (which month is highest).\n"
                            "- Add 2-4 short bullets explaining the result.\n"
                            "- Do NOT mention row counts/columns.\n"
                        ),
                    }
                )
                messages.append({"role": "user", "content": q})
                data3 = await _call_llm(tool_choice="none")
                msg3 = ((data3.get("choices") or [{}])[0].get("message") or {})
                content3 = str(msg3.get("content") or "").strip()
                if content3:
                    final_answer = content3
    except Exception:
        pass

    # Validate "no invented numbers" against: data blocks + tool outputs (schema/tool results).
    validation_packet: dict[str, Any] = {
        "data_blocks": packet,
        "schema": schema,
        "tool_results": tool_results_for_validation,
    }
    try:
        _validate_no_invented_numbers(final_answer, validation_packet)
    except AssistantLLMError:
        # Retry once with a stricter "no digits" constraint to prevent invented numbers.
        messages.append(
            {
                "role": "system",
                "content": (
                    "Retry (strict): Your previous answer included numeric values that were not present in the provided data/tool outputs.\n"
                    "Re-answer WITHOUT including any digits 0-9 anywhere. If the data is insufficient, say exactly: Data not available."
                ),
            }
        )
        messages.append({"role": "user", "content": q})
        data2 = await _call_llm(tool_choice="none")
        msg2 = ((data2.get("choices") or [{}])[0].get("message") or {})
        content2 = str(msg2.get("content") or "").strip()
        if content2:
            final_answer = content2
        # Safety: enforce no digits.
        final_answer = re.sub(r"\d", "", final_answer).strip() or "Data not available"

    return AgenticRAGResult(answer=final_answer, tool_calls=tool_calls, provider="openai")
