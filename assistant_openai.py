from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from math import isfinite
from typing import Any, Mapping, Sequence

import httpx

# `retriever` is an optional local module used for RAG. Make the import
# tolerant so the app can run when the module is not present.
try:
    import retriever  # type: ignore
except Exception:  # pragma: no cover - best-effort fallback when module missing
    retriever = None


class AssistantLLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    max_tokens: int = 2000
    temperature: float = 0.0
    timeout_s: float = 45.0

    @staticmethod
    def from_env() -> "OpenAIConfig":
        key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not key:
            raise AssistantLLMError("OPENAI_API_KEY is not set.")
        base_url = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip().rstrip("/")
        model = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
        max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
        timeout_s = float(os.getenv("OPENAI_TIMEOUT_S", "45"))
        return OpenAIConfig(
            api_key=key,
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )


_NUM_RE = re.compile(r"(?<![A-Za-z])[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?")


def _extract_numbers(text: str) -> list[tuple[str, float, bool]]:
    if not isinstance(text, str) or not text:
        return []
    hits: list[tuple[str, float, bool]] = []
    for m in _NUM_RE.finditer(text):
        tok = m.group(0)
        if not tok or tok in ("-", "+"):
            continue
        is_pct = tok.endswith("%")
        raw = tok[:-1] if is_pct else tok
        try:
            val = float(raw.replace(",", ""))
        except Exception:
            continue
        hits.append((tok, val, is_pct))
    return hits


def _collect_numbers(obj: Any, out: list[float], *, limit: int = 5000) -> None:
    if len(out) >= limit:
        return
    if obj is None:
        return
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        try:
            v = float(obj)
            if isfinite(v):
                out.append(v)
        except Exception:
            return
        return
    if isinstance(obj, Mapping):
        for v in obj.values():
            _collect_numbers(v, out, limit=limit)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, out, limit=limit)
        return


def _expand_allowed_numbers(values: Sequence[float]) -> list[float]:
    out: list[float] = []
    for v in values:
        try:
            fv = float(v)
        except Exception:
            continue
        out.append(fv)
        for d in (0, 1, 2, 3):
            out.append(round(fv, d))
            try:
                q = Decimal("1") if d == 0 else Decimal("1").scaleb(-d)
                out.append(float(Decimal(str(fv)).quantize(q, rounding=ROUND_HALF_UP)))
            except (InvalidOperation, ValueError):
                pass
        if 0.0 <= fv <= 1.0:
            pct = fv * 100.0
            out.append(pct)
            for d in (0, 1, 2):
                out.append(round(pct, d))
                try:
                    q = Decimal("1") if d == 0 else Decimal("1").scaleb(-d)
                    out.append(float(Decimal(str(pct)).quantize(q, rounding=ROUND_HALF_UP)))
                except (InvalidOperation, ValueError):
                    pass
    # Allow small ordinals for list numbering
    out.extend(float(i) for i in range(0, 13))
    # De-dup
    seen = set()
    uniq: list[float] = []
    for x in out:
        if x == 0.0:
            x = 0.0
        if x in seen:
            continue
        seen.add(x)
        uniq.append(x)
    return uniq


def _is_allowed_number(val: float, allowed: Sequence[float]) -> bool:
    for a in allowed:
        try:
            af = float(a)
        except Exception:
            continue
        if abs(val - af) <= 1e-9:
            return True
        if abs(af) > 1e-9 and abs(val - af) / abs(af) <= 1e-6:
            return True
    return False


def _validate_no_invented_numbers(answer: str, *, packet: dict[str, Any]) -> None:
    nums: list[float] = []
    _collect_numbers(packet, nums)
    expanded = _expand_allowed_numbers(nums)

    invented: list[str] = []
    for tok, val, is_pct in _extract_numbers(answer):
        candidates = [val / 100.0, val] if is_pct else [val]
        ok = any(_is_allowed_number(c, expanded) for c in candidates)
        if not ok:
            invented.append(tok)
    if invented:
        uniq = []
        for t in invented:
            if t not in uniq:
                uniq.append(t)
        raise AssistantLLMError(f"LLM invented numeric values (not allowed). Unexpected numbers: {uniq[:12]}")


def _system_prompt(*, allow_digits: bool) -> str:
    base = (
        "You are a supply chain decision assistant.\n\n"
        "Rules:\n"
        "- Answer ONLY using the data provided below.\n"
        "- Do NOT use outside knowledge.\n"
        '- If the data is insufficient, say \"Data not available\".\n'
        "- Do NOT invent numbers.\n"
        "- Keep explanations concise and business-focused.\n"
    )
    if not allow_digits:
        base += (
            "\nAdditional rule (retry):\n"
            "- Do NOT include any digits 0-9 anywhere in your answer.\n"
        )
    return base


async def answer_question_openai(
    question: str,
    *,
    context_packet: dict[str, Any],
    config: OpenAIConfig | None = None,
) -> str:
    cfg = config or OpenAIConfig.from_env()
    q = (question or "").strip()
    if not q:
        return "Data not available"

    # Retrieve relevant documents and merge into packet (RAG)
    try:
        if retriever is not None and hasattr(retriever, "retrieve"):
            retrieved = await retriever.retrieve(q, k=5)
            print(f"[RAG DEBUG] Retrieved {len(retrieved)} documents for query: {q[:50]}...")
            if retrieved:
                for i, doc in enumerate(retrieved[:3]):
                    print(f"[RAG DEBUG] Doc {i+1} type: {doc.get('metadata', {}).get('type', 'unknown')}")
        else:
            retrieved = []
            print(f"[RAG DEBUG] Retriever not available")
    except Exception as e:
        retrieved = []
        print(f"[RAG DEBUG] Retrieval failed: {e}")
    merged_packet = dict(context_packet)
    merged_packet["retrieved_docs"] = retrieved

    # Debug: Check what data is available in context
    print(f"[CONTEXT DEBUG] Keys in context_packet: {list(context_packet.keys())}")
    if "forecast_output" in context_packet:
        fc = context_packet["forecast_output"]
        if fc:
            print(f"[CONTEXT DEBUG] Forecast output has {len(fc.get('available_columns', []))} columns")
        else:
            print(f"[CONTEXT DEBUG] Forecast output is None/empty")
    else:
        print(f"[CONTEXT DEBUG] No forecast_output in context_packet")

    packet_json = json.dumps(merged_packet, ensure_ascii=False)
    user_msg = f"USER_QUESTION:\n{q}\n\nDATA_PACKET_JSON:\n{packet_json}"

    async def _call_once(*, allow_digits: bool) -> str:
        payload = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": _system_prompt(allow_digits=allow_digits)},
                {"role": "user", "content": user_msg},
            ],
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
        data = res.json()
        content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
        content = str(content or "").strip()
        if not content:
            raise AssistantLLMError("LLM returned an empty response.")
        return content

    # Attempt 1: allow digits, validate strictly against merged packet.
    ans1 = await _call_once(allow_digits=True)
    try:
        _validate_no_invented_numbers(ans1, packet=merged_packet)
        return ans1
    except AssistantLLMError:
        # Retry once with no-digits rule (qualitative response).
        ans2 = await _call_once(allow_digits=False)
        # Should have no digits; still validate if any appear.
        _validate_no_invented_numbers(ans2, packet=merged_packet)
        return ans2

