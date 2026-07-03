"""Produce strict v2 scientific attack/rescue reviews for selected candidates."""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from kb_common import safe_print
from llm_structured import extract_json
from opencode_cli_adapter import DEFAULT_OPENCODE_MODEL, run_opencode_cli
from research_seed_v2_common import artifact_dir, artifact_hashes, candidate_id, ensure_v2_dirs, read_json, write_json, write_run_artifact


REQUIRED_REVIEW_FIELDS = [
    "novelty_attack",
    "baseline_attack",
    "mechanism_attack",
    "evaluation_attack",
    "scope_attack",
    "a_plus_b_risk",
    "fatal_flaw",
    "rescue_mutation",
    "survivability_label",
    "allowed_next_stage",
]
OPTIONAL_EMPTY_REVIEW_FIELDS = {"fatal_flaw", "rescue_mutation"}
SURVIVABILITY_LABELS = {"survives", "survives_if_mutated", "park_for_unknown", "reject_with_rescue", "reject_fatal"}
ALLOWED_NEXT_STAGES = {"novelty_scan", "gemini_mutation", "parked", "rescue", "stop"}
A_PLUS_B_RISK = {"none", "low", "medium", "high", "fatal"}
DEFAULT_OPENCODE_BATCH_SIZE = 1
DEFAULT_OPENCODE_RETRIES = 1
DEFAULT_OPENCODE_FALLBACK_MODELS: tuple[str, ...] = ()
DEFAULT_OPENAI_COMPATIBLE_MAX_TOKENS = 8192
PROVIDER_CANDIDATE_FIELDS = [
    "candidate_id",
    "title",
    "problem",
    "hypothesis",
    "claim_compression",
    "non_obvious_claim",
    "mechanism",
    "core_insight",
    "engineering_pathology",
    "interface",
    "interface_innovation",
    "strongest_baseline",
    "baseline_matrix",
    "baseline_kill_table",
    "anti_combination_test",
    "falsification",
    "killer_experiment",
    "minimum_no_hardware_pilot",
    "metric_suite",
    "negative_claim_boundary",
    "novelty_risk",
    "nearest_pressure",
    "competition_map",
    "lab_fit",
    "rescue_mutation",
    "post_kill_mutation",
    "risk_assumptions",
    "domains",
    "issues",
]


def _fallback_review_candidate(item: dict[str, Any]) -> dict[str, Any]:
    cid = candidate_id(item)
    text = " ".join(str(item.get(key, "")) for key in ["title", "problem", "mechanism", "interface", "strongest_baseline", "reviewer_kill_shot"]).lower()
    has_baseline = bool(str(item.get("strongest_baseline") or item.get("baseline_kill_table") or "").strip())
    has_pilot = bool(str(item.get("minimum_no_hardware_pilot") or item.get("pilot") or item.get("killer_experiment") or "").strip())
    fatal = ""
    if "impossible" in text or "fabricated" in text:
        fatal = "candidate_contains_impossible_or_fabricated_core_claim"
    elif not has_baseline:
        fatal = "missing_strongest_baseline"
    label = "survives"
    allowed = "novelty_scan"
    if fatal:
        label = "reject_fatal"
        allowed = "stop"
    elif not has_pilot:
        label = "survives_if_mutated"
        allowed = "gemini_mutation"
    return {
        "candidate_id": cid,
        "candidate_title": item.get("title", ""),
        "status": "failed_fallback_only",
        "novelty_attack": item.get("novelty_risk") or "nearest-prior overlap must be checked before promotion.",
        "baseline_attack": item.get("strongest_baseline") or "missing strongest baseline blocks promotion.",
        "mechanism_attack": item.get("reviewer_kill_shot") or "mechanism must isolate the causal variable rather than stack modules.",
        "evaluation_attack": item.get("killer_experiment") or "needs a falsifying no-hardware pilot and metric.",
        "scope_attack": item.get("negative_claim_boundary") or "scope must be narrowed to one interface or benchmark claim.",
        "a_plus_b_risk": "high" if "with" in text or "combine" in text else "low",
        "fatal_flaw": fatal,
        "rescue_mutation": item.get("rescue_mutation") or item.get("post_kill_mutation") or "",
        "survivability_label": label,
        "allowed_next_stage": allowed,
    }


def _provider_failure_fallback_review_candidate(item: dict[str, Any], provider_error: str) -> dict[str, Any]:
    cid = candidate_id(item)
    return {
        "candidate_id": cid,
        "candidate_title": item.get("title", ""),
        "status": "failed_fallback_only",
        "novelty_attack": "Provider review did not return validated JSON; do not use this candidate as reviewed.",
        "baseline_attack": "Provider review unavailable; strongest-baseline pressure remains unverified.",
        "mechanism_attack": "Provider review unavailable; mechanism critique remains unverified.",
        "evaluation_attack": "Provider review unavailable; no-hardware evaluation critique remains unverified.",
        "scope_attack": "Provider review unavailable; scope boundary remains unverified.",
        "a_plus_b_risk": "fatal",
        "fatal_flaw": provider_error or "deepseek_provider_review_unavailable",
        "rescue_mutation": "",
        "survivability_label": "reject_fatal",
        "allowed_next_stage": "stop",
    }


def _load_provider_payload(path_value: str) -> dict[str, Any]:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        return {"_provider_error": f"provider_review_json_missing:{path}"}
    try:
        payload = read_json(path)
    except Exception as exc:
        return {"_provider_error": f"provider_review_json_invalid:{type(exc).__name__}:{exc}"}
    return payload if isinstance(payload, dict) else {"_provider_error": "provider_review_json_not_object"}


def _extract_json_object(text: str) -> dict[str, Any]:
    payload = extract_json(text, want=_looks_like_review_payload)
    if _looks_like_opencode_event(payload) and not _looks_like_review_payload(payload):
        raise ValueError("provider_output_json_object_not_found")
    return payload


def _looks_like_review_payload(payload: dict[str, Any]) -> bool:
    if isinstance(payload.get("reviews"), list):
        return True
    if isinstance(payload.get("reviews"), dict):
        return True
    if isinstance(payload.get("review"), dict):
        return True
    if any(field in payload for field in REQUIRED_REVIEW_FIELDS):
        return True
    return False


def _looks_like_opencode_event(payload: dict[str, Any]) -> bool:
    event_type = payload.get("type")
    if isinstance(event_type, str) and event_type in {"step_start", "step-start", "tool_use", "tool", "message"}:
        return True
    return bool({"sessionID", "timestamp", "part"} & set(payload)) and not _looks_like_review_payload(payload)


def _trim_provider_value(value: Any, *, limit: int = 1200) -> Any:
    if isinstance(value, str):
        text = value.strip()
        return text if len(text) <= limit else text[:limit] + "...[truncated]"
    if isinstance(value, list):
        return [_trim_provider_value(item, limit=limit) for item in value[:8]]
    if isinstance(value, dict):
        return {str(key): _trim_provider_value(item, limit=limit) for key, item in list(value.items())[:12]}
    return value


def _provider_candidate_view(item: dict[str, Any]) -> dict[str, Any]:
    view: dict[str, Any] = {"candidate_id": candidate_id(item)}
    for field in PROVIDER_CANDIDATE_FIELDS:
        if field == "candidate_id":
            continue
        value = item.get(field)
        if value not in (None, "", [], {}):
            view[field] = _trim_provider_value(value)

    evidence_items = item.get("evidence")
    if isinstance(evidence_items, list):
        view["evidence_excerpt"] = [
            {
                key: _trim_provider_value(record.get(key), limit=500)
                for key in ("claim_type", "source_title", "source_note", "statement")
                if isinstance(record, dict) and record.get(key)
            }
            for record in evidence_items[:6]
            if isinstance(record, dict)
        ]

    novelty_hits = item.get("novelty_hits")
    if isinstance(novelty_hits, list):
        view["novelty_hits_excerpt"] = [
            {
                key: _trim_provider_value(record.get(key), limit=300)
                for key in ("kind", "title", "source", "overlap_score")
                if isinstance(record, dict) and record.get(key) is not None
            }
            for record in novelty_hits[:6]
            if isinstance(record, dict)
        ]
    return view


def _provider_candidate_retry_view(item: dict[str, Any]) -> dict[str, Any]:
    view: dict[str, Any] = {"candidate_id": candidate_id(item)}
    for field in (
        "title",
        "problem",
        "hypothesis",
        "non_obvious_claim",
        "mechanism",
        "strongest_baseline",
        "falsification",
        "minimum_no_hardware_pilot",
        "novelty_risk",
        "nearest_pressure",
        "domains",
        "issues",
    ):
        value = item.get(field)
        if value not in (None, "", [], {}):
            view[field] = _trim_provider_value(value, limit=350)
    return view


def _render_opencode_prompt(
    selected: list[dict[str, Any]],
    *,
    run_date: str,
    retry_reason: str = "",
    previous_output: str = "",
    retry_details: list[str] | None = None,
    retry_compact: bool = False,
) -> str:
    candidates = [
        _provider_candidate_retry_view(item) if retry_compact else _provider_candidate_view(item)
        for item in selected
    ]
    if retry_compact:
        review_template = [
            {
                "candidate_id": str(item.get("candidate_id") or ""),
                "candidate_title": str(item.get("title") or item.get("candidate_title") or item.get("candidate_id") or ""),
                "status": "success",
                "novelty_attack": "string",
                "baseline_attack": "string",
                "mechanism_attack": "string",
                "evaluation_attack": "string",
                "scope_attack": "string",
                "a_plus_b_risk": "none|low|medium|high|fatal",
                "fatal_flaw": "",
                "rescue_mutation": "",
                "survivability_label": "survives|survives_if_mutated|park_for_unknown|reject_with_rescue|reject_fatal",
                "allowed_next_stage": "novelty_scan|gemini_mutation|parked|rescue|stop",
            }
            for item in candidates
        ]
        return json.dumps(
            {
                "task": (
                    "Return one RFC 8259 JSON object only. No tools, no markdown. "
                    "Do not call todowrite, webfetch, web search, or any other tool. "
                    "All object keys and string values must use double quotes. "
                    "Write one status=success review for every exact candidate_id in exact_candidate_ids."
                ),
                "schema_version": "deepseek_review.v1",
                "run_date": run_date,
                "retry_reason": retry_reason,
                "previous_invalid_output_kind": _classify_invalid_provider_output(previous_output),
                "previous_failure_details": list(retry_details or [])[:8],
                "input_mode": "compact_retry" if retry_reason else "compact_provider_review",
                "exact_candidate_ids": [str(item.get("candidate_id") or "") for item in candidates],
                "required_output_shape": {
                    "schema_version": "deepseek_review.v1",
                    "run_date": run_date,
                    "status": "success",
                    "reviews": review_template,
                },
                "required_review_fields": REQUIRED_REVIEW_FIELDS,
                "allowed_enums": {
                    "a_plus_b_risk": sorted(A_PLUS_B_RISK),
                    "survivability_label": sorted(SURVIVABILITY_LABELS),
                    "allowed_next_stage": sorted(ALLOWED_NEXT_STAGES),
                },
                "selected_candidates": candidates,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    task = (
        "This is not a coding or implementation task; it is a research-candidate review serialization task. "
        "Do not run tools, call todowrite, call webfetch, inspect files, search the web, modify files, or start/cancel background tasks. "
        "Return RFC 8259 JSON only. All object keys and string values must use double quotes. "
        "Do not use JavaScript object literal syntax, markdown fences, or explanatory prose."
    )
    if retry_reason == "opencode_invalid_review_payload":
        task = (
            "Retry because the previous JSON did not contain exactly one valid review for every selected candidate_id. "
            "This is not a coding or implementation task; it is a research-candidate review serialization task. "
            "Do not run tools, call todowrite, call webfetch, inspect files, search the web, modify files, or start/cancel background tasks. "
            "Return exactly one raw JSON object with a reviews array containing one status=success review per selected candidate. "
            "All object keys and string values must use double quotes. "
            "Do not include markdown, commentary, analysis text, or code fences."
        )
    elif retry_reason:
        task = (
            "Retry because the previous provider output was not valid JSON. "
            "This is not a coding or implementation task; it is a research-candidate review serialization task. "
            "Do not run tools, call todowrite, call webfetch, inspect files, search the web, modify files, or start/cancel background tasks. "
            "Return exactly one raw JSON object starting with { and ending with }. "
            "All object keys and string values must use double quotes. "
            "Do not include markdown, commentary, analysis text, or code fences."
        )
    return json.dumps(
        {
            "task": task,
            "schema_version": "deepseek_review.v1",
            "run_date": run_date,
            "retry_reason": retry_reason,
            "previous_invalid_output_kind": _classify_invalid_provider_output(previous_output),
            "previous_failure_details": list(retry_details or [])[:8],
            "input_mode": "compact_retry" if retry_compact else "standard",
            "required_output_shape": {
                "schema_version": "deepseek_review.v1",
                "run_date": run_date,
                "status": "success",
                "provider_status": {
                    "provider": "deepseek",
                    "provider_backed": True,
                    "mode": "opencode",
                    "requested_model": "<filled_by_runner>",
                    "effective_model": "<filled_by_runner>",
                    "exit_code": 0,
                },
                "reviews": [
                    {
                        "candidate_id": "string",
                        "candidate_title": "string",
                        "status": "success",
                        "novelty_attack": "string",
                        "baseline_attack": "string",
                        "mechanism_attack": "string",
                        "evaluation_attack": "string",
                        "scope_attack": "string",
                        "a_plus_b_risk": "none|low|medium|high|fatal",
                        "fatal_flaw": "",
                        "rescue_mutation": "",
                        "survivability_label": "survives|survives_if_mutated|park_for_unknown|reject_with_rescue|reject_fatal",
                        "allowed_next_stage": "novelty_scan|gemini_mutation|parked|rescue|stop",
                    }
                ],
            },
            "validation_rules": [
                "Return exactly one review per candidate_id.",
                "Use status=success only when the review is substantive.",
                "fatal_flaw and rescue_mutation may be empty strings; every other required review field must be non-empty.",
                "Do not accept A+B combinations unless the mechanism survives baseline and evaluation attacks.",
            ],
            "selected_candidates": candidates,
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def _classify_invalid_provider_output(text: str) -> str:
    sample = (text or "").strip()
    if not sample:
        return "empty_output"
    first_line = sample.splitlines()[0].strip()
    try:
        payload = json.loads(first_line)
    except json.JSONDecodeError:
        return "invalid_json"
    if _looks_like_opencode_event(payload):
        return f"opencode_event:{payload.get('type') or 'unknown'}"
    return "json_not_matching_review_schema"


def _provider_error_payload(error: str, *, status: str, provider_status: dict[str, Any]) -> dict[str, Any]:
    return {
        "_provider_error": error,
        "_provider_failure_status": status,
        "_provider_status": {
            **provider_status,
            "provider": "deepseek",
            "provider_backed": False,
            "boundary": "Provider-backed success requires a completed external CLI call and validated JSON review.",
        },
    }


def _chunks(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    safe_size = max(1, size)
    return [items[index : index + safe_size] for index in range(0, len(items), safe_size)]


def _opencode_batch_cache_path(run_date: str, batch_index: int) -> Path:
    return artifact_dir(run_date) / "deepseek-review-batches" / f"batch-{batch_index:03d}.json"


def _load_opencode_batch_cache(selected: list[dict[str, Any]], *, run_date: str, batch_index: int) -> dict[str, Any] | None:
    path = _opencode_batch_cache_path(run_date, batch_index)
    if not path.exists():
        return None
    try:
        payload = read_json(path)
    except Exception:
        return None
    if payload.get("_provider_error"):
        return None
    errors = _validate_provider_reviews(selected, payload)
    provider_status = payload.get("provider_status", {})
    cached_ids = provider_status.get("batch_candidate_ids") or []
    expected_ids = [candidate_id(item) for item in selected]
    if errors or cached_ids != expected_ids or not provider_status.get("provider_backed"):
        return None
    payload["provider_status"] = {**dict(provider_status), "cache_hit": True}
    return payload


def _write_opencode_batch_cache(payload: dict[str, Any], *, run_date: str, batch_index: int) -> None:
    if payload.get("_provider_error"):
        return
    write_json(_opencode_batch_cache_path(run_date, batch_index), payload)


def _hydrate_provider_review_metadata(selected: list[dict[str, Any]], payload: dict[str, Any]) -> None:
    titles = {candidate_id(item): str(item.get("title") or item.get("candidate_title") or candidate_id(item)) for item in selected}
    reviews = [item for item in payload.get("reviews", []) if isinstance(item, dict)]
    if len(selected) == 1 and len(reviews) == 1:
        expected_id = candidate_id(selected[0])
        if str(reviews[0].get("candidate_id") or "") != expected_id:
            reviews[0]["candidate_id"] = expected_id
            payload.setdefault("provider_status", {})["candidate_id_hydrated_from_single_candidate_batch"] = True
        if not str(reviews[0].get("status") or "").strip() and any(field in reviews[0] for field in REQUIRED_REVIEW_FIELDS):
            reviews[0]["status"] = "success"
            payload.setdefault("provider_status", {})["review_status_hydrated_from_single_candidate_batch"] = True
    for review in reviews:
        if not isinstance(review, dict):
            continue
        cid = str(review.get("candidate_id") or "")
        if cid in titles and not str(review.get("candidate_title", "")).strip():
            review["candidate_title"] = titles[cid]


def _normalize_provider_payload_shape(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("reviews"), dict):
        payload = dict(payload)
        payload["reviews"] = [payload["reviews"]]
        return payload
    if "reviews" not in payload:
        for key in ("review", "provider_review", "deepseek_review", "candidate_review"):
            if isinstance(payload.get(key), dict):
                payload = dict(payload)
                payload["reviews"] = [payload[key]]
                return payload
        if any(field in payload for field in REQUIRED_REVIEW_FIELDS):
            return {"reviews": [payload]}
    return payload


def _opencode_model_candidates(model: str) -> list[str]:
    primary = str(model or "").strip()
    env_value = str(os.environ.get("DEEPSEEK_OPENCODE_FALLBACK_MODELS", "")).strip()
    fallback_models = [item.strip() for item in env_value.split(",") if item.strip()]
    if not fallback_models:
        fallback_models = list(DEFAULT_OPENCODE_FALLBACK_MODELS)
    ordered: list[str] = []
    seen: set[str] = set()
    for item in [primary, *fallback_models]:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered or [primary]


def _provider_auth_failure(error_text: str) -> bool:
    text = str(error_text or "").strip().lower()
    if not text:
        return False
    return any(
        marker in text
        for marker in (
            "invalid token",
            "status=401",
            "bad response status code 401",
            "token refresh failed: 401",
            "apierror:401",
        )
    )


def _redact_secret_text(text: str) -> str:
    return re.sub(r"(?i)(sk-[A-Za-z0-9_\-]{8,}|Bearer\s+\S+|api[_-]?key[=:]\s*\S+)", "[REDACTED]", str(text or ""))


def _global_opencode_config_paths() -> list[Path]:
    home = Path.home()
    return [
        home / ".config" / "opencode" / "opencode.json",
        home / ".opencode" / "opencode.json",
        home / ".config" / "opencode.json",
    ]


def _load_global_opencode_config() -> dict[str, Any]:
    for path in _global_opencode_config_paths():
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _openai_compatible_config(model: str) -> dict[str, str]:
    requested_model = str(model or DEFAULT_OPENCODE_MODEL).strip() or DEFAULT_OPENCODE_MODEL
    provider_id, model_key = (requested_model.split("/", 1) + [""])[:2] if "/" in requested_model else ("abrdns", requested_model)
    config = _load_global_opencode_config()
    provider = config.get("provider", {}).get(provider_id, {}) if isinstance(config.get("provider"), dict) else {}
    provider_options = provider.get("options", {}) if isinstance(provider, dict) else {}
    model_config = provider.get("models", {}).get(model_key, {}) if isinstance(provider.get("models"), dict) else {}
    base_url = str(os.environ.get("DEEPSEEK_OPENAI_BASE_URL") or provider_options.get("baseURL") or "").strip()
    api_key = str(os.environ.get("DEEPSEEK_OPENAI_API_KEY") or provider_options.get("apiKey") or "").strip()
    api_model = str(os.environ.get("DEEPSEEK_OPENAI_MODEL") or model_config.get("id") or model_key or requested_model).strip()
    return {
        "provider_id": provider_id,
        "requested_model": requested_model,
        "api_model": api_model,
        "base_url": base_url,
        "api_key": api_key,
    }


def _base_url_origin(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"


def _openai_compatible_chat_completion(prompt: str, *, model: str, timeout_sec: int) -> dict[str, Any]:
    config = _openai_compatible_config(model)
    result: dict[str, Any] = {
        "mode": "openai-compatible",
        "requested_model": config["requested_model"],
        "effective_model": config["api_model"],
        "base_url_origin": _base_url_origin(config["base_url"]),
        "exit_code": None,
        "timed_out": False,
        "clean_output": "",
        "raw_response_excerpt": "",
        "error": "",
    }
    if not config["base_url"]:
        result["error"] = "openai_compatible_config_missing_base_url"
        result["exit_code"] = 1
        return result
    if not config["api_key"]:
        result["error"] = "openai_compatible_config_missing_api_key"
        result["exit_code"] = 1
        return result
    url = config["base_url"].rstrip("/") + "/chat/completions"
    body = {
        "model": config["api_model"],
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict JSON-only scientific reviewer. Return exactly one RFC 8259 JSON object. "
                    "Do not return markdown, prose, code fences, or tool calls."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "stream": False,
        "max_tokens": int(os.environ.get("DEEPSEEK_OPENAI_MAX_TOKENS") or DEFAULT_OPENAI_COMPATIBLE_MAX_TOKENS),
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        error_body = ""
        try:
            error_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            error_body = ""
        result["exit_code"] = 1
        result["error"] = _redact_secret_text(f"openai_compatible_http_error:{exc.code}:{error_body[:500]}")
        return result
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        result["exit_code"] = 1
        result["timed_out"] = isinstance(exc, (TimeoutError, socket.timeout))
        result["error"] = _redact_secret_text(f"openai_compatible_request_error:{type(exc).__name__}:{exc}")
        return result
    except Exception as exc:
        result["exit_code"] = 1
        result["error"] = _redact_secret_text(f"openai_compatible_unexpected_error:{type(exc).__name__}:{exc}")
        return result
    result["exit_code"] = 0
    result["raw_response_excerpt"] = _redact_secret_text(raw[:2000])
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        result["exit_code"] = 1
        result["error"] = f"openai_compatible_invalid_response_json:{type(exc).__name__}:{exc}"
        return result
    result["response_model"] = payload.get("model")
    result["usage"] = payload.get("usage")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        result["error"] = "openai_compatible_response_missing_choices"
        return result
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content") if isinstance(message, dict) else ""
    if not isinstance(content, str) or not content.strip():
        result["error"] = "openai_compatible_empty_message_content"
        return result
    result["clean_output"] = content.strip()
    return result


def _openai_compatible_provider_batch(
    selected: list[dict[str, Any]],
    *,
    run_date: str,
    model: str,
    timeout_sec: int,
    batch_index: int,
    retries: int = DEFAULT_OPENCODE_RETRIES,
) -> dict[str, Any]:
    attempt_count = max(1, retries + 1)
    attempt_statuses: list[dict[str, Any]] = []
    retry_reason = ""
    previous_output = ""
    retry_details: list[str] = []
    last_error_payload: dict[str, Any] | None = None
    for model_attempt in range(1, attempt_count + 1):
        result = _openai_compatible_chat_completion(
            _render_opencode_prompt(
                selected,
                run_date=run_date,
                retry_reason=retry_reason,
                previous_output=previous_output,
                retry_details=retry_details,
                retry_compact=True,
            ),
            model=model,
            timeout_sec=timeout_sec,
        )
        status = {
            "mode": "openai-compatible",
            "requested_model": model,
            "effective_model": result.get("effective_model") or model,
            "response_model": result.get("response_model"),
            "base_url_origin": result.get("base_url_origin"),
            "exit_code": result.get("exit_code"),
            "timed_out": bool(result.get("timed_out")),
            "batch_index": batch_index,
            "batch_size": len(selected),
            "batch_candidate_ids": [candidate_id(item) for item in selected],
            "attempt": model_attempt,
            "attempt_count": attempt_count,
            "usage": result.get("usage"),
        }
        attempt_statuses.append(dict(status))
        if result.get("exit_code") != 0 or result.get("error") and not str(result.get("clean_output") or "").strip():
            provider_error = str(result.get("error") or f"openai_compatible_nonzero_exit:{result.get('exit_code')}")
            status["attempt_statuses"] = attempt_statuses
            status["raw_response_excerpt"] = str(result.get("raw_response_excerpt") or "")[:2000]
            last_error_payload = _provider_error_payload(
                provider_error,
                status="partial_provider_unavailable",
                provider_status=status,
            )
            retry_reason = "openai_compatible_provider_error"
            retry_details = [provider_error]
            continue
        try:
            payload = _extract_json_object(str(result.get("clean_output") or ""))
        except Exception as exc:
            previous_output = str(result.get("clean_output") or "")
            status["attempt_statuses"] = attempt_statuses
            status["clean_output_excerpt"] = previous_output[:500]
            status["raw_response_excerpt"] = str(result.get("raw_response_excerpt") or "")[:2000]
            last_error_payload = _provider_error_payload(
                f"openai_compatible_invalid_json:{type(exc).__name__}:{exc}",
                status="partial_provider_invalid",
                provider_status=status,
            )
            retry_reason = "openai_compatible_invalid_json"
            retry_details = [f"{type(exc).__name__}:{exc}"]
            continue
        payload = _normalize_provider_payload_shape(payload)
        payload["schema_version"] = "deepseek_review.v1"
        payload["run_date"] = run_date
        payload["status"] = "success"
        payload["provider_status"] = {
            **dict(payload.get("provider_status", {})),
            **status,
            "provider": "deepseek",
            "provider_backed": True,
            "attempt_statuses": attempt_statuses,
        }
        _hydrate_provider_review_metadata(selected, payload)
        validation_errors = _validate_provider_reviews(selected, payload)
        if validation_errors:
            previous_output = str(result.get("clean_output") or "")
            status["attempt_statuses"] = attempt_statuses
            status["validation_errors"] = validation_errors
            status["clean_output_excerpt"] = previous_output[:500]
            last_error_payload = _provider_error_payload(
                f"openai_compatible_invalid_review_payload:{';'.join(validation_errors)}",
                status="partial_provider_invalid",
                provider_status=status,
            )
            retry_reason = "openai_compatible_invalid_review_payload"
            retry_details = validation_errors
            continue
        return payload
    return last_error_payload or _provider_error_payload(
        "openai_compatible_attempts_exhausted",
        status="partial_provider_unavailable",
        provider_status={
            "mode": "openai-compatible",
            "requested_model": model,
            "effective_model": model,
            "exit_code": 1,
            "timed_out": False,
            "batch_index": batch_index,
            "batch_size": len(selected),
            "batch_candidate_ids": [candidate_id(item) for item in selected],
            "attempt_count": attempt_count,
            "attempt_statuses": attempt_statuses,
        },
    )


def _opencode_provider_batch(
    selected: list[dict[str, Any]],
    *,
    run_date: str,
    model: str,
    timeout_sec: int,
    batch_index: int,
    retries: int = DEFAULT_OPENCODE_RETRIES,
) -> dict[str, Any]:
    attempt_count = max(1, retries + 1)
    model_candidates = _opencode_model_candidates(model)
    attempt_statuses: list[dict[str, Any]] = []
    last_error_payload: dict[str, Any] | None = None
    for model_index, attempt_model in enumerate(model_candidates, start=1):
        retry_reason = ""
        previous_output = ""
        retry_details: list[str] = []
        for model_attempt in range(1, attempt_count + 1):
            overall_attempt = len(attempt_statuses) + 1
            result = run_opencode_cli(
                _render_opencode_prompt(
                    selected,
                    run_date=run_date,
                    retry_reason=retry_reason,
                    previous_output=previous_output,
                    retry_details=retry_details,
                    retry_compact=True,
                ),
                model=attempt_model,
                timeout_sec=timeout_sec,
                title=f"{run_date}-v2-deepseek-scientific-review-batch-{batch_index}-model-{model_index}-attempt-{model_attempt}",
            )
            status = {
                "mode": "opencode",
                "requested_model": attempt_model,
                "primary_requested_model": model,
                "effective_model": result.get("effective_model") or attempt_model,
                "exit_code": result.get("exit_code"),
                "timed_out": bool(result.get("timed_out")),
                "event_count": result.get("event_count", 0),
                "batch_index": batch_index,
                "batch_size": len(selected),
                "batch_candidate_ids": [candidate_id(item) for item in selected],
                "attempt": overall_attempt,
                "attempt_count": attempt_count * len(model_candidates),
                "model_candidate_index": model_index,
                "model_candidate_count": len(model_candidates),
                "model_attempt": model_attempt,
                "model_attempt_count": attempt_count,
                "model_fallback_used": attempt_model != model,
            }
            attempt_statuses.append(dict(status))
            if result.get("timed_out"):
                status["attempt_statuses"] = attempt_statuses
                last_error_payload = _provider_error_payload(f"opencode_timeout:{timeout_sec}s", status="partial_provider_unavailable", provider_status=status)
                retry_reason = "opencode_timeout"
                retry_details = [f"timeout:{timeout_sec}s"]
                continue
            if result.get("exit_code") != 0:
                status["attempt_statuses"] = attempt_statuses
                last_error_payload = _provider_error_payload(str(result.get("error") or f"opencode_nonzero_exit:{result.get('exit_code')}"), status="partial_provider_unavailable", provider_status=status)
                retry_reason = "opencode_nonzero_exit"
                retry_details = [str(result.get("error") or f"exit_code:{result.get('exit_code')}")]
                continue
            if result.get("error") and not str(result.get("clean_output") or "").strip():
                provider_error = str(result.get("error"))
                status["attempt_statuses"] = attempt_statuses
                status["raw_stdout_excerpt"] = str(result.get("raw_stdout") or "")[:2000]
                status["raw_stderr_excerpt"] = str(result.get("raw_stderr") or "")[:1000]
                status["provider_event_errors"] = list(result.get("event_errors") or [])[:8]
                last_error_payload = _provider_error_payload(
                    provider_error,
                    status="partial_provider_unavailable",
                    provider_status=status,
                )
                if _provider_auth_failure(provider_error) and model_index < len(model_candidates):
                    status["model_fallback_triggered"] = True
                    status["fallback_next_model"] = model_candidates[model_index]
                    break
                retry_reason = "opencode_provider_error"
                retry_details = [provider_error]
                continue
            try:
                payload = _extract_json_object(str(result.get("clean_output") or ""))
            except Exception as exc:
                previous_output = str(result.get("clean_output") or "")
                status["attempt_statuses"] = attempt_statuses
                status["clean_output_excerpt"] = previous_output[:500]
                status["raw_stdout_excerpt"] = str(result.get("raw_stdout") or "")[:2000]
                status["raw_stderr_excerpt"] = str(result.get("raw_stderr") or "")[:1000]
                last_error_payload = _provider_error_payload(f"opencode_invalid_json:{type(exc).__name__}:{exc}", status="partial_provider_invalid", provider_status=status)
                retry_reason = "opencode_invalid_json"
                retry_details = [f"{type(exc).__name__}:{exc}"]
                continue
            payload = _normalize_provider_payload_shape(payload)
            payload["schema_version"] = "deepseek_review.v1"
            payload["run_date"] = run_date
            payload["status"] = "success"
            payload["provider_status"] = {
                **dict(payload.get("provider_status", {})),
                **status,
                "provider": "deepseek",
                "provider_backed": True,
                "attempt_statuses": attempt_statuses,
                "model_candidates": model_candidates,
            }
            _hydrate_provider_review_metadata(selected, payload)
            validation_errors = _validate_provider_reviews(selected, payload)
            if validation_errors:
                previous_output = str(result.get("clean_output") or "")
                status["attempt_statuses"] = attempt_statuses
                status["validation_errors"] = validation_errors
                status["clean_output_excerpt"] = previous_output[:500]
                status["raw_stdout_excerpt"] = str(result.get("raw_stdout") or "")[:2000]
                status["raw_stderr_excerpt"] = str(result.get("raw_stderr") or "")[:1000]
                last_error_payload = _provider_error_payload(
                    f"opencode_invalid_review_payload:{';'.join(validation_errors)}",
                    status="partial_provider_invalid",
                    provider_status=status,
                )
                retry_reason = "opencode_invalid_review_payload"
                retry_details = validation_errors
                continue
            return payload
    return last_error_payload or _provider_error_payload(
        "opencode_attempts_exhausted",
        status="partial_provider_unavailable",
        provider_status={
            "mode": "opencode",
            "requested_model": model,
            "primary_requested_model": model,
            "effective_model": model,
            "exit_code": 1,
            "timed_out": False,
            "event_count": 0,
            "batch_index": batch_index,
            "batch_size": len(selected),
            "batch_candidate_ids": [candidate_id(item) for item in selected],
            "attempt_count": attempt_count * len(model_candidates),
            "attempt_statuses": attempt_statuses,
            "model_candidates": model_candidates,
        },
    )


def _merge_opencode_provider_batches(
    selected: list[dict[str, Any]],
    *,
    run_date: str,
    model: str,
    batch_payloads: list[dict[str, Any]],
    batch_size: int,
    mode: str = "opencode",
) -> dict[str, Any]:
    batch_statuses = [dict(payload.get("_provider_status") or payload.get("provider_status") or {}) for payload in batch_payloads]
    failures = [payload for payload in batch_payloads if payload.get("_provider_error")]
    reviews: list[dict[str, Any]] = []
    for payload in batch_payloads:
        reviews.extend([item for item in payload.get("reviews", []) if isinstance(item, dict)])
    if failures:
        failure_errors = [str(failure.get("_provider_error") or "provider_batch_failed") for failure in failures]
        failure_status = (
            "partial_provider_invalid"
            if any(str(failure.get("_provider_failure_status")) == "partial_provider_invalid" for failure in failures)
            else "partial_provider_unavailable"
        )
        by_id = {candidate_id(item): item for item in selected}
        reviewed_ids = {str(review.get("candidate_id")) for review in reviews if str(review.get("candidate_id", "")).strip()}
        failed_ids: list[str] = []
        for failure in failures:
            status = dict(failure.get("_provider_status") or failure.get("provider_status") or {})
            failed_ids.extend(str(value) for value in status.get("batch_candidate_ids", []) if value)
        fallback_ids = [cid for cid in [*failed_ids, *by_id] if cid in by_id and cid not in reviewed_ids]
        fallback_ids = list(dict.fromkeys(fallback_ids))
        fallback_reviews = [
            _provider_failure_fallback_review_candidate(by_id[cid], ";".join(failure_errors))
            for cid in fallback_ids
        ]
        provider_status = {
            "provider": "deepseek",
            "mode": mode,
            "requested_model": model,
            "effective_model": model,
            "exit_code": 1,
            "timed_out": any(bool(status.get("timed_out")) for status in batch_statuses),
            "event_count": sum(int(status.get("event_count") or 0) for status in batch_statuses),
            "batch_count": len(batch_payloads),
            "batch_size": batch_size,
            "batch_statuses": batch_statuses,
            "batch_candidate_ids": [candidate_id(item) for item in selected],
            "provider_backed": False,
            "candidate_level_fail_closed": True,
            "provider_backed_success_count": len(reviews),
            "provider_fallback_count": len(fallback_reviews),
            "provider_error": ";".join(failure_errors),
            "boundary": "Failed provider candidates are marked failed_fallback_only and cannot pass survival.",
        }
        return {
            "schema_version": "deepseek_review.v1",
            "run_date": run_date,
            "status": failure_status,
            "provider_status": provider_status,
            "reviews": [*reviews, *fallback_reviews],
        }

    return {
        "schema_version": "deepseek_review.v1",
        "run_date": run_date,
        "status": "success",
        "provider_status": {
            "provider": "deepseek",
            "provider_backed": True,
            "mode": mode,
            "requested_model": model,
            "effective_model": model,
            "exit_code": 0,
            "timed_out": False,
            "event_count": sum(int(status.get("event_count") or 0) for status in batch_statuses),
            "batch_count": len(batch_payloads),
            "batch_size": batch_size,
            "batch_statuses": batch_statuses,
            "batch_candidate_ids": [candidate_id(item) for item in selected],
        },
        "reviews": reviews,
    }


def _opencode_provider_payload(
    selected: list[dict[str, Any]],
    *,
    run_date: str,
    model: str,
    timeout_sec: int,
    batch_size: int = DEFAULT_OPENCODE_BATCH_SIZE,
    retries: int = DEFAULT_OPENCODE_RETRIES,
) -> dict[str, Any]:
    if not selected:
        return {
            "schema_version": "deepseek_review.v1",
            "run_date": run_date,
            "status": "success",
            "provider_status": {
                "provider": "deepseek",
                "provider_backed": True,
                "mode": "opencode",
                "requested_model": model,
                "effective_model": model,
                "exit_code": 0,
                "batch_count": 0,
                "batch_size": max(1, batch_size),
            },
            "reviews": [],
        }
    batches = _chunks(selected, batch_size)
    payloads: list[dict[str, Any]] = []
    for index, batch in enumerate(batches, start=1):
        cached_payload = _load_opencode_batch_cache(batch, run_date=run_date, batch_index=index)
        if cached_payload is not None:
            payloads.append(cached_payload)
            continue
        payload = _opencode_provider_batch(
            batch,
            run_date=run_date,
            model=model,
            timeout_sec=timeout_sec,
            batch_index=index,
            retries=retries,
        )
        if not payload.get("_provider_error"):
            _write_opencode_batch_cache(payload, run_date=run_date, batch_index=index)
        payloads.append(payload)
        if payload.get("_provider_error"):
            status = payload.setdefault("_provider_status", payload.get("provider_status") or {})
            if isinstance(status, dict):
                status["batch_fail_fast_after_provider_failure"] = True
                status["unattempted_batch_count"] = max(0, len(batches) - index)
            break
    if len(payloads) == 1 and len(batches) == 1:
        payloads[0].setdefault("provider_status", {})
        payloads[0]["provider_status"].setdefault("batch_count", 1)
        payloads[0]["provider_status"].setdefault("batch_size", max(1, batch_size))
        return payloads[0]
    return _merge_opencode_provider_batches(
        selected,
        run_date=run_date,
        model=model,
        batch_payloads=payloads,
        batch_size=max(1, batch_size),
    )


def _openai_compatible_provider_payload(
    selected: list[dict[str, Any]],
    *,
    run_date: str,
    model: str,
    timeout_sec: int,
    batch_size: int = DEFAULT_OPENCODE_BATCH_SIZE,
    retries: int = DEFAULT_OPENCODE_RETRIES,
) -> dict[str, Any]:
    if not selected:
        return {
            "schema_version": "deepseek_review.v1",
            "run_date": run_date,
            "status": "success",
            "provider_status": {
                "provider": "deepseek",
                "provider_backed": True,
                "mode": "openai-compatible",
                "requested_model": model,
                "effective_model": model,
                "exit_code": 0,
                "batch_count": 0,
                "batch_size": max(1, batch_size),
            },
            "reviews": [],
        }
    batches = _chunks(selected, batch_size)
    payloads: list[dict[str, Any]] = []
    for index, batch in enumerate(batches, start=1):
        payload = _openai_compatible_provider_batch(
            batch,
            run_date=run_date,
            model=model,
            timeout_sec=timeout_sec,
            batch_index=index,
            retries=retries,
        )
        payloads.append(payload)
        if payload.get("_provider_error"):
            status = payload.setdefault("_provider_status", payload.get("provider_status") or {})
            if isinstance(status, dict):
                status["batch_fail_fast_after_provider_failure"] = True
                status["unattempted_batch_count"] = max(0, len(batches) - index)
            break
    if len(payloads) == 1 and len(batches) == 1:
        payloads[0].setdefault("provider_status", {})
        payloads[0]["provider_status"].setdefault("batch_count", 1)
        payloads[0]["provider_status"].setdefault("batch_size", max(1, batch_size))
        return payloads[0]
    return _merge_opencode_provider_batches(
        selected,
        run_date=run_date,
        model=model,
        batch_payloads=payloads,
        batch_size=max(1, batch_size),
        mode="openai-compatible",
    )


def _validate_provider_reviews(selected: list[dict[str, Any]], payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    provider = payload.get("provider_status", {})
    allow_candidate_fallback = bool(provider.get("candidate_level_fail_closed"))
    if not provider.get("provider_backed") and not allow_candidate_fallback:
        errors.append("provider_status_not_provider_backed")
    if provider.get("provider") != "deepseek":
        errors.append(f"provider_not_deepseek:{provider.get('provider')}")
    reviews = [item for item in payload.get("reviews", []) if isinstance(item, dict)]
    selected_ids = [candidate_id(item) for item in selected]
    seen: dict[str, int] = {}
    for review in reviews:
        cid = str(review.get("candidate_id"))
        seen[cid] = seen.get(cid, 0) + 1
        if cid not in selected_ids:
            errors.append(f"unexpected_provider_review:{cid}")
    for cid, count in seen.items():
        if count > 1:
            errors.append(f"duplicate_provider_review:{cid}")
    by_id = {str(item.get("candidate_id")): item for item in reviews}
    for item in selected:
        cid = candidate_id(item)
        review = by_id.get(cid)
        if not review:
            errors.append(f"missing_provider_review:{cid}")
            continue
        review_status = review.get("status")
        if review_status == "failed_fallback_only" and allow_candidate_fallback:
            if review.get("survivability_label") != "reject_fatal":
                errors.append(f"provider_fallback_review_not_reject_fatal:{cid}")
            if review.get("allowed_next_stage") != "stop":
                errors.append(f"provider_fallback_review_not_stop:{cid}")
        elif review_status != "success":
            errors.append(f"provider_review_not_success:{cid}:{review.get('status')}")
        for field in REQUIRED_REVIEW_FIELDS:
            if field not in review:
                errors.append(f"provider_review_missing_field:{cid}:{field}")
                continue
            value = review.get(field)
            if field != "fatal_flaw" and not isinstance(value, str):
                errors.append(f"provider_review_invalid_field_type:{cid}:{field}:{type(value).__name__}")
            if field not in OPTIONAL_EMPTY_REVIEW_FIELDS and not str(value or "").strip():
                errors.append(f"provider_review_empty_field:{cid}:{field}")
        if not str(review.get("candidate_title", "")).strip():
            errors.append(f"provider_review_empty_field:{cid}:candidate_title")
        a_plus_b_risk = review.get("a_plus_b_risk")
        if not isinstance(a_plus_b_risk, str) or a_plus_b_risk not in A_PLUS_B_RISK:
            errors.append(f"provider_review_invalid_a_plus_b_risk:{cid}:{a_plus_b_risk}")
        survivability_label = review.get("survivability_label")
        if not isinstance(survivability_label, str) or survivability_label not in SURVIVABILITY_LABELS:
            errors.append(f"provider_review_invalid_survivability_label:{cid}:{survivability_label}")
        allowed_next_stage = review.get("allowed_next_stage")
        if not isinstance(allowed_next_stage, str) or allowed_next_stage not in ALLOWED_NEXT_STAGES:
            errors.append(f"provider_review_invalid_allowed_next_stage:{cid}:{allowed_next_stage}")
    return errors


def _schema_safe_deepseek_reviews(reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe_reviews: list[dict[str, Any]] = []
    for review in reviews:
        safe = dict(review)
        if safe.get("status") not in {"success", "failed_fallback_only"}:
            safe["status"] = "failed_fallback_only"
        if not isinstance(safe.get("a_plus_b_risk"), str) or safe.get("a_plus_b_risk") not in A_PLUS_B_RISK:
            safe["a_plus_b_risk"] = "fatal"
        if not isinstance(safe.get("survivability_label"), str) or safe.get("survivability_label") not in SURVIVABILITY_LABELS:
            safe["survivability_label"] = "reject_fatal"
        if not isinstance(safe.get("allowed_next_stage"), str) or safe.get("allowed_next_stage") not in ALLOWED_NEXT_STAGES:
            safe["allowed_next_stage"] = "stop"
        for field in REQUIRED_REVIEW_FIELDS:
            if field == "fatal_flaw":
                if not isinstance(safe.get(field), (str, bool)):
                    safe[field] = True
                continue
            if not isinstance(safe.get(field), str) or not str(safe.get(field, "")).strip():
                safe[field] = "not_provided"
        safe_reviews.append(safe)
    return safe_reviews


def build_payload(selected: list[dict[str, Any]], *, run_date: str, provider_payload: dict[str, Any] | None = None) -> tuple[dict[str, Any], int]:
    provider_payload = provider_payload or {}
    if not selected:
        payload = {
            "schema_version": "deepseek_review.v1",
            "run_date": run_date,
            "status": "success_empty_selection",
            "reviews": [],
            "provider_status": {
                "provider": "deepseek",
                "provider_backed": False,
                "mode": "no_selection",
                "boundary": "No selected candidates; no model review was required or used as gate evidence.",
            },
            "artifact_hashes": artifact_hashes(run_date, ["selected-candidates.json"]),
        }
        return payload, 0
    if provider_payload and not provider_payload.get("_provider_error"):
        errors = _validate_provider_reviews(selected, provider_payload)
        provider_status = {**dict(provider_payload.get("provider_status", {})), "validation_errors": errors}
        candidate_level_fallback = bool(provider_status.get("candidate_level_fail_closed"))
        status = (
            str(provider_payload.get("status") or "partial_provider_invalid")
            if candidate_level_fallback
            else ("success" if not errors and selected else "partial_provider_invalid")
        )
        if status != "success" and not candidate_level_fallback:
            provider_status["provider_backed"] = False
        payload = {
            "schema_version": "deepseek_review.v1",
            "run_date": run_date,
            "status": status,
            "reviews": _schema_safe_deepseek_reviews([item for item in provider_payload.get("reviews", []) if isinstance(item, dict)])
            if status != "success"
            else provider_payload.get("reviews", []),
            "provider_status": provider_status,
            "artifact_hashes": artifact_hashes(run_date, ["selected-candidates.json"]),
        }
        return payload, 0 if (status == "success" or (candidate_level_fallback and not errors)) else 2

    reviews = [_fallback_review_candidate(item) for item in selected]
    provider_error = provider_payload.get("_provider_error", "deepseek_provider_unavailable")
    provider_status = provider_payload.get("_provider_status") or {
        "provider": "deepseek",
        "provider_backed": False,
        "mode": "deterministic_strict_local_adapter",
        "provider_error": provider_error,
        "boundary": "Fallback artifact only; survival_decision must not accept this as provider-backed success.",
    }
    provider_status["provider_error"] = provider_error
    payload = {
        "schema_version": "deepseek_review.v1",
        "run_date": run_date,
        "status": provider_payload.get("_provider_failure_status") or ("partial_provider_unavailable" if reviews else "partial_empty_selection"),
        "reviews": reviews,
        "provider_status": provider_status,
        "artifact_hashes": artifact_hashes(run_date, ["selected-candidates.json"]),
    }
    return payload, 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-date", required=True)
    parser.add_argument("--provider", choices=["json", "opencode", "openai-compatible", "none"], default="json")
    parser.add_argument("--provider-review-json", default=os.environ.get("DEEPSEEK_REVIEW_JSON", ""))
    parser.add_argument("--model", default=DEFAULT_OPENCODE_MODEL)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--provider-batch-size", type=int, default=DEFAULT_OPENCODE_BATCH_SIZE)
    parser.add_argument("--provider-retries", type=int, default=DEFAULT_OPENCODE_RETRIES)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ensure_v2_dirs(args.run_date)
    selected = read_json(artifact_dir(args.run_date) / "selected-candidates.json").get("selected", [])
    if args.provider == "opencode":
        provider_payload = _opencode_provider_payload(
            selected,
            run_date=args.run_date,
            model=args.model,
            timeout_sec=args.timeout,
            batch_size=args.provider_batch_size,
            retries=args.provider_retries,
        )
    elif args.provider == "openai-compatible":
        provider_payload = _openai_compatible_provider_payload(
            selected,
            run_date=args.run_date,
            model=args.model,
            timeout_sec=args.timeout,
            batch_size=args.provider_batch_size,
            retries=args.provider_retries,
        )
    elif args.provider_review_json:
        provider_payload = _load_provider_payload(args.provider_review_json)
    else:
        provider_payload = {}
    payload, exit_code = build_payload(selected, run_date=args.run_date, provider_payload=provider_payload)
    write_run_artifact(args.run_date, "deepseek-review.json", payload, state="attacked_by_deepseek", dry_run=args.dry_run)
    safe_print(f"DEEPSEEK_REVIEW: status={payload['status']} reviews={len(payload['reviews'])}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
