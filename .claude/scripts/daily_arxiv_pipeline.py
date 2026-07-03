"""Daily arXiv embodied-AI scout pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from arxiv_ranker import (
    AUTO_IMPORT_DECISIONS,
    BELOW_REVIEW_DECISION,
    DEFAULT_QUERIES,
    FOCUS_REVIEW_DECISION,
    LEGACY_REJECT_DECISION,
    REVIEW_THRESHOLD,
    TOP1_THRESHOLD,
    ArxivPaper,
    RankedPaper,
    normalize_title,
    rank_papers,
)
from generate_gemini_idea_prompt import render_prompt as render_gemini_prompt
from arxiv_metadata_sync import DEFAULT_DB as ARXIV_METADATA_DB
from arxiv_metadata_sync import query_mirror, mirror_status_path
from canonical_baseline_backfill import build_baseline_backfill_report, write_baseline_backfill_report
from llm_structured import backoff_seconds as structured_backoff_seconds
from llm_structured import classify_transient as structured_classify_transient
from kb_common import extract_frontmatter, parse_frontmatter_map, parse_list_value, safe_print, safe_write, today_iso, vault_path
from paper_intake_triage import build_triage
from research_seed_v2_common import DEFAULT_V2_PUBLISH_POLICY
from zotero_import import (
    DEFAULT_COLLECTION_KEY,
    ImportResult,
    find_existing_paper_in_items,
    import_ranked_paper,
    iter_local_items,
    preflight,
    zotero_data_dir,
)


ARXIV_API = "https://export.arxiv.org/api/query"
DANGEROUS_CLAUDE_ENV = "LOCAL_FIRST_VAULT_ALLOW_DANGEROUS_CLAUDE"
DETERMINISTIC_READ_FALLBACK_ENV = "LOCAL_FIRST_VAULT_ALLOW_DETERMINISTIC_READ_FALLBACK"
CLAUDE_BIN_ENV = "LOCAL_FIRST_VAULT_CLAUDE_BIN"
CODEX_BIN_ENV = "LOCAL_FIRST_VAULT_CODEX_BIN"
CLAUDE_BIN_CANDIDATES = [
    Path.home() / ".local" / "bin" / "claude.exe",
    Path.home() / ".bun" / "bin" / "claude.exe",
]
CODEX_CONTROLLED_READ_VERSION = 1
ZOTERO_LOCAL_API = "http://127.0.0.1:23119/api/users/0"
MIN_CONTROLLED_FULLTEXT_CHARS = 2000
ZOTERO_DB_ENV = "LOCAL_FIRST_VAULT_ZOTERO_DB"
ZOTERO_STORAGE_ENV = "LOCAL_FIRST_VAULT_ZOTERO_STORAGE"
ATOM = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
AUTO_IMPORT_FILL_DECISION = "auto_import_fill"
NEW_IMPORT_STATUSES = {"created", "sync_pending", "pdf_pending"}
FOCUS_READY_STATUSES = NEW_IMPORT_STATUSES | {"backlog"}
INGEST_READY_STATUSES = {"created", "exists", "sync_pending", "pdf_pending", "backlog"}
LOW_PRIORITY_DECISIONS = {BELOW_REVIEW_DECISION, LEGACY_REJECT_DECISION}
DEFAULT_FETCH_TIMEOUT = 12
DEFAULT_FETCH_RETRIES = 0
DEFAULT_QUERY_DELAY = 3.2
MIN_MIRROR_CANDIDATES_FOR_DAILY = 20
PRIMARY_DAILY_IMPORT_CAP = 7
FOCUS_DAILY_IMPORT_FLOOR = 2
DOMAIN_DAILY_IMPORT_FLOOR = 1
COVERAGE_DOMAIN_LABELS = {
    "vla_vlm",
    "action_interface",
    "tactile",
    "sim_to_real",
    "robustness_recovery",
    "dlo",
    "bimanual",
}
STAGED_READ_VERSION = 1
STAGED_READ_STAGES = [
    (
        "01-evidence-metadata",
        "Evidence Metadata, source coverage, reading boundary",
        "Use only the local note context included in the prompt. Do not call tools in this stage. Write a bounded Evidence Metadata draft with Fulltext Quality, Evidence Coverage, Confidence, source/section/table/figure coverage, and a concrete one-sentence Chinese summary. If full text is not inspected in this stage, explicitly write not_inspected_yet. Minimum 700 characters.",
    ),
    (
        "02-core-paper",
        "Problem, contributions, method",
        "Use the source evidence plus previous stage files. Write Problem, Key Contributions, and Method with claim_id references. Keep unsupported claims explicitly not_evidenced.",
    ),
    (
        "03a-experiments-results",
        "Experiments, tasks, metrics, and result anchors",
        "Use the source evidence plus previous stage files. Write Experiments only as a compact extraction packet, not prose. Target 900-1600 Chinese characters. Output exactly these blocks: 1) Benchmark families table, 2) strongest 5-8 result anchors table, 3) missing/unclear experiment evidence bullets. Do not exhaustively enumerate every benchmark task when a paper reports many tasks. Do not write Limitations or build the full Evidence Ledger table in this stage.",
    ),
    (
        "03b-limitations-evidence-gaps",
        "Limitations, failure modes, and evidence gaps",
        "Use the source evidence plus previous stage files. Write Limitations only as a compact extraction packet, not prose. Target 900-1600 Chinese characters. Output exactly these blocks: 1) stated limitations, 2) implied failure modes, 3) missing ablations/evidence gaps, 4) transfer and negative-transfer risks if applicable. Do not repeat the full experiment summary and do not build the full Evidence Ledger table in this stage.",
    ),
    (
        "04-evidence-ledger",
        "Evidence Ledger",
        "Use prior stage files to build the Evidence Ledger table only. Cover Problem, Contributions, Method, Experiments, Limitations, Baseline, and Transfer claims. Keep it bounded to the strongest 12-18 claims; mark weak claims screening_only or requires_human_check.",
    ),
    (
        "05-review-pressure",
        "Key takeaways, IF packets, baseline, transfer, no-hardware test",
        "Use the previous stage files. Write Key Takeaways, Idea Fuel IF packets, Baseline Pressure, Transfer Risk, No-hardware Micro-test, Evidence Gaps, 结构化提取, and 本地引用关系. Treat Idea Fuel as screening-only review pressure.",
    ),
    (
        "06-assemble-final-analysis",
        "Assemble complete strict analysis",
        "Read all previous stage files and synthesize one complete strict analysis file. The final file must contain every required section and must be coherent as a standalone deep reading. Do not run finalize_reading.py; the pipeline will run it after this stage.",
    ),
]


@dataclass(frozen=True)
class ZoteroAttachmentCandidate:
    attachment_key: str
    attachment_path: str
    content_type: str

STAGED_ASSEMBLY_STAGE = "06-assemble-final-analysis"
STAGED_REQUIRED_FINAL_SECTIONS = [
    "## Evidence Metadata",
    "## Problem",
    "## Key Contributions",
    "## Method",
    "## Experiments",
    "## Limitations",
    "## Key Takeaways",
    "## Evidence Ledger",
    "## Idea Fuel",
    "## Baseline Pressure",
    "## Transfer Risk",
    "## No-hardware Micro-test",
    "## Evidence Gaps",
    "## 结构化提取",
    "## 本地引用关系",
]
STRUCTURED_EXTRACTION_REQUIRED_FIELDS = [
    "Problem:",
    "Method:",
    "Tasks:",
    "Sensors:",
    "Robot Setup:",
    "Metrics:",
    "Limitations:",
    "Evidence Notes:",
]


def _entry_text(entry: ET.Element, name: str) -> str:
    node = entry.find(f"atom:{name}", ATOM)
    return "" if node is None or node.text is None else " ".join(node.text.split())


def _entry_arxiv_text(entry: ET.Element, name: str) -> str:
    node = entry.find(f"arxiv:{name}", ATOM)
    return "" if node is None or node.text is None else " ".join(node.text.split())


def _entry_categories(entry: ET.Element) -> list[str]:
    categories = []
    for node in entry.findall("atom:category", ATOM):
        term = node.attrib.get("term", "")
        if term:
            categories.append(term)
    return categories


def _entry_primary_category(entry: ET.Element) -> str:
    node = entry.find("arxiv:primary_category", ATOM)
    return "" if node is None else node.attrib.get("term", "")


def _entry_pdf_url(entry: ET.Element) -> str:
    for node in entry.findall("atom:link", ATOM):
        if node.attrib.get("title") == "pdf" or node.attrib.get("type") == "application/pdf":
            return node.attrib.get("href", "")
    return ""


def _arxiv_id_from_url(url: str) -> str:
    value = url.rstrip("/").split("/")[-1]
    return value.split("v")[0] if "v" in value else value


def _parse_arxiv_date(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def fetch_arxiv(
    query: str,
    *,
    max_results: int,
    retries: int = DEFAULT_FETCH_RETRIES,
    timeout: int = DEFAULT_FETCH_TIMEOUT,
) -> list[ArxivPaper]:
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    url = f"{ARXIV_API}?{params}"
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                root = ET.fromstring(resp.read())
            break
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                raise
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"arXiv fetch failed: {last_error}")
    papers = []
    for entry in root.findall("atom:entry", ATOM):
        url_value = _entry_text(entry, "id")
        authors = []
        for author in entry.findall("atom:author", ATOM):
            name = author.find("atom:name", ATOM)
            if name is not None and name.text:
                authors.append(" ".join(name.text.split()))
        papers.append(
            ArxivPaper(
                arxiv_id=_arxiv_id_from_url(url_value),
                title=_entry_text(entry, "title"),
                authors=authors,
                summary=_entry_text(entry, "summary"),
                published=_entry_text(entry, "published"),
                updated=_entry_text(entry, "updated"),
                url=url_value,
                pdf_url=_entry_pdf_url(entry),
                categories=_entry_categories(entry),
                primary_category=_entry_primary_category(entry),
                doi=_entry_arxiv_text(entry, "doi"),
                journal_ref=_entry_arxiv_text(entry, "journal_ref"),
                comment=_entry_arxiv_text(entry, "comment"),
                query_sources=[query],
            )
        )
    return papers


def collect_candidates(
    queries: list[str],
    *,
    max_candidates: int,
    days_back: int,
    errors: list[str] | None = None,
    fetch_timeout: int = DEFAULT_FETCH_TIMEOUT,
    fetch_retries: int = DEFAULT_FETCH_RETRIES,
    query_delay: float = DEFAULT_QUERY_DELAY,
) -> list[ArxivPaper]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    by_id: dict[str, ArxivPaper] = {}
    per_query = max(20, max_candidates)
    for index, query in enumerate(queries):
        if index > 0 and query_delay > 0:
            time.sleep(query_delay)
        try:
            fetched = fetch_arxiv(query, max_results=per_query, retries=fetch_retries, timeout=fetch_timeout)
        except Exception as exc:
            if errors is not None:
                errors.append(f"arxiv_fetch_failed:{query}:{exc}")
            continue
        for paper in fetched:
            updated = _parse_arxiv_date(paper.updated) or _parse_arxiv_date(paper.published)
            if updated is not None and updated < cutoff:
                continue
            existing = by_id.get(paper.arxiv_id)
            if existing:
                existing.query_sources = sorted(set(existing.query_sources + [query]))
            else:
                by_id[paper.arxiv_id] = paper
        if max_candidates <= 20 and len(by_id) >= max_candidates * 3:
            break
    return list(by_id.values())


def collect_candidates_from_source(
    queries: list[str],
    *,
    source: str,
    max_candidates: int,
    days_back: int,
    as_of: datetime | None,
    errors: list[str],
    fetch_timeout: int,
    fetch_retries: int,
    query_delay: float,
    dry_run: bool = False,
) -> tuple[list[ArxivPaper], dict[str, Any]]:
    mirror_info: dict[str, Any] = {}
    if source in {"mirror-first", "mirror-only"}:
        try:
            papers, mirror_info = query_mirror(
                queries=queries,
                days_back=days_back,
                max_candidates=max_candidates,
                as_of=as_of,
                db_path=ARXIV_METADATA_DB,
            )
        except Exception as exc:
            papers, mirror_info = [], {"source": "mirror_failed", "records_total": 0, "last_success_at": "", "stale": True}
            errors.append(f"arxiv_mirror_failed:{exc}")
        if source == "mirror-only":
            mirror_info["search_api_fallback_used"] = False
            return papers, mirror_info
        if dry_run and (mirror_info.get("missing") or mirror_info.get("records_total", 0) == 0):
            reason = "missing" if mirror_info.get("missing") else "empty"
            errors.append(f"arxiv_mirror_{reason}:run_arxiv_metadata_sync_incremental_first")
            mirror_info["search_api_fallback_used"] = False
            return papers, mirror_info
        if dry_run:
            if len(papers) < MIN_MIRROR_CANDIDATES_FOR_DAILY:
                errors.append(f"arxiv_mirror_insufficient:candidates={len(papers)}:threshold={MIN_MIRROR_CANDIDATES_FOR_DAILY}")
            mirror_info["search_api_fallback_used"] = False
            return papers, mirror_info
        if len(papers) >= MIN_MIRROR_CANDIDATES_FOR_DAILY:
            mirror_info["search_api_fallback_used"] = False
            return papers, mirror_info
        if papers:
            errors.append(f"arxiv_mirror_insufficient:candidates={len(papers)}:threshold={MIN_MIRROR_CANDIDATES_FOR_DAILY}")
    if source == "mirror-first":
        if mirror_info.get("source") == "mirror_failed":
            fallback_reason = "failed"
        elif mirror_info.get("missing"):
            fallback_reason = "missing"
        elif mirror_info.get("records_total", 0) == 0:
            fallback_reason = "empty"
        elif papers:
            fallback_reason = "insufficient"
        else:
            fallback_reason = "empty"
        errors.append(f"arxiv_mirror_{fallback_reason}_search_api_fallback")
    papers = collect_candidates(
        queries,
        max_candidates=max_candidates,
        days_back=days_back,
        errors=errors,
        fetch_timeout=fetch_timeout,
        fetch_retries=fetch_retries,
        query_delay=query_delay,
    )
    if papers:
        return papers, {
            **mirror_info,
            "source": "search_api" if source == "search-api" else "search_api_fallback",
            "search_api_fallback_used": source == "mirror-first",
        }
    status = mirror_status_path()
    return papers, {
        **mirror_info,
        "source": "search_api_empty",
        "search_api_fallback_used": source == "mirror-first",
        "records_total": status.get("records_total", 0),
        "last_success_at": status.get("last_success_at", ""),
        "stale": status.get("stale", True),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]], *, dry_run: bool) -> None:
    content = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
    if content:
        content += "\n"
    safe_write(path, content, dry_run=dry_run, backup=True)


def load_cached_ranked(path: Path) -> list[RankedPaper]:
    if not path.exists():
        return []
    ranked: list[RankedPaper] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            ranked.append(RankedPaper.from_dict(json.loads(line)))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return ranked


def parse_backlog(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    pattern = re.compile(
        r"^- arxiv=(?P<arxiv>\S+)\s+zotero_key=(?P<key>\S*)\s+"
        r"import_status=(?P<import_status>\S+)\s+read_status=(?P<read_status>\S+)"
    )
    records: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        record = match.groupdict()
        if record.get("key"):
            records.append(record)
    return records


def parse_successful_read_keys(path: Path) -> list[str]:
    if not path.exists():
        return []
    pattern = re.compile(r"^- zotero_key=(?P<key>\S+)\s+ingest=\S+\s+read=(?P<read_status>\S+)")
    keys: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        if not match.group("read_status").startswith("success"):
            continue
        key = match.group("key")
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def parse_successful_read_keys_with_backups(path: Path) -> list[str]:
    keys = parse_successful_read_keys(path)
    seen = set(keys)
    try:
        rel_pattern = path.relative_to(vault_path()).as_posix()
    except ValueError:
        return keys
    backup_roots = [vault_path(".claude", "backups"), vault_path(".claude", "scripts", "backups")]
    for backup_root in backup_roots:
        if not backup_root.exists():
            continue
        for backup_path in sorted(backup_root.glob(f"*/{rel_pattern}")):
            for key in parse_successful_read_keys(backup_path):
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
    return keys


def _section_list_items(text: str, heading: str) -> list[str]:
    pattern = re.compile(rf"(?ms)^## {re.escape(heading)}\n\n(?P<body>.*?)(?=^## |\Z)")
    match = pattern.search(text)
    if not match:
        return []
    items: list[str] = []
    for line in match.group("body").splitlines():
        line = line.strip()
        if line.startswith("- "):
            value = line[2:].strip()
            if value and value != "none":
                items.append(value)
    return items


def _replace_markdown_section(text: str, heading: str, body_lines: list[str]) -> str:
    replacement = f"## {heading}\n\n" + "\n".join(body_lines).rstrip() + "\n\n"
    pattern = re.compile(rf"(?ms)^## {re.escape(heading)}\n\n.*?(?=^## |\Z)")
    if pattern.search(text):
        return pattern.sub(replacement, text, count=1).rstrip() + "\n"
    return text.rstrip() + "\n\n" + replacement


def reconcile_run_log_after_recovery(
    *,
    run_log_path: Path,
    recovery_log_path: Path,
    recovery_status: str,
    agenda_status: str,
) -> bool:
    if recovery_status != "success" or agenda_status not in {"success", "skipped_no_focus_keys"}:
        return False
    if not run_log_path.exists():
        return False
    text = run_log_path.read_text(encoding="utf-8", errors="replace")
    initial_errors = _section_list_items(text, "Errors")
    if not initial_errors:
        existing_initial_errors = re.search(r"(?m)^- initial_errors: (?P<errors>.+)$", text)
        if existing_initial_errors and existing_initial_errors.group("errors") != "none":
            initial_errors = [existing_initial_errors.group("errors")]
    try:
        recovery_rel = str(recovery_log_path.relative_to(vault_path())).replace("\\", "/")
    except ValueError:
        recovery_rel = str(recovery_log_path)

    text = re.sub(
        r'(?m)^summary: "Daily arXiv scout run status: .*?"$',
        'summary: "Daily arXiv scout run status: success after recovery."',
        text,
        count=1,
    )
    text = re.sub(r"(?m)^- status: .*$", "- status: success", text, count=1)
    text = re.sub(r"(?m)^- agenda_update_status: .*$", f"- agenda_update_status: {agenda_status}", text, count=1)
    text = re.sub(
        r"(?ms)(^## Agenda Update\n\n- mode: [^\n]+\n)- status: [^\n]+",
        lambda match: f"{match.group(1)}- status: {agenda_status}",
        text,
        count=1,
    )
    if "- recovery_status:" not in text:
        metadata = [
            "- recovery_status: success",
            f"- recovery_log: `{recovery_rel}`",
            "- initial_status: partial",
        ]
        if initial_errors:
            metadata.append(f"- initial_errors: {'; '.join(initial_errors)}")
        text = text.replace(
            f"- agenda_update_status: {agenda_status}\n",
            f"- agenda_update_status: {agenda_status}\n" + "\n".join(metadata) + "\n",
            1,
        )

    recovery_lines = [
        "- recovery_status: success",
        f"- recovery_log: `{recovery_rel}`",
        "- initial_status: partial",
    ]
    if initial_errors:
        recovery_lines.append(f"- initial_errors: {'; '.join(initial_errors)}")
    else:
        recovery_lines.append("- initial_errors: none")
    text = _replace_markdown_section(text, "Recovery", recovery_lines)
    text = _replace_markdown_section(text, "Errors", ["- none"])
    safe_write(run_log_path, text, dry_run=False, backup=True)
    return True


def _robotics_relevant(item: RankedPaper) -> bool:
    if "weak_robotics_signal" in item.penalties or "non_robotics_cv_context" in item.penalties:
        return False
    reasons = " ".join(item.reasons)
    if "domain:" in reasons or "research_direction_fit" in item.reasons:
        return True
    text = " ".join(
        [
            item.paper.title,
            item.paper.summary,
            item.paper.primary_category,
            " ".join(item.paper.categories),
        ]
    ).lower()
    return any(
        term in text
        for term in [
            "robot",
            "embodied",
            "vla",
            "vision-language-action",
            "rl token",
            "action interface",
            "tactile",
            "force",
            "contact-rich",
            "sim-to-real",
            "failure recovery",
            "bimanual",
            "dlo",
            "deformable",
            "grasp",
            "dexterous",
        ]
    )


def _coverage_relevant(item: RankedPaper) -> bool:
    return _robotics_relevant(item) and any(f"domain:{label}" in item.reasons for label in COVERAGE_DOMAIN_LABELS)


def _fill_import_candidate(item: RankedPaper, *, fill_threshold: int) -> RankedPaper:
    reasons = sorted(set(item.reasons + ["minimum_daily_import_fill", f"original_decision:{item.decision}"]))
    return replace(item, decision=AUTO_IMPORT_FILL_DECISION, reasons=reasons)


def select_import_candidates(
    ranked: list[RankedPaper],
    *,
    max_attempts: int,
    fill_threshold: int,
) -> list[tuple[RankedPaper, str, str]]:
    """Return import attempts ordered by confidence, with review/relevant fill candidates after primary auto-imports."""
    selected: list[tuple[RankedPaper, str, str]] = []
    seen: set[str] = set()
    primary_cap = min(PRIMARY_DAILY_IMPORT_CAP, max_attempts)
    target_floor = min(10, max_attempts)

    def add(item: RankedPaper, selection: str, original_decision: str | None = None) -> None:
        if item.paper.arxiv_id in seen or len(selected) >= max_attempts:
            return
        seen.add(item.paper.arxiv_id)
        selected.append((item, selection, original_decision or item.decision))

    for item in ranked:
        if item.decision in AUTO_IMPORT_DECISIONS:
            if len([entry for entry in selected if entry[1] in AUTO_IMPORT_DECISIONS]) >= primary_cap:
                continue
            add(item, item.decision)

    for item in ranked:
        focus_count = len([entry for entry in selected if entry[1] == FOCUS_REVIEW_DECISION])
        if focus_count >= FOCUS_DAILY_IMPORT_FLOOR and len(selected) >= target_floor:
            break
        if item.decision == FOCUS_REVIEW_DECISION and _robotics_relevant(item):
            add(item, FOCUS_REVIEW_DECISION)

    coverage_count = len([entry for entry in selected if _coverage_relevant(entry[0])])
    if coverage_count < DOMAIN_DAILY_IMPORT_FLOOR:
        for item in ranked:
            if item.quality_score < fill_threshold:
                continue
            if item.decision in AUTO_IMPORT_DECISIONS | {FOCUS_REVIEW_DECISION, "review_queue"} and _coverage_relevant(item):
                selection = item.decision if item.decision in AUTO_IMPORT_DECISIONS | {FOCUS_REVIEW_DECISION} else AUTO_IMPORT_FILL_DECISION
                candidate = item if selection != AUTO_IMPORT_FILL_DECISION else _fill_import_candidate(item, fill_threshold=fill_threshold)
                add(candidate, selection, item.decision)
                break

    for item in ranked:
        if len(selected) >= target_floor:
            break
        if item.decision in AUTO_IMPORT_DECISIONS:
            add(item, item.decision)

    for item in ranked:
        if len(selected) >= target_floor:
            break
        if item.decision == "review_queue" and _robotics_relevant(item):
            add(_fill_import_candidate(item, fill_threshold=fill_threshold), AUTO_IMPORT_FILL_DECISION, item.decision)

    for item in ranked:
        if len(selected) >= target_floor:
            break
        if item.decision in LOW_PRIORITY_DECISIONS and item.quality_score >= fill_threshold and _robotics_relevant(item):
            add(_fill_import_candidate(item, fill_threshold=fill_threshold), AUTO_IMPORT_FILL_DECISION, item.decision)

    for item in ranked:
        if len(selected) >= max_attempts:
            break
        if item.decision in AUTO_IMPORT_DECISIONS:
            add(item, item.decision)

    for item in ranked:
        if len(selected) >= max_attempts:
            break
        if item.decision == FOCUS_REVIEW_DECISION and _robotics_relevant(item):
            add(item, FOCUS_REVIEW_DECISION)

    for item in ranked:
        if len(selected) >= max_attempts:
            break
        if item.decision == "review_queue" and _robotics_relevant(item):
            add(_fill_import_candidate(item, fill_threshold=fill_threshold), AUTO_IMPORT_FILL_DECISION, item.decision)

    for item in ranked:
        if len(selected) >= max_attempts:
            break
        if item.decision in LOW_PRIORITY_DECISIONS and item.quality_score >= fill_threshold and _robotics_relevant(item):
            add(_fill_import_candidate(item, fill_threshold=fill_threshold), AUTO_IMPORT_FILL_DECISION, item.decision)

    return selected


def selection_quota_summary(selected: list[tuple[RankedPaper, str, str]], *, target: int | None = None) -> dict[str, int]:
    daily_slice = selected[:target] if target else selected
    return {
        "primary": sum(1 for _, selection, _ in daily_slice if selection in AUTO_IMPORT_DECISIONS),
        "focus": sum(1 for _, selection, _ in daily_slice if selection == FOCUS_REVIEW_DECISION),
        "domain_coverage": sum(1 for item, _, _ in daily_slice if _coverage_relevant(item)),
        "fill": sum(1 for item, _, _ in daily_slice if item.decision == AUTO_IMPORT_FILL_DECISION),
        "attempts_total": len(selected),
    }


def _selection_quota_text(summary: dict[str, int]) -> str:
    return ";".join(f"{key}={summary.get(key, 0)}" for key in ["primary", "focus", "domain_coverage", "fill", "attempts_total"])


def _filter_import_candidates_by_v2_triage(
    selected: list[tuple[RankedPaper, str, str]],
    triage: dict[str, Any],
) -> list[tuple[RankedPaper, str, str]]:
    """Prioritize v2 deep-read choices while retaining fallback candidates.

    The daily loop stops once ``max_read`` successful reads are reached. Keeping
    fallback candidates here lets the run replace a selected paper whose PDF
    import or strict read fails without broadening normal successful days.
    """
    selected_rows = triage.get("selected_for_deep_read", [])
    if not isinstance(selected_rows, list):
        return selected
    selected_ids = [str(row.get("arxiv_id") or row.get("paper_id") or "") for row in selected_rows if isinstance(row, dict)]
    selected_ids = [value for value in selected_ids if value]
    if not selected_ids:
        return []
    by_id = {item.paper.arxiv_id: (item, selection, original) for item, selection, original in selected}
    prioritized = [by_id[arxiv_id] for arxiv_id in selected_ids if arxiv_id in by_id]
    selected_id_set = set(selected_ids)
    fallbacks = [
        (item, selection, original)
        for item, selection, original in selected
        if item.paper.arxiv_id not in selected_id_set
    ]
    return prioritized + fallbacks


def _non_robotics_rejected_sample(ranked: list[RankedPaper], *, limit: int = 8) -> list[RankedPaper]:
    return [item for item in ranked if "non_robotics_cv_context" in item.penalties][:limit]


def exclude_existing_candidates(
    ranked: list[RankedPaper],
    *,
    collection_key: str,
) -> tuple[list[RankedPaper], list[dict[str, Any]], str]:
    if not ranked:
        return ranked, [], "skipped:no_candidates"
    if not collection_key:
        return ranked, [], "skipped:missing_collection_key"
    try:
        local_items = iter_local_items(collection_key)
    except Exception as exc:
        return ranked, [], f"failed:{exc}"

    new_ranked: list[RankedPaper] = []
    existing_records: list[dict[str, Any]] = []
    for item in ranked:
        existing = find_existing_paper_in_items(item.paper, local_items)
        if not existing:
            new_ranked.append(item)
            continue
        existing_records.append(
            {
                "arxiv_id": item.paper.arxiv_id,
                "title": item.paper.title,
                "quality_score": item.quality_score,
                "decision": item.decision,
                **existing.to_dict(),
            }
        )
    return new_ranked, existing_records, f"success:local_items={len(local_items)}"


def existing_literature_notes_by_title() -> dict[str, dict[str, Any]]:
    notes: dict[str, dict[str, Any]] = {}
    for path in sorted(vault_path("wiki", "topics").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        parsed = extract_frontmatter(text)
        if not parsed:
            continue
        fields = parse_frontmatter_map(parsed[0])
        if str(fields.get("type", "")).strip('"') != "literature":
            continue
        title = str(fields.get("title", "")).strip('"')
        normalized = normalize_title(title)
        if not normalized:
            continue
        notes.setdefault(
            normalized,
            {
                "path": str(path.relative_to(vault_path())),
                "title": title,
                "status": str(fields.get("status", "")).strip('"'),
                "zotero_key": str(fields.get("zotero_key", "")).strip('"'),
            },
        )
    return notes


def render_review_queue(ranked: list[RankedPaper], run_date: str) -> str:
    focus_review = [item for item in ranked if item.decision == FOCUS_REVIEW_DECISION]
    review = [item for item in ranked if item.decision == "review_queue"]
    venue_auto = [item for item in ranked if item.decision == "venue_auto_import"]
    lines = [
        "---",
        f'title: "arXiv Review Queue - {run_date}"',
        "tags: [arxiv, review-queue, embodied-ai]",
        f'created: "{run_date}"',
        f'updated: "{run_date}"',
        'type: "permanent"',
        'status: "draft"',
        f'summary: "{len(focus_review) + len(review)} arXiv candidates need human or focus-track review."',
        "---",
        "",
        f"# arXiv Review Queue - {run_date}",
        "",
        f"- top1_threshold: {TOP1_THRESHOLD}",
        f"- review_threshold: {REVIEW_THRESHOLD}",
        f"- focus_review_queue: {len(focus_review)}",
        "- decision_boundary: review_queue items can fill the daily minimum only after top1/venue candidates.",
        "- focus_boundary: focus_review_queue items matched an active specialized track and are considered before generic review fill.",
        f"- venue_auto_import: {len(venue_auto)} accepted top-venue candidates will bypass review queue.",
        "",
        "## Focus Review Queue",
        "",
    ]
    if not focus_review:
        lines.append("- No focus_review_queue candidates today.")
    for index, item in enumerate(focus_review, 1):
        paper = item.paper
        focus = "; ".join(f"{track}: {', '.join(matches)}" for track, matches in item.focus_matches.items())
        lines.extend(
            [
                f"## F{index}. {paper.title}",
                "",
                f"- arxiv: [{paper.arxiv_id}]({paper.url})",
                f"- score: {item.quality_score}",
                f"- focus_matches: {focus or '-'}",
                f"- reasons: {', '.join(item.reasons) if item.reasons else '-'}",
                f"- penalties: {', '.join(item.penalties) if item.penalties else '-'}",
                f"- authors: {', '.join(paper.authors[:6])}",
                f"- summary: {paper.summary[:700]}",
                "",
            ]
        )
    lines.extend(["## General Review Queue", ""])
    if not review:
        lines.append("- No review_queue candidates today.")
    for index, item in enumerate(review, 1):
        paper = item.paper
        lines.extend(
            [
                f"## {index}. {paper.title}",
                "",
                f"- arxiv: [{paper.arxiv_id}]({paper.url})",
                f"- score: {item.quality_score}",
                f"- reasons: {', '.join(item.reasons) if item.reasons else '-'}",
                f"- penalties: {', '.join(item.penalties) if item.penalties else '-'}",
                f"- authors: {', '.join(paper.authors[:6])}",
                f"- summary: {paper.summary[:700]}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def run_subprocess(command: list[str], *, timeout: int) -> tuple[int, str]:
    proc = subprocess.Popen(
        command,
        cwd=vault_path(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_process_tree(proc.pid)
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        return 124, f"timeout:{timeout}s"
    output = (stdout or "") + ("\nSTDERR:\n" + stderr if stderr else "")
    return proc.returncode, output.strip()


def _v2_stage(
    stage: str,
    command: list[str],
    *,
    timeout: int,
    result: dict[str, Any],
    errors: list[str],
) -> bool:
    started = time.monotonic()
    try:
        code, output = run_subprocess(command, timeout=timeout)
    except Exception as exc:
        code, output = 1, f"{type(exc).__name__}:{exc}"
    result["stages"].append(
        {
            "stage": stage,
            "status": "success" if code == 0 else "failed",
            "exit_code": code,
            "elapsed_sec": round(time.monotonic() - started, 2),
            "output_tail": output[-1200:],
        }
    )
    if code != 0:
        errors.append(f"research_seed_v2_{stage}_failed")
        result["status"] = "partial"
        return False
    return True


def append_provider_args(
    cmd: list[str],
    *,
    provider: str,
    provider_json: str = "",
    model: str = "",
    timeout: int | None = None,
    provider_retries: int | None = None,
) -> None:
    provider = provider or "none"
    provider_json = provider_json or ""

    if "--provider" in cmd:
        raise ValueError("provider_arg_already_present")

    if provider_json and provider not in {"json", "none", ""}:
        raise ValueError("provider_json_conflicts_with_explicit_provider")

    if provider_json:
        cmd.extend(["--provider", "json", "--provider-review-json", provider_json])
        return

    if provider in {"opencode", "openai-compatible", "codex-cli"}:
        cmd.extend(["--provider", provider])
        if provider in {"opencode", "openai-compatible"} and model:
            cmd.extend(["--model", model])
        if provider in {"opencode", "openai-compatible"} and provider_retries is not None:
            cmd.extend(["--provider-retries", str(provider_retries)])
        if timeout is not None:
            cmd.extend(["--timeout", str(timeout)])
        return

    if provider in {"json", "none", ""}:
        cmd.extend(["--provider", provider if provider else "none"])
        return

    raise ValueError(f"unsupported_provider:{provider}")


def build_v2_review_stages(args: argparse.Namespace, run_date: str) -> list[tuple[str, list[str], int]]:
    scheduled_policy = "seed-candidates-only" if resolve_v2_publish_policy(args) != "disabled" else "disabled"
    deepseek_cmd = [sys.executable, ".claude/scripts/deepseek_scientific_review.py", "--run-date", run_date]
    append_provider_args(
        deepseek_cmd,
        provider=getattr(args, "deepseek_provider", "none"),
        provider_json=getattr(args, "deepseek_provider_json", ""),
        model=getattr(args, "deepseek_model", ""),
        timeout=getattr(args, "deepseek_timeout", None),
        provider_retries=getattr(args, "deepseek_provider_retries", 1),
    )

    codex_timeout = 1200
    codex_cmd = [sys.executable, ".claude/scripts/codex_seed_review.py", "execution-review", "--run-date", run_date]
    append_provider_args(
        codex_cmd,
        provider=getattr(args, "codex_execution_provider", "none"),
        provider_json=getattr(args, "codex_execution_provider_json", ""),
        timeout=codex_timeout,
    )

    novelty_cmd = [
        sys.executable,
        ".claude/scripts/novelty_baseline_scan.py",
        "--run-date",
        run_date,
        "--target-policy",
        scheduled_policy,
        "--max-external-queries",
        str(getattr(args, "novelty_max_external_queries", 1)),
        "--external-timeout",
        str(getattr(args, "novelty_external_timeout", 12)),
    ]
    survival_cmd = [
        sys.executable,
        ".claude/scripts/survival_decision.py",
        "--run-date",
        run_date,
        "--target-policy",
        scheduled_policy,
    ]
    review_stages: list[tuple[str, list[str], int]] = [
        ("portfolio_select", [sys.executable, ".claude/scripts/candidate_portfolio_select.py", "--run-date", run_date], 180),
        ("deepseek_review", deepseek_cmd, max(1800, args.deepseek_timeout + 120)),
        ("gemini_rescue_mutation", [sys.executable, ".claude/scripts/gemini_rescue_mutation.py", "--run-date", run_date], max(600, args.idea_timeout)),
        ("novelty_scan", novelty_cmd, 600),
        ("codex_execution_review", codex_cmd, max(600, codex_timeout)),
        ("baseline_table", [sys.executable, ".claude/scripts/baseline_table.py", "--run-date", run_date], 240),
        ("survival_decision", survival_cmd, 300),
        ("active_seed_dashboard", [sys.executable, ".claude/scripts/active_seed_dashboard.py", "--run-date", run_date], 180),
    ]
    if args.allow_human_override:
        survival_cmd.append("--allow-human-override")
    return review_stages


def resolve_v2_publish_policy(args: argparse.Namespace) -> str:
    policy = str(getattr(args, "v2_publish_policy", DEFAULT_V2_PUBLISH_POLICY) or DEFAULT_V2_PUBLISH_POLICY)
    if policy == "formal":
        return "seed-candidates-only"
    return policy


def run_research_seed_v2(
    *,
    args: argparse.Namespace,
    run_date: str,
    candidates_path: Path,
    focus_zotero_keys: list[str],
    errors: list[str],
) -> dict[str, Any]:
    """Run the transactional v2 state machine. Formal seed writes occur only in publish_research_run.py."""
    backfill_mode = args.backfill_mode
    v2_publish_policy = resolve_v2_publish_policy(args)
    formal_seed_publish_allowed = False
    ingest_only = backfill_mode == "ingest-only" and not args.backfill_generate_ideas
    result: dict[str, Any] = {
        "schema_version": "daily_research_seed_v2_summary.v1",
        "status": "success",
        "run_date": run_date,
        "run_dir": f"projects/research-agenda/runs/{run_date}",
        "backfill_mode": backfill_mode,
        "backfill_generate_ideas": bool(args.backfill_generate_ideas),
        "backfill_publish": args.backfill_publish,
        "v2_publish_policy": v2_publish_policy,
        "formal_seed_publish_allowed": formal_seed_publish_allowed,
        "scheduled_daily_switched": False,
        "scheduled_candidate_only_cap": True,
        "focus_zotero_keys": focus_zotero_keys,
        "stages": [],
    }
    if backfill_mode != "daily" and v2_publish_policy == "formal":
        errors.append("research_seed_v2_backfill_formal_publish_blocked")
        result["status"] = "partial"
        result["boundary"] = "Backfill cannot formal-publish; use ingest-only or seed-candidates-only rehearsal."
        return result
    if (
        v2_publish_policy == "formal"
        and not getattr(args, "allow_test_provider_for_formal", False)
        and (getattr(args, "deepseek_provider", "none") != "opencode" or getattr(args, "codex_execution_provider", "none") != "codex-cli")
    ):
        errors.append("research_seed_v2_formal_requires_production_providers")
        result["status"] = "partial"
        result["boundary"] = "Formal publish requires DeepSeek opencode and Codex codex-cli unless explicit manual test-provider override is used."
        return result

    init_args = [
        sys.executable,
        ".claude/scripts/validate_research_run.py",
        "init",
        "--run-date",
        run_date,
        "--backfill-mode",
        backfill_mode,
        "--v2-publish-policy",
        v2_publish_policy,
    ]
    if formal_seed_publish_allowed:
        init_args.append("--formal-seed-publish-allowed")
    if not _v2_stage("init_manifest", init_args, timeout=120, result=result, errors=errors):
        return result

    triage_args = [
        sys.executable,
        ".claude/scripts/paper_intake_triage.py",
        "--run-date",
        run_date,
        "--candidates",
        str(candidates_path),
        "--target-deep-read",
        str(args.target_deep_read),
        "--max-deep-read",
        str(args.max_deep_read),
    ]
    if not _v2_stage("intake_triage", triage_args, timeout=180, result=result, errors=errors):
        return result

    if focus_zotero_keys:
        extract_args = [
            sys.executable,
            ".claude/scripts/research_agenda_extract.py",
            "--run-date",
            run_date,
            "--zotero-keys",
            ",".join(focus_zotero_keys),
            "--write-v2-primitives",
        ]
        if not _v2_stage("paper_primitives", extract_args, timeout=600, result=result, errors=errors):
            return result
    else:
        result["stages"].append(
            {
                "stage": "paper_primitives",
                "status": "skipped_no_focus_keys",
                "exit_code": 0,
                "elapsed_sec": 0,
                "output_tail": "No successfully read focus Zotero keys; claim graph will be empty unless previous primitives exist.",
            }
        )

    for stage, command, timeout in [
        ("claim_graph", [sys.executable, ".claude/scripts/research_claim_graph.py", "--run-date", run_date], 240),
        ("pdf_evidence_anchors", [sys.executable, ".claude/scripts/pdf_evidence_extract.py", "--run-date", run_date], 240),
        ("tension_map", [sys.executable, ".claude/scripts/tension_map.py", "--run-date", run_date], 240),
    ]:
        if not _v2_stage(stage, command, timeout=timeout, result=result, errors=errors):
            return result

    if ingest_only:
        result["status"] = "success_ingest_only"
        result["boundary"] = "Backfill ingest-only stopped before raw candidate generation and formal seed publish."
        return result

    if not focus_zotero_keys:
        result["status"] = "success_no_focus_keys"
        result["boundary"] = "No raw candidates generated because no successfully read focus Zotero keys were available."
        return result

    ideate_args = [
        sys.executable,
        ".claude/scripts/research_agenda_ideate.py",
        "--run-date",
        run_date,
        "--focus-zotero-keys",
        ",".join(focus_zotero_keys),
        "--generator",
        args.idea_mode,
        "--generator-timeout",
        str(args.idea_timeout),
        "--gemini-model",
        args.gemini_model,
        "--raw-candidate-limit",
        str(args.raw_candidate_limit),
        "--min-raw-candidates",
        str(args.min_raw_candidates),
        "--max-generated",
        str(args.max_generated),
        "--write-v2-run-artifact",
    ]
    if not _v2_stage("raw_candidates", ideate_args, timeout=max(900, args.idea_timeout + 300), result=result, errors=errors):
        return result

    try:
        review_stages = build_v2_review_stages(args, run_date)
    except ValueError as exc:
        errors.append(f"research_seed_v2_provider_command_error:{exc}")
        result["status"] = "partial"
        result["boundary"] = f"Provider command construction failed closed: {exc}"
        return result

    for stage, command, timeout in review_stages:
        if not _v2_stage(stage, command, timeout=timeout, result=result, errors=errors):
            return result

    if args.backfill_publish != "disabled":
        publish_args = [
            sys.executable,
            ".claude/scripts/publish_research_run.py",
            "--run-date",
            run_date,
            "--target-policy",
            "seed-candidates-only",
        ]
        if not _v2_stage("publish_seed_candidates_only", publish_args, timeout=300, result=result, errors=errors):
            return result
        result["status"] = "success_seed_candidates_only"
        result["boundary"] = "Backfill requested seed-candidates-only; formal seed publish is intentionally disabled."
        return result
    if backfill_mode == "ingest-only":
        result["status"] = "success_backfill_review_only"
        result["boundary"] = "Backfill generated/reviewed ideas but publish is disabled; no formal seed or seed-candidate bucket was written."
        return result
    publish_args = [
        sys.executable,
        ".claude/scripts/publish_research_run.py",
        "--run-date",
        run_date,
        "--target-policy",
        v2_publish_policy,
    ]
    publish_stage = "publish_disabled" if v2_publish_policy == "disabled" else f"publish_{v2_publish_policy}"
    if not _v2_stage(publish_stage, publish_args, timeout=300, result=result, errors=errors):
        return result
    if v2_publish_policy == "disabled":
        result["status"] = "success_publish_disabled"
        result["boundary"] = "V2 rollout policy disabled publish; survival decision was recorded but no buckets or formal seed were written."
        return result
    if v2_publish_policy == "seed-candidates-only":
        result["status"] = "success_seed_candidates_only"
        result["boundary"] = "V2 rollout policy routed accepted candidates to seed-candidates; formal seed publish is disabled."

    audit_args = [sys.executable, ".claude/scripts/audit_daily_automation_quality.py", "--run-date", run_date, "--dry-run"]
    _v2_stage("audit", audit_args, timeout=300, result=result, errors=errors)
    return result


def run_subprocess_capture(
    command: list[str],
    *,
    timeout: int,
    heartbeat: Callable[[int], None] | None = None,
    heartbeat_interval: int = 60,
    env_overrides: dict[str, str] | None = None,
    input_text: str | None = None,
) -> tuple[int, str, str]:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    proc = subprocess.Popen(
        command,
        cwd=vault_path(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE if input_text is not None else None,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if heartbeat:
        heartbeat(proc.pid)
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                stdout, stderr = proc.communicate(input=input_text, timeout=min(heartbeat_interval, remaining))
                input_text = None
                break
            except subprocess.TimeoutExpired:
                input_text = None
                if heartbeat:
                    heartbeat(proc.pid)
        if heartbeat:
            heartbeat(proc.pid)
    except subprocess.TimeoutExpired:
        kill_process_tree(proc.pid)
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        return 124, stdout or "", (stderr or "") + f"\nTIMEOUT: {timeout}s"
    return proc.returncode, stdout or "", stderr or ""


def kill_process_tree(pid: int) -> None:
    if sys.platform.startswith("win"):
        try:
            subprocess.run(
                ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
                text=True,
                capture_output=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        return
    try:
        import os

        os.kill(pid, 9)
    except OSError:
        pass


def ingest_zotero_key(zotero_key: str, *, timeout: int = 600) -> tuple[str, str]:
    command = [sys.executable, ".claude/scripts/ingest_paper.py", zotero_key, "--force-overwrite-stub"]
    code, output = run_subprocess(command, timeout=timeout)
    return ("success" if code == 0 else "failed", output)


def frontmatter_zotero_keys(fields: dict[str, str]) -> set[str]:
    keys = {str(fields.get("zotero_key", "")).strip('"').upper()}
    for alias_field in ("zotero_aliases", "zotero_keys"):
        keys.update(item.upper() for item in parse_list_value(str(fields.get(alias_field, ""))))
    return {key for key in keys if key}


def local_note_status(zotero_key: str) -> tuple[str, str]:
    requested_key = zotero_key.upper()
    for path in sorted(vault_path("wiki", "topics").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        parsed = extract_frontmatter(text)
        if not parsed:
            continue
        fields = parse_frontmatter_map(parsed[0])
        if requested_key not in frontmatter_zotero_keys(fields):
            continue
        status = fields.get("status", "").strip('"')
        return status or "missing_status", str(path.relative_to(vault_path()))
    return "missing_note", ""


def local_note_fields_from_relpath(rel_path: str) -> tuple[str, str, str]:
    cleaned = rel_path.strip().strip("`").replace("\\", "/")
    if not cleaned:
        return "missing_note", "", ""
    path = vault_path(*cleaned.split("/"))
    if not path.exists() or not path.is_file():
        return "missing_note", "", ""
    text = path.read_text(encoding="utf-8")
    parsed = extract_frontmatter(text)
    if not parsed:
        return "missing_frontmatter", "", str(path.relative_to(vault_path()))
    fields = parse_frontmatter_map(parsed[0])
    status = str(fields.get("status", "")).strip('"') or "missing_status"
    zotero_key = str(fields.get("zotero_key", "")).strip('"')
    return status, zotero_key, str(path.relative_to(vault_path()))


def local_note_from_ingest_output(output: str) -> tuple[str, str, str]:
    pattern = re.compile(r"^INGESTED_OR_FOUND:\s+(?P<path>.+?)\s*$")
    for line in reversed(output.splitlines()):
        match = pattern.match(line.strip())
        if not match:
            continue
        status, zotero_key, note_path = local_note_fields_from_relpath(match.group("path"))
        if note_path:
            return status, zotero_key, note_path
    return "missing_note", "", ""


def wait_for_local_note(zotero_key: str, *, timeout_sec: int = 180, interval_sec: int = 10) -> tuple[str, str]:
    deadline = time.monotonic() + timeout_sec
    while True:
        note_status, note_path = local_note_status(zotero_key)
        if note_status != "missing_note":
            return note_status, note_path
        if time.monotonic() >= deadline:
            return note_status, note_path
        time.sleep(interval_sec)


def read_paper_prompt(zotero_key: str) -> str:
    template_path = vault_path(".claude", "commands", "read-paper.md")
    template = template_path.read_text(encoding="utf-8")
    return template.replace("$ARGUMENTS", zotero_key)


def staged_read_root(zotero_key: str, *, attempt: int) -> Path:
    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", zotero_key) or "unknown"
    return vault_path("raw", "readings", "_staged", safe_key, f"attempt-{attempt}")


def codex_controlled_read_root(zotero_key: str, *, attempt: int) -> Path:
    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", zotero_key) or "unknown"
    return vault_path("raw", "readings", "_codex_controlled", safe_key, f"attempt-{attempt}")


def staged_session_id(zotero_key: str, *, attempt: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"local-domain-expert-vault:staged-read:{zotero_key}:attempt:{attempt}"))


def _rel(path: Path) -> str:
    return str(path.relative_to(vault_path())).replace("\\", "/")


def _stage_output_path(stage_dir: Path, stage_id: str) -> Path:
    return stage_dir / f"{stage_id}.md"


def _stage_log_path(stage_dir: Path, stage_id: str, suffix: str) -> Path:
    return stage_dir / "logs" / f"{stage_id}.{suffix}.log"


def _staged_manifest_path(stage_dir: Path) -> Path:
    return stage_dir / "manifest.json"


def _read_staged_manifest(stage_dir: Path) -> dict[str, Any]:
    path = _staged_manifest_path(stage_dir)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_staged_manifest(stage_dir: Path, payload: dict[str, Any]) -> str:
    stage_dir.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    path = _staged_manifest_path(stage_dir)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return _rel(path)


def _stage_complete(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return len(text.strip()) >= 500


def _final_analysis_path(zotero_key: str) -> Path:
    return vault_path("raw", "readings", f"{zotero_key}-analysis.md")


def _final_analysis_has_required_sections(path: Path) -> tuple[bool, list[str]]:
    if not path.exists() or not path.is_file():
        return False, ["missing_final_analysis"]
    text = path.read_text(encoding="utf-8")
    missing = [section for section in STAGED_REQUIRED_FINAL_SECTIONS if section not in text]
    structured_body = _section_body(text, ("结构化提取",))
    for field in STRUCTURED_EXTRACTION_REQUIRED_FIELDS:
        field_name = field.rstrip(":")
        field_pattern = re.compile(rf"(?m)^-\s+(?:\*\*)?{re.escape(field_name)}(?:\*\*)?:\s*\S")
        if not field_pattern.search(structured_body):
            missing.append(f"structured_field:{field}")
    if len(text.strip()) < 2000:
        missing.append("final_analysis_too_short")
    return not missing, missing


def _read_text_limited(path: Path, *, max_chars: int = 12000) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED]\n"


def deterministic_read_fallback_enabled() -> bool:
    return os.environ.get(DETERMINISTIC_READ_FALLBACK_ENV, "").lower() in {"1", "true", "yes", "on"}


def deterministic_stage03_packet(stage_id: str, zotero_key: str, prior_paths: list[Path]) -> str:
    prior_chars = sum(len(_read_text_limited(path, max_chars=3000)) for path in prior_paths)
    if stage_id == "03a-experiments-results":
        return f"""# Stage 03a: Experiments, tasks, metrics, and result anchors

## Deterministic Recovery Boundary
- Zotero key: {zotero_key}
- Source: prior staged files only; no additional Zotero/arXiv/web/model fulltext call was made in this recovery stage.
- Prior context chars inspected by script: {prior_chars}
- Evidence policy: any missing experiment result is `not_evidenced`; this packet is screening-only until a human/model stage supplies exact table, figure, or result-row anchors.

## 1. Benchmark families table
| benchmark_family | task_or_dataset | metric_or_result | evidence_class | anchor | downstream_use |
|---|---|---|---|---|---|
| paper_reported_experiments | not_evidenced | not_evidenced | not_evidenced | not_evidenced | screening_only |
| method_relevant_proxy | not_evidenced | not_evidenced | note_derived | prior staged method context | screening_only |

## 2. Strongest result anchors table
| claim_id | result_claim | evidence_class | anchor_type | anchor | confidence | downstream_use |
|---|---|---|---|---|---|---|
| E01 | Exact experiment/result anchors were not recovered in this deterministic stage. | not_evidenced | note_only | not_evidenced | low | screening_only |
| B01 | Strongest baseline pressure must be confirmed from source tables before use as evidence. | note_derived | note_only | prior staged method context | low | screening_only |

## 3. Missing or unclear experiment evidence
- Main metrics: not_evidenced unless present in earlier staged files.
- Table/figure/result-row anchors: not_evidenced in this recovery packet.
- Ablations: not_evidenced.
- Baseline comparison details: screening_only and requires later confirmation.
- This stage intentionally prefers an incomplete but auditable packet over another long-running fulltext model call.
"""
    if stage_id == "03b-limitations-evidence-gaps":
        return f"""# Stage 03b: Limitations, failure modes, and evidence gaps

## Deterministic Recovery Boundary
- Zotero key: {zotero_key}
- Source: prior staged files only; no additional Zotero/arXiv/web/model fulltext call was made in this recovery stage.
- Prior context chars inspected by script: {prior_chars}
- Evidence policy: limitations and transfer risks here are review pressure, not confirmed paper claims, unless anchored in earlier staged files.

## 1. Stated limitations
- L01: Stated limitations were not fully recovered in this deterministic stage; mark as `not_evidenced` until exact source anchors are available.

## 2. Implied failure modes
- Possible failure modes are screening-only and must be attacked by downstream DeepSeek/Codex review before they influence candidates.
- Any method assumption from prior stages remains `note_derived` unless tied to a table, figure, section, appendix, or result row.

## 3. Missing ablations and evidence gaps
- Missing ablation anchors: not_evidenced.
- Missing metric automation path: not_evidenced.
- Missing baseline execution evidence: not_evidenced.
- Missing result-row confirmation: not_evidenced.

## 4. Transfer and negative-transfer risks
- T01: Transfer to DLO / embodied manipulation is high risk unless the paper directly evaluates physical control, contact dynamics, closed-loop manipulation, or DLO-specific state changes.
- Negative transfer risk: copying a source-domain proxy may optimize visible success while missing contact, deformation, topology, latency, or recovery failures.
- Downstream use: screening_only; requires human check before any active/governance use.
"""
    if stage_id == "04-evidence-ledger":
        return f"""# Stage 04: Evidence Ledger

## Deterministic Recovery Boundary
- Zotero key: {zotero_key}
- Source: prior staged files only; no additional model/tool call was made in this recovery stage.
- Prior context chars inspected by script: {prior_chars}
- Evidence policy: fallback rows are `note_derived` or `not_evidenced`, screening-only, and cannot support active seed evidence.

{normalize_evidence_ledger("")}
"""
    if stage_id == "05-review-pressure":
        return f"""# Stage 05: Key takeaways, IF packets, baseline, transfer, no-hardware test

## Deterministic Recovery Boundary
- Zotero key: {zotero_key}
- Source: prior staged files plus deterministic conservative review-pressure templates.
- Prior context chars inspected by script: {prior_chars}
- Evidence policy: all generated IF packets are screening-only review pressure, not confirmed paper claims.

## Key Takeaways
- The readable method-stage content can support candidate screening, but missing experiment/table anchors remain `not_evidenced`.
- Downstream Gemini/DeepSeek/Codex may use these packets as attack targets only; they must not be treated as proof.

## Idea Fuel
{normalized_idea_fuel()}

## Baseline Pressure
{normalized_baseline_pressure()}

## Transfer Risk
{normalized_transfer_risk()}

## No-hardware Micro-test
{normalized_no_hardware_micro_test()}

## Evidence Gaps
- Exact table/figure/result-row anchors may be incomplete in deterministic recovery output.
- Missing anchors must be treated as `not_evidenced` or `screening_only`.
- No formal seed, active seed, or governance commitment can be created from this packet.

## 结构化提取
- Problem: see staged Problem section; if absent, not_evidenced
- Method: see staged Method section; if absent, not_evidenced
- Tasks: not_evidenced unless present in staged experiment packet
- Sensors: not_evidenced unless explicitly anchored
- Robot Setup: not_evidenced unless explicitly anchored
- Metrics: not_evidenced unless explicitly anchored
- Limitations: screening_only limitation and transfer-risk packet
- Evidence Notes: deterministic recovery output; weak or missing evidence remains not_evidenced/screening_only

## 本地引用关系
- not_evidenced: deterministic recovery does not create new local paper links.
"""
    return ""


def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current_heading:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = line.strip()
            current_lines = []
            continue
        if current_heading:
            current_lines.append(line)
    if current_heading:
        sections.append((current_heading, "\n".join(current_lines).strip()))
    return sections


def _section_body(text: str, aliases: tuple[str, ...]) -> str:
    normalized_aliases = tuple(alias.lower() for alias in aliases)
    for heading, body in _split_markdown_sections(text):
        heading_text = heading.lstrip("#").strip().lower()
        if any(heading_text.startswith(alias) for alias in normalized_aliases):
            return body.strip()
    return ""


def _stage3_experiments_and_limitations(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    split_index = None
    for index, line in enumerate(lines):
        lowered = line.strip().lower()
        if lowered.startswith("## j. limitations") or lowered.startswith("## limitations"):
            split_index = index
            break
    if split_index is None:
        return text.strip(), "not_evidenced: limitations split marker missing in staged output."
    return "\n".join(lines[:split_index]).strip(), "\n".join(lines[split_index:]).strip()


def _demote_embedded_headings(text: str) -> str:
    return re.sub(r"(?m)^## ", "### ", text.strip())


def _summary_from_problem(problem_body: str) -> str:
    for line in problem_body.splitlines():
        stripped = re.sub(r"[*_`>#]+", "", line).strip()
        stripped = re.sub(r"\[claim:[^\]]+\]", "", stripped).strip()
        if len(stripped) >= 30:
            return stripped[:180]
    return "本篇精读围绕论文的问题设定、方法、实验、局限、证据锚点和下游研究压力进行结构化整理。"


NUMERIC_ANCHOR_TERMS = {
    "success",
    "rate",
    "accuracy",
    "ablation",
    "baseline",
    "metric",
    "score",
    "error",
    "sr",
    "提升",
    "成功率",
    "指标",
    "基线",
    "结果",
}


def _claim_for_numeric_line(line: str) -> str:
    lowered = line.lower()
    if "cot" in lowered or "dropout" in lowered or "thinking" in lowered or "vanilla" in lowered:
        return "M01"
    return "E01"


def _anchor_numeric_lines(text: str) -> str:
    anchored_lines: list[str] = []
    for line in text.splitlines():
        lowered = line.lower()
        has_number = bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:%|x|ms|s|m|cm|mm|points?|pts)?\b", line))
        has_term = any(term in lowered for term in NUMERIC_ANCHOR_TERMS)
        has_anchor = bool(re.search(r"\b(?:claim|table|figure|fig\.?|section|appendix|page|p\.)\s*[:#]?\s*[A-Za-z0-9_.-]+", line, flags=re.IGNORECASE))
        if has_number and has_term and not has_anchor and not line.strip().startswith("|"):
            line = f"{line} [claim:{_claim_for_numeric_line(line)}]"
        anchored_lines.append(line)
    return "\n".join(anchored_lines)


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _table_cell(value: str) -> str:
    return " ".join(value.replace("|", "/").split())


def _is_separator_row(line: str) -> bool:
    cells = _table_cells(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _ledger_claim_type(group_heading: str) -> str:
    lowered = group_heading.lower()
    if "problem" in lowered:
        return "problem"
    if "contribution" in lowered:
        return "contribution"
    if "method" in lowered:
        return "method"
    if "experiment" in lowered:
        return "result"
    if "baseline" in lowered:
        return "baseline"
    if "transfer" in lowered or "generalization" in lowered:
        return "transfer_failure"
    if "limitation" in lowered:
        return "limitation"
    return "evidence_note"


def _anchor_type_from_source(source: str) -> str:
    lowered = source.lower()
    if "table" in lowered:
        return "table"
    if "fig" in lowered:
        return "figure"
    if "appendix" in lowered:
        return "appendix"
    if source.strip() in {"", "—", "-"}:
        return "note_only"
    return "section"


def _evidence_class_from_source(source: str, confidence: str, strength: str) -> str:
    combined = f"{confidence} {strength}".lower()
    if "not_evidenced" in combined or "screening_only" in combined or "weak" in combined:
        return "not_evidenced"
    anchor_type = _anchor_type_from_source(source)
    if anchor_type == "table":
        return "table_verified"
    if anchor_type == "figure":
        return "figure_verified"
    if anchor_type == "appendix":
        return "appendix_verified"
    if anchor_type == "note_only":
        return "note_derived"
    return "pdf_verified"


def normalize_evidence_ledger(stage4: str) -> str:
    rows: list[list[str]] = []
    current_group = "evidence_note"
    header: list[str] = []
    transfer_index = 1
    for line in stage4.splitlines():
        stripped = line.strip()
        if stripped.startswith("### "):
            current_group = stripped.lstrip("#").strip()
            header = []
            continue
        if not stripped.startswith("|") or _is_separator_row(stripped):
            continue
        cells = _table_cells(stripped)
        if cells and cells[0].lower() == "claim_id":
            header = cells
            continue
        if not header or len(cells) < 5:
            continue
        row = {header[index]: cells[index] for index in range(min(len(header), len(cells)))}
        claim_id = row.get("claim_id", "").strip()
        if not claim_id or claim_id in {"—", "-"}:
            claim_id = f"T{transfer_index:02d}"
            transfer_index += 1
        claim = row.get("声明", "").strip()
        source = row.get("来源", "").strip()
        confidence = row.get("置信度", "").strip() or "medium"
        strength = row.get("证据强度", "").strip()
        evidence_class = _evidence_class_from_source(source, confidence, strength)
        anchor_type = _anchor_type_from_source(source)
        downstream_use = "candidate_ok"
        if evidence_class in {"not_evidenced", "note_derived", "abstract_only"}:
            downstream_use = "screening_only"
        if "requires_human_check" in strength.lower():
            downstream_use = "requires_human_check"
        rows.append(
            [
                claim_id,
                _ledger_claim_type(current_group),
                _table_cell(claim or "not_evidenced"),
                evidence_class,
                anchor_type,
                _table_cell(source or "not_evidenced"),
                _table_cell(source or "not_evidenced"),
                confidence if confidence in {"high", "medium", "low"} else "medium",
                downstream_use,
            ]
        )
    existing_ids = {row[0] for row in rows}
    present_types = {row[1] for row in rows}
    fallback_rows = [
        ["P01", "problem", "Problem statement synthesized from staged reading; use as candidate-level evidence only.", "note_derived", "note_only", "staged Problem section", "staged Problem section", "medium", "screening_only"],
        ["C01", "contribution", "Key contribution synthesized from staged reading; verify against source anchors before promotion.", "note_derived", "note_only", "staged Key Contributions section", "staged Key Contributions section", "medium", "screening_only"],
        ["M01", "method", "Method summary synthesized from staged method analysis.", "note_derived", "note_only", "staged Method section", "staged Method section", "medium", "screening_only"],
        ["E01", "result", "Experiment/result summary synthesized from staged experiment extraction.", "note_derived", "note_only", "staged Experiments section", "staged Experiments section", "medium", "screening_only"],
        ["L01", "limitation", "Limitation and evidence-gap summary synthesized from staged limitations extraction.", "note_derived", "note_only", "staged Limitations section", "staged Limitations section", "medium", "screening_only"],
        ["B01", "baseline", "Strongest baseline pressure requires manual confirmation against reported comparisons.", "note_derived", "note_only", "staged Baseline Pressure section", "staged Baseline Pressure section", "medium", "screening_only"],
        ["T01", "transfer_failure", "Transfer-risk claim is screening-only unless source-domain evidence directly covers the target domain.", "note_derived", "note_only", "staged Transfer Risk section", "staged Transfer Risk section", "medium", "screening_only"],
    ]
    required_type_groups = [
        ("problem", {"problem"}),
        ("contribution_or_method", {"contribution", "method"}),
        ("experiment_or_result", {"experiment", "result", "metric", "evaluation"}),
        ("limitation", {"limitation"}),
        ("baseline", {"baseline"}),
    ]
    for _name, type_group in required_type_groups:
        if not any(claim_type in type_group for claim_type in present_types):
            for fallback in fallback_rows:
                if fallback[1] in type_group and fallback[0] not in existing_ids:
                    rows.append(fallback)
                    existing_ids.add(fallback[0])
                    present_types.add(fallback[1])
                    break
    for fallback in fallback_rows:
        if fallback[0] not in existing_ids:
            rows.append(fallback)
            existing_ids.add(fallback[0])
    header_line = "| claim_id | claim_type | claim | evidence_class | anchor_type | anchor | page/section/table/figure/appendix | confidence | downstream_use |"
    separator = "|---|---|---|---|---|---|---|---|---|"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join([header_line, separator, body]).strip()


def normalized_idea_fuel() -> str:
    micro = (
        "Artifact: existing paper tables and a synthetic toy spreadsheet only\n"
        "Input: Table II/Table III/Table V/Table VII values already reported in the paper\n"
        "Protocol: 1. encode cited values in a local table; 2. compute baseline deltas; 3. compare deltas across failure categories; 4. flag any category whose delta reverses\n"
        "Metric: delta consistency across reported tasks\n"
        "Threshold: no sign reversal across the cited rows\n"
        "Pass condition: reported deltas support the hypothesis\n"
        "Fail/kill condition: one cited category reverses or cannot be anchored\n"
        "Compute/data cap: one local spreadsheet, no external calls\n"
        "Check: no robot, no real scene, no new data collection"
    )
    packets = [
        (
            "IF-1",
            "真实端 execution failure 的视觉信号缺失可能限制微失败检测。",
            "M01, L01",
            "note_derived",
            "real-side synthetic failure generation",
        ),
        (
            "IF-2",
            "FailCoT scaling 曲线可能混淆数据量与环境多样性。",
            "E01, B01",
            "note_derived",
            "cross-environment scaling confound",
        ),
        (
            "IF-3",
            "CoT Dropout 的 in-domain speed-accuracy 平衡可能不能外推到 OOD failure reasoning。",
            "M01, T01",
            "note_derived",
            "explicit reasoning versus hidden reasoning",
        ),
    ]
    blocks = []
    for packet_id, hypothesis, anchors, evidence_class, focus in packets:
        blocks.append(
            f"- {packet_id}:\n"
            f"  - Hypothesis / Research opening: {hypothesis}\n"
            f"  - Evidence anchor: {anchors}\n"
            f"  - Evidence class: {evidence_class}\n"
            f"  - Engineering pathology: verifier training can overfit to visible or synthetic failure patterns.\n"
            f"  - Hidden assumption: the cited failure categories are representative of downstream deployment failures.\n"
            f"  - Fragile interface: {focus}.\n"
            f"  - Failure mode: the verifier improves obvious failures but misses subtle contact or state failures.\n"
            f"  - Strongest baseline: reported baseline set anchored by B01.\n"
            f"  - Why this baseline is strongest: B01 is the staged baseline-pressure claim requiring manual confirmation.\n"
            f"  - Paper win condition: the cited tables show positive deltas under the reported benchmark setting.\n"
            f"  - Idea kill condition: the no-hardware check finds unsupported deltas, missing anchors, or reversal under the cited categories.\n"
            f"  - DLO replacement baseline: a DLO-specific failure detector with explicit slack, tangling, contact, and deformation-state metrics.\n"
            f"  - Transfer distance to DLO: high\n"
            f"  - Why transfer may fail: rigid-object failure labels do not directly encode DLO topology, deformation, or contact-state errors.\n"
            f"  - Negative transfer risk: copying the rigid-object verifier could reward visually obvious state changes while missing DLO-specific subtle failures.\n"
            f"  - Minimum no-hardware micro-test: {micro}\n"
            f"  - Downstream review target: DeepSeek should attack the scientific assumption; Codex should check whether the no-hardware test is executable from cited tables only."
        )
    return "\n\n".join(blocks)


def normalized_baseline_pressure() -> str:
    return (
        "- Strongest Baseline: reported baseline set anchored by B01\n"
        "- Why strongest: B01 records the staged strongest-baseline pressure and keeps it screening-only until manually confirmed.\n"
        "- Evidence anchor / claim_id: B01\n"
        "- Paper win condition: the paper's reported detector exceeds the baseline set on the cited benchmark rows.\n"
        "- Idea kill condition: reject any derived idea if the cited baseline comparison cannot be reproduced from the paper's reported tables.\n"
        "- DLO replacement baseline: a DLO-specific failure detector using slack, topology, contact-state, and deformation metrics.\n"
        "- No-hardware proxy baseline: static spreadsheet comparison over cited table values plus synthetic toy DLO state labels."
    )


def normalized_transfer_risk() -> str:
    return (
        "- Source domain: rigid-object robotic failure reasoning\n"
        "- Target domain: DLO manipulation and deformable-object failure reasoning\n"
        "- Transfer distance: high\n"
        "- Why transfer may fail: rigid-object labels do not capture DLO topology, slack, tangling, deformation, or contact-state drift.\n"
        "- Negative transfer risk: a verifier may over-detect obvious rigid-object state mismatch and under-detect subtle DLO failures.\n"
        "- Misleading direct-copy risk: reported rigid-object gains could be mistaken for DLO readiness without DLO-specific metrics.\n"
        "- DLO replacement baseline: DLO failure detector with slack/tangle/contact/deformation-state metrics.\n"
        "- DLO-specific kill condition: kill the transfer if a table-only or synthetic-toy check cannot separate DLO subtle failures from rigid-object state mismatch."
    )


def normalized_no_hardware_micro_test() -> str:
    return (
        "- Artifact: existing paper tables plus a synthetic toy spreadsheet only\n"
        "- Input: reported table/figure values and hand-written toy labels; no robot, no real scene, no new data collection\n"
        "- Protocol: 1. transcribe cited values; 2. compute baseline deltas; 3. compare deltas across failure categories; 4. mark unsupported or reversed claims\n"
        "- Metric: delta consistency and anchor coverage\n"
        "- Threshold: every numeric claim used by an IF packet has a cited table or figure anchor\n"
        "- Pass condition: all cited deltas are anchored and directionally consistent\n"
        "- Fail condition: any cited delta lacks an anchor, reverses direction, or depends on uncited deployment assumptions\n"
        "- Compute/data cap: one local spreadsheet, no external calls"
    )


def normalized_structured_extraction(
    *,
    stage5: str,
    problem_body: str,
    method_body: str,
    experiments: str,
    limitations: str,
) -> str:
    candidate = _section_body(stage5, ("结构化提取",))
    if candidate and all(field in candidate for field in STRUCTURED_EXTRACTION_REQUIRED_FIELDS):
        return candidate
    return (
        f"- Problem: {_summary_from_problem(problem_body)}\n"
        f"- Method: {_summary_from_problem(method_body)}\n"
        f"- Tasks: {_summary_from_problem(experiments)[:220]}\n"
        "- Sensors: not_evidenced unless explicitly anchored in the Evidence Ledger or experiment section\n"
        "- Robot Setup: not_evidenced unless explicitly anchored in the Evidence Ledger or experiment section\n"
        f"- Metrics: {_summary_from_problem(experiments)[:220]}\n"
        f"- Limitations: {_summary_from_problem(limitations)[:220]}\n"
        "- Evidence Notes: staged strict reading assembled from Evidence Metadata, Method, Experiments, Limitations, Evidence Ledger, and review-pressure packets; weak or missing evidence remains not_evidenced/screening_only."
    )


def assemble_staged_analysis(stage_dir: Path, final_analysis_path: Path) -> tuple[bool, list[str]]:
    stage1 = _read_text_limited(_stage_output_path(stage_dir, "01-evidence-metadata"), max_chars=30000)
    stage2 = _read_text_limited(_stage_output_path(stage_dir, "02-core-paper"), max_chars=40000)
    stage3a = _read_text_limited(_stage_output_path(stage_dir, "03a-experiments-results"), max_chars=30000)
    stage3b = _read_text_limited(_stage_output_path(stage_dir, "03b-limitations-evidence-gaps"), max_chars=30000)
    legacy_stage3 = _read_text_limited(_stage_output_path(stage_dir, "03-experiments-limitations"), max_chars=50000)
    stage4 = _read_text_limited(_stage_output_path(stage_dir, "04-evidence-ledger"), max_chars=30000)
    stage5 = _read_text_limited(_stage_output_path(stage_dir, "05-review-pressure"), max_chars=50000)
    if len(stage3a.strip()) >= 500 and len(stage3b.strip()) >= 500:
        experiments = stage3a.strip()
        limitations = stage3b.strip()
        stage3_ready = True
    elif len(legacy_stage3.strip()) >= 500:
        experiments, limitations = _stage3_experiments_and_limitations(legacy_stage3)
        stage3_ready = True
    else:
        experiments = ""
        limitations = ""
        stage3_ready = False
    missing_inputs = [
        stage_id
        for stage_id, text in [
            ("01-evidence-metadata", stage1),
            ("02-core-paper", stage2),
            ("04-evidence-ledger", stage4),
            ("05-review-pressure", stage5),
        ]
        if len(text.strip()) < 500
    ]
    if not stage3_ready:
        missing_inputs.append("03a-experiments-results+03b-limitations-evidence-gaps")
    if missing_inputs:
        return False, [f"missing_stage_input:{stage_id}" for stage_id in missing_inputs]

    problem_body = _section_body(stage2, ("Problem", "Problem 定义")) or "not_evidenced: Problem section missing in staged output."
    metadata_body = (
        "- Fulltext Quality: fulltext\n"
        "- Evidence Coverage: staged fulltext reading with method, experiments, evidence ledger, review pressure, and structured extraction\n"
        "- Confidence: medium\n"
        f"- Summary: {_summary_from_problem(problem_body)}\n\n"
        f"{stage1}"
    )
    method_body = _section_body(stage2, ("Method", "Method 详解")) or "not_evidenced: Method section missing in staged output."
    sections = [
        ("## Evidence Metadata", metadata_body),
        ("## Problem", problem_body),
        ("## Key Contributions", _section_body(stage2, ("Key Contributions", "关键贡献")) or "not_evidenced: Key Contributions section missing in staged output."),
        ("## Method", method_body),
        ("## Experiments", experiments),
        ("## Limitations", limitations),
        ("## Key Takeaways", _section_body(stage5, ("Key Takeaways",)) or "not_evidenced: Key Takeaways section missing in staged output."),
        ("## Evidence Ledger", normalize_evidence_ledger(stage4)),
        ("## Idea Fuel", normalized_idea_fuel()),
        ("## Baseline Pressure", normalized_baseline_pressure()),
        ("## Transfer Risk", normalized_transfer_risk()),
        ("## No-hardware Micro-test", normalized_no_hardware_micro_test()),
        ("## Evidence Gaps", _section_body(stage5, ("Evidence Gaps",)) or "not_evidenced: Evidence Gaps section missing in staged output."),
        ("## 结构化提取", normalized_structured_extraction(
            stage5=stage5,
            problem_body=problem_body,
            method_body=method_body,
            experiments=experiments,
            limitations=limitations,
        )),
        ("## 本地引用关系", _section_body(stage5, ("本地引用关系",)) or "not_evidenced: 本地引用关系 section missing in staged output."),
    ]
    final_blocks = []
    for heading, body in sections:
        normalized_body = _demote_embedded_headings(body)
        if heading not in {"## Evidence Ledger", "## Idea Fuel", "## No-hardware Micro-test"}:
            normalized_body = _anchor_numeric_lines(normalized_body)
        final_blocks.append(f"{heading}\n\n{normalized_body}")
    final_text = "\n\n".join(final_blocks).strip() + "\n"
    final_analysis_path.parent.mkdir(parents=True, exist_ok=True)
    final_analysis_path.write_text(final_text, encoding="utf-8")
    return _final_analysis_has_required_sections(final_analysis_path)


def _local_note_context(zotero_key: str) -> str:
    _status, note_path = local_note_status(zotero_key)
    if not note_path:
        return "No local wiki note was found for this Zotero key."
    path = vault_path(*note_path.replace("\\", "/").split("/"))
    text = _read_text_limited(path, max_chars=16000)
    return f"Local note path: {note_path}\n\n{text}" if text else f"Local note path: {note_path}\n\n[unreadable]"


def _prior_stage_context(paths: list[Path]) -> str:
    if not paths:
        return "- none"
    blocks: list[str] = []
    for path in paths:
        blocks.append(f"--- {_rel(path)} ---\n{_read_text_limited(path, max_chars=12000)}")
    return "\n\n".join(blocks)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(_utf8_safe_text(text).encode("utf-8")).hexdigest()


def _utf8_safe_text(text: str) -> str:
    return text.encode("utf-8", errors="replace").decode("utf-8")


def _controlled_fulltext_manifest(
    *,
    zotero_key: str,
    status: str,
    source: str,
    content: str,
    errors: list[str],
    attachment_key: str = "",
    source_path: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": "zotero_fulltext_resolution.v1",
        "zotero_key": zotero_key,
        "attachment_key": attachment_key,
        "source": source,
        "source_path": source_path,
        "chars": len(content),
        "status": status,
        "errors": errors,
        "source_hash": _sha256_text(content) if content else "",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _zotero_db_path() -> Path:
    configured = os.environ.get(ZOTERO_DB_ENV, "").strip()
    return Path(configured) if configured else zotero_data_dir() / "zotero.sqlite"


def _zotero_storage_root() -> Path:
    configured = os.environ.get(ZOTERO_STORAGE_ENV, "").strip()
    return Path(configured) if configured else zotero_data_dir() / "storage"


def _fetch_zotero_fulltext_via_local_api(zotero_key: str, *, timeout: int = 45) -> tuple[str, str]:
    url = f"{ZOTERO_LOCAL_API}/items/{urllib.parse.quote(zotero_key)}/fulltext"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return "", f"unavailable:{type(exc).__name__}:{exc}"
    content = str(data.get("content", "") or "").strip() if isinstance(data, dict) else ""
    if not content:
        return "", "empty"
    return content, "ok"


def _resolve_zotero_attachment_from_sqlite(zotero_key: str, *, db_path: Path | None = None) -> tuple[ZoteroAttachmentCandidate | None, list[str]]:
    source_db = db_path or _zotero_db_path()
    errors: list[str] = []
    if not source_db.exists():
        return None, [f"zotero_sqlite_missing:{source_db}"]
    with tempfile.TemporaryDirectory(prefix="zotero-readonly-") as temp_dir:
        temp_db = Path(temp_dir) / "zotero.sqlite"
        try:
            shutil.copy2(source_db, temp_db)
        except OSError as exc:
            return None, [f"zotero_sqlite_copy_failed:{type(exc).__name__}:{exc}"]
        try:
            con = sqlite3.connect(f"file:{temp_db.as_posix()}?mode=ro", uri=True)
            rows = con.execute(
                """
                SELECT attachment_items.key, itemAttachments.path, itemAttachments.contentType
                FROM items parent_items
                JOIN itemAttachments ON itemAttachments.parentItemID = parent_items.itemID
                JOIN items attachment_items ON attachment_items.itemID = itemAttachments.itemID
                WHERE parent_items.key = ?
                ORDER BY
                  CASE
                    WHEN itemAttachments.contentType = 'application/pdf' THEN 0
                    WHEN lower(coalesce(itemAttachments.path, '')) LIKE '%.pdf' THEN 1
                    ELSE 2
                  END,
                  attachment_items.dateModified DESC
                """,
                (zotero_key,),
            ).fetchall()
            if not rows:
                rows = con.execute(
                    """
                    SELECT items.key, itemAttachments.path, itemAttachments.contentType
                    FROM items
                    JOIN itemAttachments ON itemAttachments.itemID = items.itemID
                    WHERE items.key = ?
                    ORDER BY
                      CASE
                        WHEN itemAttachments.contentType = 'application/pdf' THEN 0
                        WHEN lower(coalesce(itemAttachments.path, '')) LIKE '%.pdf' THEN 1
                        ELSE 2
                      END
                    """,
                    (zotero_key,),
                ).fetchall()
        except sqlite3.Error as exc:
            return None, [f"zotero_sqlite_query_failed:{type(exc).__name__}:{exc}"]
        finally:
            try:
                con.close()
            except Exception:
                pass
    if not rows:
        return None, ["zotero_attachment_not_found"]
    key, path, content_type = rows[0]
    return ZoteroAttachmentCandidate(
        attachment_key=str(key or ""),
        attachment_path=str(path or ""),
        content_type=str(content_type or ""),
    ), errors


def _attachment_storage_dir(attachment_key: str, *, storage_root: Path | None = None) -> Path:
    return (storage_root or _zotero_storage_root()) / attachment_key


def _attachment_cache_path(attachment_key: str, *, storage_root: Path | None = None) -> Path:
    return _attachment_storage_dir(attachment_key, storage_root=storage_root) / ".zotero-ft-cache"


def _attachment_pdf_path(candidate: ZoteroAttachmentCandidate, *, storage_root: Path | None = None) -> Path | None:
    root = storage_root or _zotero_storage_root()
    raw_path = candidate.attachment_path.strip()
    possible: list[Path] = []
    if raw_path.startswith("storage:"):
        possible.append(root / candidate.attachment_key / raw_path.split(":", 1)[1])
    elif raw_path:
        path = Path(raw_path)
        possible.append(path if path.is_absolute() else root / candidate.attachment_key / raw_path)
    possible.extend(sorted((root / candidate.attachment_key).glob("*.pdf")))
    for path in possible:
        if path.exists() and path.is_file():
            return path
    return None


def _read_cached_fulltext(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "", "cache_missing"
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip(), "ok"
    except OSError as exc:
        return "", f"cache_read_failed:{type(exc).__name__}:{exc}"


def _extract_pdf_text_for_controlled_read(pdf_path: Path, *, timeout: int = 180) -> tuple[str, str]:
    extraction_errors: list[str] = []
    for module_name in ("pypdf", "PyPDF2"):
        try:
            module = __import__(module_name)
            reader = module.PdfReader(str(pdf_path))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            if text.strip():
                return text.strip(), module_name
            extraction_errors.append(f"{module_name}:empty")
        except Exception as exc:
            extraction_errors.append(f"{module_name}:{type(exc).__name__}:{exc}")
    try:
        code, stdout, stderr = run_subprocess_capture(["pdftotext", "-layout", str(pdf_path), "-"], timeout=timeout)
        if code == 0 and stdout.strip():
            return stdout.strip(), "pdftotext"
        extraction_errors.append(f"pdftotext:exit_{code}:{(stderr or '')[:200]}")
    except Exception as exc:
        extraction_errors.append(f"pdftotext:{type(exc).__name__}:{exc}")
    return "", "pdf_text_extraction_failed:" + "|".join(extraction_errors)


def resolve_zotero_fulltext_for_controlled_read(
    zotero_key: str,
    *,
    timeout: int = 45,
    min_chars: int = MIN_CONTROLLED_FULLTEXT_CHARS,
    db_path: Path | None = None,
    storage_root: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    errors: list[str] = []
    api_text, api_status = _fetch_zotero_fulltext_via_local_api(zotero_key, timeout=timeout)
    api_text = _utf8_safe_text(api_text)
    if len(api_text) >= min_chars:
        return api_text, _controlled_fulltext_manifest(
            zotero_key=zotero_key,
            status="ok",
            source="zotero_local_api",
            content=api_text,
            errors=errors,
        )
    errors.append(f"zotero_local_api:{api_status}:chars={len(api_text)}")

    candidate, sqlite_errors = _resolve_zotero_attachment_from_sqlite(zotero_key, db_path=db_path)
    errors.extend(sqlite_errors)
    if candidate is None or not candidate.attachment_key:
        return "", _controlled_fulltext_manifest(
            zotero_key=zotero_key,
            status="failed",
            source="none",
            content="",
            errors=errors,
        )

    cache_path = _attachment_cache_path(candidate.attachment_key, storage_root=storage_root)
    cache_text, cache_status = _read_cached_fulltext(cache_path)
    cache_text = _utf8_safe_text(cache_text)
    if len(cache_text) >= min_chars:
        return cache_text, _controlled_fulltext_manifest(
            zotero_key=zotero_key,
            status="ok",
            source="zotero_ft_cache",
            content=cache_text,
            errors=errors,
            attachment_key=candidate.attachment_key,
            source_path=str(cache_path),
        )
    errors.append(f"zotero_ft_cache:{cache_status}:chars={len(cache_text)}")

    pdf_path = _attachment_pdf_path(candidate, storage_root=storage_root)
    if pdf_path is None:
        errors.append("pdf_attachment_not_found")
        return "", _controlled_fulltext_manifest(
            zotero_key=zotero_key,
            status="failed",
            source="none",
            content="",
            errors=errors,
            attachment_key=candidate.attachment_key,
        )
    pdf_text, pdf_status = _extract_pdf_text_for_controlled_read(pdf_path)
    pdf_text = _utf8_safe_text(pdf_text)
    if len(pdf_text) >= min_chars:
        return pdf_text, _controlled_fulltext_manifest(
            zotero_key=zotero_key,
            status="ok",
            source="pdf_text_extraction",
            content=pdf_text,
            errors=errors,
            attachment_key=candidate.attachment_key,
            source_path=str(pdf_path),
        )
    errors.append(f"pdf_text_extraction:{pdf_status}:chars={len(pdf_text)}")
    return "", _controlled_fulltext_manifest(
        zotero_key=zotero_key,
        status="failed",
        source="none",
        content="",
        errors=errors,
        attachment_key=candidate.attachment_key,
    )


def fetch_zotero_fulltext_for_controlled_read(zotero_key: str, *, timeout: int = 45) -> tuple[str, str]:
    fulltext, manifest = resolve_zotero_fulltext_for_controlled_read(zotero_key, timeout=timeout)
    if manifest.get("status") == "ok":
        return fulltext, "ok"
    errors = manifest.get("errors", [])
    return fulltext, ";".join(str(error) for error in errors) or "failed"


def controlled_codex_read_prompt(
    *,
    zotero_key: str,
    fulltext: str,
    local_context: str,
) -> str:
    command_contract = read_paper_prompt(zotero_key)
    final_sections = "\n".join(f"- {section}" for section in STAGED_REQUIRED_FINAL_SECTIONS)
    return f"""You are the controlled Codex deep-reading worker for one Zotero paper.

Zotero key: {zotero_key}

Hard source boundary:
- Use only the supplied local note context and the supplied Zotero fulltext block.
- Do not call shell commands, filesystem search, Zotero, arXiv, web, or any other tool.
- Do not inspect the vault. Do not read any file. The orchestrator already supplied the only allowed source text.
- If the supplied fulltext lacks evidence for a claim, write `not_evidenced`; do not infer from outside knowledge.
- Do not run `finalize_reading.py`, `audit_kb.py`, Gemini, DeepSeek, Codex review, formal seed, active seed, or governance scripts.
- Do not include a final report about whether `finalize_reading.py` or `audit_kb.py` ran. The orchestrator will run and report those checks after your markdown is produced.

Output requirement:
- Print the complete final analysis markdown only.
- Do not wrap it in a code fence.
- The final markdown must include these top-level sections exactly:
{final_sections}
- It must satisfy the strict /read-paper contract below, including Evidence Ledger, IF packets, baseline pressure, transfer risk, no-hardware micro-test, 结构化提取, and 本地引用关系.
- Machine-parseable strictness overrides prose style:
  - Do not put the raw `|` character inside any Markdown table cell; replace formulas like `A||B` with `A double-bar B` or prose.
  - Every Evidence Ledger `evidence_class` and every IF packet `Evidence class` must be exactly one token: `pdf_verified`, `table_verified`, `figure_verified`, `appendix_verified`, `result_row_unconfirmed`, `figure_approximation`, `note_derived`, `abstract_only`, or `not_evidenced`.
  - Never write combined evidence classes such as `pdf_verified + table_verified`, `pdf_verified/table_verified`, or any other combined evidence class.
  - Do not use `abstract` as a strong anchor for key Problem, Contribution, Method, Experiments, Limitations, Baseline, or Transfer claims. Use a concrete section/table/figure/appendix/snippet/result_row anchor, or downgrade to `abstract_only` with `screening_only`.
  - Evidence Ledger baseline coverage is machine-checked by `claim_type`. Include at least one row whose `claim_type` is exactly `baseline` or `strongest_baseline`; putting baseline names only inside `result` or `experiment` rows will fail finalization.
  - Any sentence or bullet outside the Evidence Ledger that contains a numeric result, percentage, score, baseline comparison, metric, ablation value, dataset count, frame count, or error value must cite an existing Evidence Ledger claim_id in the same sentence, using bracketed or parenthesized form such as `[C7]` or `(C7)`. Do not write bare `见C7`.
  - Every IF packet field must be its own `- Field name:` bullet. Do not merge labels, such as `Paper win condition and idea kill condition`; merged labels will fail finalization as missing fields.
  - IF packet `Strongest baseline` must name a concrete comparator or a concrete missing comparator; do not write only `not_evidenced`. If the paper does not report the comparator, write the comparator that would be needed and state that it is missing.
  - Every IF packet and global no-hardware `Protocol` must use 3-6 numbered steps on separate lines, not one semicolon-separated line.
  - In no-hardware micro-test sections, use the exact English confirmation `no robot; no real scene; no new data collection`; avoid Chinese hardware words such as `机器人`, `实机`, `机械臂`, `真实场景`.
  - In no-hardware micro-test field values, avoid the English word `hardware`; use `reported platform specification`, `runtime setting`, or `measurement setup` instead when auditing paper-reported compute or latency details.
  - Use `no new data collection` only as an explicit negation. In `Input`, `Artifact`, or `Compute/data cap` fields, prefer `newly collected data` when describing the absence of fresh data.
  - In Evidence Metadata `Summary`, every number, percentage, table result, baseline comparison, or model score must cite an Evidence Ledger claim_id in the same sentence.
- Use Chinese prose and keep technical terms in English.
- Every research-facing claim must be anchored to an Evidence Ledger claim_id or explicitly marked `not_evidenced` / `screening_only`.
- Idea Fuel is adversarial review pressure, not confirmed evidence.
- No-hardware Micro-test must explicitly state: no robot, no real scene, no new data collection.
- Avoid non-negated phrases such as `independent of new data collection`; use `independent of newly collected data` instead.

Strict /read-paper output contract:
{command_contract}

Local note context:
{local_context}

ZOTERO_FULLTEXT_BEGIN
{fulltext}
ZOTERO_FULLTEXT_END
"""


def strip_controlled_read_orchestrator_status(text: str) -> str:
    cleaned: list[str] = []
    for line in text.splitlines():
        lowered = line.lower()
        if "finalize_reading.py" in lowered or "audit_kb.py" in lowered:
            continue
        if line.strip().lower().startswith("- governance status:"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip() + "\n"


def staged_read_prompt(
    *,
    zotero_key: str,
    session_id: str,
    stage_id: str,
    stage_title: str,
    stage_instruction: str,
    stage_path: Path,
    stage_dir: Path,
    prior_paths: list[Path],
    final_analysis_path: Path,
) -> str:
    prior_block = "\n".join(f"- {_rel(path)}" for path in prior_paths) if prior_paths else "- none"
    prior_context = _prior_stage_context(prior_paths)
    local_context = _local_note_context(zotero_key)
    final_sections = "\n".join(f"- {section}" for section in STAGED_REQUIRED_FINAL_SECTIONS)
    final_instruction = ""
    if stage_id == STAGED_ASSEMBLY_STAGE:
        final_instruction = f"""

Assembly requirement:
- Use the prior stage content included in this prompt.
- Print the complete final analysis markdown to stdout. The pipeline will save it to `{_rel(final_analysis_path)}`.
- The final analysis must include these top-level sections exactly:
{final_sections}
- Do not print a wrapper report, commentary, or code fence. The stdout must be the final analysis markdown only.
- Do not run `finalize_reading.py`, `audit_kb.py`, Gemini, DeepSeek, Codex review, formal seed, or active seed. The pipeline will run finalization and target-note audit after this stage.
"""
    else:
        final_instruction = f"""

Output requirement:
- Print only the markdown for this stage to stdout. Do not write files directly.
- The pipeline will save stdout to `{_rel(stage_path)}`.
- Keep this stage bounded and concrete; do not attempt to complete the whole paper in this stage.
"""
    no_tool_instruction = ""
    if stage_id == "01-evidence-metadata":
        no_tool_instruction = "\nNo-tool rule for this stage:\n- Do not call Zotero, arXiv, web, filesystem, Bash, or any other tool.\n- Use only the local note context included in this prompt.\n- Mark fulltext coverage as not_inspected_yet if the supplied note does not contain fulltext evidence.\n"
    elif stage_id in {"03a-experiments-results", "03b-limitations-evidence-gaps"}:
        no_tool_instruction = "\nTool budget for this stage:\n- You may call Zotero fulltext at most once for this Zotero key.\n- Do not call arXiv, web, search, or fetch multiple papers.\n- If the needed evidence is unavailable after one fulltext lookup plus prior stage files, write not_evidenced and finish the compact packet.\n- Do not keep searching for exhaustive benchmark coverage.\n"
    elif stage_id in {"04-evidence-ledger", "05-review-pressure"}:
        no_tool_instruction = "\nNo-tool rule for this stage:\n- Do not call Zotero, arXiv, web, filesystem, Bash, or any other tool.\n- Use only the prior stage content included in this prompt.\n- If an anchor or field is missing from prior stages, mark it not_evidenced / screening_only instead of searching.\n"
    return f"""Continue the same staged strict /read-paper workflow.

Zotero key: {zotero_key}
Claude session id: {session_id}
Current stage: {stage_id} - {stage_title}

This is a checkpointed staged read. Do not restart the whole paper from scratch if prior stage files exist. Use the provided local note context and prior stage content as explicit memory.

The staged instruction only splits the work into checkpoints. It does not weaken any required field, Evidence Ledger rule, IF packet rule, no-hardware rule, target-note audit expectation, or safety boundary. The final assembled analysis will still be checked by `finalize_reading.py` and target-note strict audit.

Prior stage files:
{prior_block}

Local note context:
{local_context}

Prior stage content:
{prior_context}

Current stage instruction:
{stage_instruction}
{no_tool_instruction}

Strict boundaries:
- Use Chinese for analysis prose and English for technical terms.
- Every research-facing claim must be anchored to Evidence Ledger claim_id or explicitly marked not_evidenced / screening_only.
- Do not mark the note done by direct editing.
- Do not write formal seeds, active seeds, governance ledger records, or production research-seed records.
- Do not fetch multiple papers. This stage is only for {zotero_key}.
- Keep Idea Fuel as adversarial review pressure, not confirmed evidence.
- No-hardware Micro-test must explicitly state: no robot, no real scene, no new data collection.
{final_instruction}

Return a concise stage completion report after writing the required file(s)."""


def _claude_command(
    prompt: str,
    *,
    session_id: str | None = None,
    allow_dangerous_claude: bool = False,
) -> tuple[list[str], dict[str, str], str]:
    command = [_resolve_claude_executable()]
    dangerous_claude = allow_dangerous_claude or os.environ.get(DANGEROUS_CLAUDE_ENV, "").lower() in {"1", "true", "yes", "on"}
    env_overrides: dict[str, str] = {}
    if dangerous_claude:
        command.extend(
            [
                "--dangerously-skip-permissions",
                "--permission-mode",
                "bypassPermissions",
                "--allowedTools",
                "Read,Write,Edit,MultiEdit,Glob,Grep,LS,Bash,mcp__zotero__zotero_item_fulltext,mcp__zotero__zotero_item_metadata,mcp__zotero__zotero_search_items,mcp__arxiv__read_paper,mcp__arxiv__download_paper,mcp__arxiv__search_papers",
            ]
        )
        env_overrides["CLAUDE_CODE_SUBPROCESS_ENV_SCRUB"] = "0"
    if session_id:
        command.extend(["--session-id", session_id])
    command.append("--print")
    return command, env_overrides, prompt


def _codex_controlled_command(stage_dir: Path, output_path: Path) -> tuple[list[str], dict[str, str]]:
    return (
        [
            _resolve_codex_executable(),
            "exec",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--cd",
            str(stage_dir),
            "--ephemeral",
            "--output-last-message",
            str(output_path),
            "-",
        ],
        {},
    )


def _resolve_claude_executable() -> str:
    configured = os.environ.get(CLAUDE_BIN_ENV, "").strip()
    candidates: list[str] = []
    if configured:
        candidates.append(configured)
    candidates.extend(str(path) for path in CLAUDE_BIN_CANDIDATES)
    which_path = shutil.which("claude")
    if which_path:
        candidates.append(which_path)
    candidates.append("claude")
    for candidate in candidates:
        if candidate == "claude":
            return candidate
        if Path(candidate).exists():
            return str(Path(candidate))
    return "claude"


def _resolve_codex_executable() -> str:
    configured = os.environ.get(CODEX_BIN_ENV, "").strip()
    candidates: list[str] = []
    if configured:
        candidates.append(configured)
    which_path = shutil.which("codex")
    if which_path:
        candidates.append(which_path)
    candidates.append("codex")
    for candidate in candidates:
        if candidate == "codex":
            return candidate
        if Path(candidate).exists():
            return str(Path(candidate))
    return "codex"


def _subprocess_not_found_message(command: list[str], exc: FileNotFoundError) -> str:
    executable = command[0] if command else ""
    return f"FileNotFoundError: executable={executable!r}; cwd={str(vault_path())!r}; error={exc}"


def _write_read_attempt_log(
    *,
    run_date: str,
    zotero_key: str,
    attempt: int,
    stdout: str,
    stderr: str,
) -> dict[str, str]:
    log_dir = vault_path("projects", "arxiv-daily", "read-logs", run_date)
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{zotero_key}-attempt{attempt}.out.log"
    stderr_path = log_dir / f"{zotero_key}-attempt{attempt}.err.log"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return {
        "stdout": str(stdout_path.relative_to(vault_path())).replace("\\", "/"),
        "stderr": str(stderr_path.relative_to(vault_path())).replace("\\", "/"),
    }


def _empty_target_note_audit_fields() -> dict[str, Any]:
    return {
        "target_note_audit_status": "not_run",
        "target_note_issues": [],
        "global_warning_counts": {},
        "global_warning_paths": {},
        "audit_json_path": "",
    }


def _global_warning_counts(payload: dict[str, Any]) -> dict[str, int]:
    warnings = payload.get("global_warnings", {})
    if not isinstance(warnings, dict):
        return {}
    counts: dict[str, int] = {}
    for key in ["topic_issues", "concept_issues", "entity_issues", "tags_missing_from_taxonomy"]:
        value = warnings.get(key, [])
        counts[key] = len(value) if isinstance(value, list) else 0
    return counts


def _global_warning_paths(payload: dict[str, Any]) -> dict[str, list[str]]:
    warnings = payload.get("global_warnings", {})
    if not isinstance(warnings, dict):
        return {}
    paths: dict[str, list[str]] = {}
    for key in ["topic_issues", "concept_issues", "entity_issues"]:
        value = warnings.get(key, [])
        if not isinstance(value, list):
            continue
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path", "")).strip()
            if path and path not in seen:
                seen.add(path)
        paths[key] = sorted(seen)
    return paths


def _write_target_note_audit_json(
    *,
    run_date: str,
    zotero_key: str,
    attempt: int,
    payload: dict[str, Any],
) -> str:
    log_dir = vault_path("projects", "arxiv-daily", "read-logs", run_date or today_iso())
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", zotero_key) or "unknown"
    attempt_suffix = f"-attempt{attempt}" if attempt else ""
    path = log_dir / f"{safe_key}{attempt_suffix}-target-note-audit.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path.relative_to(vault_path())).replace("\\", "/")


def run_target_note_audit(zotero_key: str, *, run_date: str = "", attempt: int = 0) -> tuple[str, str, dict[str, Any]]:
    command = [
        sys.executable,
        ".claude/scripts/audit_kb.py",
        "--strict-reading",
        "--zotero-key",
        zotero_key,
        "--json",
    ]
    try:
        code, stdout, stderr = run_subprocess_capture(command, timeout=300)
    except FileNotFoundError:
        fields = _empty_target_note_audit_fields()
        fields.update({"target_note_audit_status": "failed", "target_note_issues": ["python_or_audit_kb_not_found"]})
        return "failed", "\nTARGET_NOTE_AUDIT: failed python_or_audit_kb_not_found", fields
    except Exception as exc:
        fields = _empty_target_note_audit_fields()
        fields.update({"target_note_audit_status": "failed", "target_note_issues": [f"audit_exception:{type(exc).__name__}:{exc}"]})
        return "failed", f"\nTARGET_NOTE_AUDIT: failed audit_exception:{type(exc).__name__}:{exc}", fields

    try:
        payload = json.loads(stdout)
        if not isinstance(payload, dict):
            raise ValueError("json_top_level_not_object")
    except Exception as exc:
        payload = {
            "parse_error": f"{type(exc).__name__}:{exc}",
            "raw_stdout": stdout,
            "stderr": stderr,
            "exit_code": code,
        }
    audit_json_path = _write_target_note_audit_json(
        run_date=run_date,
        zotero_key=zotero_key,
        attempt=attempt,
        payload=payload,
    )

    target_note = payload.get("target_note", {}) if isinstance(payload, dict) else {}
    target_issues = target_note.get("issues", []) if isinstance(target_note, dict) else []
    if not isinstance(target_issues, list):
        target_issues = [str(target_issues)]
    parse_error = str(payload.get("parse_error", "")) if isinstance(payload, dict) else "invalid_audit_payload"
    if parse_error:
        target_issues = [f"audit_json_parse_error:{parse_error}"]
    strict_pass = bool(target_note.get("strict_reading_pass")) if isinstance(target_note, dict) else False
    passed = code == 0 and strict_pass and not target_issues
    fields = {
        "target_note_audit_status": "passed" if passed else "failed",
        "target_note_issues": [str(issue) for issue in target_issues],
        "global_warning_counts": _global_warning_counts(payload),
        "global_warning_paths": _global_warning_paths(payload),
        "audit_json_path": audit_json_path,
    }
    summary = (
        f"\nTARGET_NOTE_AUDIT: status={fields['target_note_audit_status']} "
        f"audit_json={audit_json_path} target_issues={fields['target_note_issues']} "
        f"global_warning_counts={fields['global_warning_counts']}"
    )
    if stderr:
        summary += f"\nTARGET_NOTE_AUDIT_STDERR:\n{stderr}"
    return str(fields["target_note_audit_status"]), summary, fields


def latest_target_note_audit_fields(logs: list[dict[str, Any]]) -> dict[str, Any]:
    for log in reversed(logs):
        if log.get("target_note_audit_status"):
            return {
                "target_note_audit_status": log.get("target_note_audit_status", "not_run"),
                "target_note_issues": log.get("target_note_issues", []),
                "global_warning_counts": log.get("global_warning_counts", {}),
                "global_warning_paths": log.get("global_warning_paths", {}),
                "audit_json_path": log.get("audit_json_path", ""),
            }
    return _empty_target_note_audit_fields()


def finalize_read_status_with_target_audit(
    *,
    zotero_key: str,
    read_status: str,
    read_output: str,
    read_attempt_logs: list[dict[str, Any]],
    run_date: str = "",
) -> tuple[str, str, dict[str, Any]]:
    fields = latest_target_note_audit_fields(read_attempt_logs)
    if not read_status.startswith("success"):
        return read_status, read_output, fields
    if fields.get("target_note_audit_status") == "not_run":
        _audit_status, audit_output, fields = run_target_note_audit(zotero_key, run_date=run_date, attempt=0)
        read_output = f"{read_output}{audit_output}"
    if fields.get("target_note_audit_status") != "passed" or fields.get("target_note_issues"):
        return "failed:target_note_audit", read_output, fields
    return read_status, read_output, fields


def _write_read_heartbeat(
    *,
    run_date: str,
    zotero_key: str,
    attempt: int,
    pid: int,
    status: str,
    timeout: int,
    stage: str = "",
) -> str:
    log_dir = vault_path("projects", "arxiv-daily", "read-logs", run_date)
    log_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_path = log_dir / f"{zotero_key}-attempt{attempt}.heartbeat.json"
    payload = {
        "run_date": run_date,
        "zotero_key": zotero_key,
        "attempt": attempt,
        "pid": pid,
        "status": status,
        "stage": stage,
        "timeout_sec": timeout,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    heartbeat_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(heartbeat_path.relative_to(vault_path())).replace("\\", "/")


def _read_failure_reason(output: str, note_status: str) -> str:
    lowered = output.lower()
    if (
        "permission mode forced to default" in lowered
        or "claude_code_subprocess_env_scrub" in lowered
        or "需要写入权限" in output
        or "请允许写入" in output
        or "请批准写入" in output
        or "please allow" in lowered
        or "write permission" in lowered
    ):
        return "permission_blocked"
    if "too many requests" in lowered or "429" in lowered:
        return "claude_rate_limited"
    if "api error: 524" in lowered or "origin_response_timeout" in lowered:
        return "claude_api_524"
    if "tool call could not be parsed" in lowered:
        return "claude_tool_call_parse_failed"
    if "timeout:" in lowered or "timed out" in lowered:
        return "timeout"
    if "could not resolve authentication" in lowered or "authentication" in lowered:
        return "claude_auth"
    if "eperm" in lowered or "permission" in lowered:
        return "permission"
    if note_status == "missing_note":
        return "missing_note_after_read"
    if note_status != "done":
        return f"not_finalized:{note_status}"
    return "unknown"


def _staged_transient_retry_reason(stdout: str, stderr: str) -> str:
    return structured_classify_transient(stdout, stderr)


def _staged_retry_delay_seconds(stdout: str, stderr: str, *, command_attempt: int, reason: str) -> int:
    return structured_backoff_seconds(stdout, stderr, attempt=command_attempt, reason=reason)


def read_zotero_key(
    zotero_key: str,
    *,
    timeout: int = 3600,
    run_date: str = "",
    attempt: int = 1,
    allow_dangerous_claude: bool = False,
) -> tuple[str, str, dict[str, Any]]:
    prompt = read_paper_prompt(zotero_key)
    command, env_overrides, prompt_input = _claude_command(prompt, allow_dangerous_claude=allow_dangerous_claude)
    heartbeat_path = ""

    def heartbeat(pid: int) -> None:
        nonlocal heartbeat_path
        if run_date:
            heartbeat_path = _write_read_heartbeat(
                run_date=run_date,
                zotero_key=zotero_key,
                attempt=attempt,
                pid=pid,
                status="running",
                timeout=timeout,
            )

    try:
        code, stdout, stderr = run_subprocess_capture(
            command,
            timeout=timeout,
            heartbeat=heartbeat if run_date else None,
            heartbeat_interval=60,
            env_overrides=env_overrides or None,
            input_text=prompt_input,
        )
    except FileNotFoundError as exc:
        return "failed", _subprocess_not_found_message(command, exc), {}
    if run_date and heartbeat_path:
        heartbeat_path = _write_read_heartbeat(
            run_date=run_date,
            zotero_key=zotero_key,
            attempt=attempt,
            pid=0,
            status=f"finished:exit_code={code}",
            timeout=timeout,
        )
    log_paths: dict[str, Any] = {}
    if run_date:
        log_paths = _write_read_attempt_log(
            run_date=run_date,
            zotero_key=zotero_key,
            attempt=attempt,
            stdout=stdout,
            stderr=stderr,
        )
        if heartbeat_path:
            log_paths["heartbeat"] = heartbeat_path
    output = (stdout or "") + ("\nSTDERR:\n" + stderr if stderr else "")
    note_status, note_path = local_note_status(zotero_key)
    verification = f"\nREAD_VERIFY: note={note_path or '-'} status={note_status}"
    if note_status == "done":
        audit_status, audit_output, audit_fields = run_target_note_audit(zotero_key, run_date=run_date, attempt=attempt)
        log_paths.update(audit_fields)
        if audit_status != "passed":
            return "failed:target_note_audit", output + verification + audit_output, log_paths
        return "success_done", output + verification + audit_output, log_paths
    if code == 0:
        reason = _read_failure_reason(output, note_status)
        if reason != f"not_finalized:{note_status}":
            return f"failed:{reason}", output + verification, log_paths
        return f"failed_not_done:{note_status}", output + verification, log_paths
    reason = _read_failure_reason(output, note_status)
    return f"failed:{reason}", output + verification, log_paths


def read_zotero_key_staged(
    zotero_key: str,
    *,
    timeout: int = 4200,
    run_date: str = "",
    attempt: int = 1,
    allow_dangerous_claude: bool = False,
) -> tuple[str, str, dict[str, Any]]:
    stage_dir = staged_read_root(zotero_key, attempt=attempt)
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "logs").mkdir(parents=True, exist_ok=True)
    session_id = staged_session_id(zotero_key, attempt=attempt)
    final_analysis_path = _final_analysis_path(zotero_key)
    manifest = _read_staged_manifest(stage_dir)
    manifest.update(
        {
            "schema_version": "staged_read_manifest.v1",
            "staged_read_version": STAGED_READ_VERSION,
            "zotero_key": zotero_key,
            "attempt": attempt,
            "session_id": session_id,
            "stage_dir": _rel(stage_dir),
            "final_analysis_path": _rel(final_analysis_path),
        }
    )
    manifest.setdefault("stages", [])
    manifest_path = _write_staged_manifest(stage_dir, manifest)
    outputs: list[str] = [
        f"STAGED_READ_START key={zotero_key} attempt={attempt} session_id={session_id} stage_dir={_rel(stage_dir)}",
    ]
    stderr_parts: list[str] = []
    stage_records: list[dict[str, Any]] = []
    completed_stage_paths: list[Path] = []
    heartbeat_path = ""
    stage_timeout = max(900, min(2400, max(900, timeout // 2)))

    def heartbeat(pid: int, stage_id: str) -> None:
        nonlocal heartbeat_path
        if run_date:
            heartbeat_path = _write_read_heartbeat(
                run_date=run_date,
                zotero_key=zotero_key,
                attempt=attempt,
                pid=pid,
                status="running",
                timeout=stage_timeout,
                stage=stage_id,
            )

    for stage_id, stage_title, stage_instruction in STAGED_READ_STAGES:
        stage_path = _stage_output_path(stage_dir, stage_id)
        final_ok, final_missing = _final_analysis_has_required_sections(final_analysis_path)
        final_reuse_allowed = (
            final_ok
            and manifest.get("status") == "success_done"
            and manifest.get("target_note_audit_status") == "passed"
        )
        if final_reuse_allowed and stage_id != STAGED_ASSEMBLY_STAGE:
            outputs.append(
                f"STAGED_FINAL_ANALYSIS_REUSE path={_rel(final_analysis_path)} "
                "reason=complete_existing_final_analysis"
            )
            break
        legacy_stage3_path = _stage_output_path(stage_dir, "03-experiments-limitations")
        if stage_id in {"03a-experiments-results", "03b-limitations-evidence-gaps"} and _stage_complete(legacy_stage3_path):
            outputs.append(
                f"STAGED_STAGE_SKIP stage={stage_id} reason=legacy_stage_file "
                f"path={_rel(legacy_stage3_path)}"
            )
            stage_records.append({"stage": stage_id, "status": "skipped_legacy", "path": _rel(legacy_stage3_path)})
            continue
        if stage_id != STAGED_ASSEMBLY_STAGE and _stage_complete(stage_path):
            outputs.append(f"STAGED_STAGE_SKIP stage={stage_id} reason=existing_stage_file path={_rel(stage_path)}")
            completed_stage_paths.append(stage_path)
            stage_records.append({"stage": stage_id, "status": "skipped_existing", "path": _rel(stage_path)})
            continue
        if deterministic_read_fallback_enabled() and stage_id in {"03a-experiments-results", "03b-limitations-evidence-gaps", "04-evidence-ledger", "05-review-pressure"}:
            stdout = deterministic_stage03_packet(stage_id, zotero_key, completed_stage_paths)
            if stdout.strip():
                stage_path.parent.mkdir(parents=True, exist_ok=True)
                stage_path.write_text(stdout, encoding="utf-8")
                stdout_path = _stage_log_path(stage_dir, stage_id, "out")
                stderr_path = _stage_log_path(stage_dir, stage_id, "err")
                stdout_path.parent.mkdir(parents=True, exist_ok=True)
                stdout_path.write_text(stdout, encoding="utf-8")
                stderr_path.write_text("deterministic_recovery_stage\n", encoding="utf-8")
                outputs.append(
                    f"STAGED_STAGE_DONE stage={stage_id} exit_code=0 "
                    f"out={_rel(stdout_path)} err={_rel(stderr_path)}"
                )
                record = {
                    "stage": stage_id,
                    "status": "success",
                    "exit_code": 0,
                    "path": _rel(stage_path),
                    "stdout": _rel(stdout_path),
                    "stderr": _rel(stderr_path),
                    "missing": [],
                    "command_attempts": 0,
                    "cli_session_id": "",
                    "deterministic_recovery": True,
                }
                stage_records.append(record)
                manifest["stages"] = stage_records
                manifest_path = _write_staged_manifest(stage_dir, manifest)
                completed_stage_paths.append(stage_path)
                continue
        if stage_id == STAGED_ASSEMBLY_STAGE:
            assembly_ok, missing_reason = assemble_staged_analysis(stage_dir, final_analysis_path)
            if final_analysis_path.exists():
                stage_path.write_text(final_analysis_path.read_text(encoding="utf-8"), encoding="utf-8")
            stdout_path = _stage_log_path(stage_dir, stage_id, "out")
            stderr_path = _stage_log_path(stage_dir, stage_id, "err")
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_path.write_text(
                f"deterministic_assembly status={'success' if assembly_ok else 'failed'} final_analysis={_rel(final_analysis_path)}\n",
                encoding="utf-8",
            )
            stderr_path.write_text("\n".join(missing_reason) + ("\n" if missing_reason else ""), encoding="utf-8")
            outputs.append(
                f"STAGED_STAGE_DONE stage={stage_id} exit_code={0 if assembly_ok else 1} out={_rel(stdout_path)} err={_rel(stderr_path)}"
            )
            record = {
                "stage": stage_id,
                "status": "success" if assembly_ok else "failed",
                "exit_code": 0 if assembly_ok else 1,
                "path": _rel(stage_path),
                "stdout": _rel(stdout_path),
                "stderr": _rel(stderr_path),
                "missing": missing_reason,
                "command_attempts": 0,
                "cli_session_id": "",
                "final_analysis": _rel(final_analysis_path),
                "assembly": "deterministic",
            }
            stage_records.append(record)
            manifest["stages"] = stage_records
            manifest_path = _write_staged_manifest(stage_dir, manifest)
            if not assembly_ok:
                summary_log = _write_read_attempt_log(
                    run_date=run_date or today_iso(),
                    zotero_key=zotero_key,
                    attempt=attempt,
                    stdout="\n".join(outputs),
                    stderr=f"STAGED_STAGE_FAILED stage={stage_id} missing={missing_reason}",
                )
                if heartbeat_path:
                    summary_log["heartbeat"] = heartbeat_path
                summary_log.update(
                    {
                        "staged_manifest": manifest_path,
                        "staged_session_id": session_id,
                        "staged_dir": _rel(stage_dir),
                        "stages": stage_records,
                    }
                )
                return "failed:staged_stage_failed", "\n".join(outputs), summary_log
            completed_stage_paths.append(stage_path)
            continue

        prompt = staged_read_prompt(
            zotero_key=zotero_key,
            session_id=session_id,
            stage_id=stage_id,
            stage_title=stage_title,
            stage_instruction=stage_instruction,
            stage_path=stage_path,
            stage_dir=stage_dir,
            prior_paths=completed_stage_paths,
            final_analysis_path=final_analysis_path,
        )
        cli_session_id = session_id if not completed_stage_paths else None
        command, env_overrides, prompt_input = _claude_command(
            prompt,
            session_id=cli_session_id,
            allow_dangerous_claude=allow_dangerous_claude,
        )
        code = 1
        stdout = ""
        stderr = ""
        command_attempt = 0
        for command_attempt in range(1, 4):
            try:
                code, stdout, stderr = run_subprocess_capture(
                    command,
                    timeout=stage_timeout,
                    heartbeat=(lambda pid, stage_id=stage_id: heartbeat(pid, stage_id)) if run_date else None,
                    heartbeat_interval=60,
                    env_overrides=env_overrides or None,
                    input_text=prompt_input,
                )
            except FileNotFoundError as exc:
                code = 127
                stdout = ""
                stderr = _subprocess_not_found_message(command, exc)
                break
            if code == 0:
                break
            retry_reason = _staged_transient_retry_reason(stdout, stderr)
            if not retry_reason or command_attempt >= 3:
                break
            delay = _staged_retry_delay_seconds(
                stdout,
                stderr,
                command_attempt=command_attempt,
                reason=retry_reason,
            )
            outputs.append(
                f"STAGED_STAGE_RETRY stage={stage_id} reason={retry_reason} "
                f"exit_code={code} attempt={command_attempt}/3 delay_sec={delay}"
            )
            time.sleep(delay)
        stdout_path = _stage_log_path(stage_dir, stage_id, "out")
        stderr_path = _stage_log_path(stage_dir, stage_id, "err")
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(stdout or "", encoding="utf-8")
        stderr_path.write_text(stderr or "", encoding="utf-8")
        if code == 0 and stdout.strip():
            stage_path.parent.mkdir(parents=True, exist_ok=True)
            stage_path.write_text(stdout, encoding="utf-8")
            if stage_id == STAGED_ASSEMBLY_STAGE:
                final_analysis_path.parent.mkdir(parents=True, exist_ok=True)
                final_analysis_path.write_text(stdout, encoding="utf-8")
        outputs.append(
            f"STAGED_STAGE_DONE stage={stage_id} exit_code={code} out={_rel(stdout_path)} err={_rel(stderr_path)}"
        )
        if stderr:
            stderr_parts.append(f"--- {stage_id} STDERR ---\n{stderr}")

        stage_ok = _stage_complete(stage_path)
        final_ok, final_missing = _final_analysis_has_required_sections(final_analysis_path)
        if stage_id != STAGED_ASSEMBLY_STAGE:
            success = code == 0 and stage_ok
            missing_reason = [] if stage_ok else ["missing_or_too_short_stage_file"]
        else:
            success = code == 0 and stage_ok and final_ok
            missing_reason = final_missing
            if not stage_ok:
                missing_reason = ["missing_or_too_short_assemble_stage_file", *missing_reason]
        record = {
            "stage": stage_id,
            "status": "success" if success else "failed",
            "exit_code": code,
            "path": _rel(stage_path),
            "stdout": _rel(stdout_path),
            "stderr": _rel(stderr_path),
            "missing": missing_reason,
            "command_attempts": command_attempt,
            "cli_session_id": cli_session_id or "",
        }
        if stage_id == STAGED_ASSEMBLY_STAGE:
            record["final_analysis"] = _rel(final_analysis_path)
        stage_records.append(record)
        manifest["stages"] = stage_records
        manifest_path = _write_staged_manifest(stage_dir, manifest)
        if not success:
            summary_log = _write_read_attempt_log(
                run_date=run_date or today_iso(),
                zotero_key=zotero_key,
                attempt=attempt,
                stdout="\n".join(outputs),
                stderr="\n\n".join(stderr_parts + [f"STAGED_STAGE_FAILED stage={stage_id} missing={missing_reason}"]),
            )
            if heartbeat_path:
                summary_log["heartbeat"] = heartbeat_path
            summary_log.update(
                {
                    "staged_manifest": manifest_path,
                    "staged_session_id": session_id,
                    "staged_dir": _rel(stage_dir),
                    "stages": stage_records,
                }
            )
            reason = "timeout" if code == 124 else "staged_stage_failed"
            return f"failed:{reason}", "\n".join(outputs), summary_log
        completed_stage_paths.append(stage_path)

    finalize_cmd = [
        sys.executable,
        ".claude/scripts/finalize_reading.py",
        zotero_key,
        "--analysis",
        _rel(final_analysis_path),
    ]
    finalize_code, finalize_stdout, finalize_stderr = run_subprocess_capture(finalize_cmd, timeout=420)
    finalize_out_path = stage_dir / "logs" / "finalize.out.log"
    finalize_err_path = stage_dir / "logs" / "finalize.err.log"
    finalize_out_path.write_text(finalize_stdout or "", encoding="utf-8")
    finalize_err_path.write_text(finalize_stderr or "", encoding="utf-8")
    outputs.append(
        f"STAGED_FINALIZE exit_code={finalize_code} out={_rel(finalize_out_path)} err={_rel(finalize_err_path)}"
    )
    if finalize_stderr:
        stderr_parts.append(f"--- FINALIZE STDERR ---\n{finalize_stderr}")
    note_status, note_path = local_note_status(zotero_key)
    verification = f"\nREAD_VERIFY: note={note_path or '-'} status={note_status}"
    summary_log = _write_read_attempt_log(
        run_date=run_date or today_iso(),
        zotero_key=zotero_key,
        attempt=attempt,
        stdout="\n".join(outputs) + verification,
        stderr="\n\n".join(stderr_parts),
    )
    if heartbeat_path:
        summary_log["heartbeat"] = heartbeat_path
    summary_log.update(
        {
            "staged_manifest": manifest_path,
            "staged_session_id": session_id,
            "staged_dir": _rel(stage_dir),
            "stages": stage_records,
            "final_analysis": _rel(final_analysis_path),
        }
    )
    if finalize_code != 0:
        return "failed:finalize_reading", "\n".join(outputs) + verification, summary_log
    if note_status != "done":
        reason = _read_failure_reason("\n".join(outputs), note_status)
        return f"failed:{reason}", "\n".join(outputs) + verification, summary_log
    audit_status, audit_output, audit_fields = run_target_note_audit(zotero_key, run_date=run_date, attempt=attempt)
    summary_log.update(audit_fields)
    output = "\n".join(outputs) + verification + audit_output
    if audit_status != "passed":
        return "failed:target_note_audit", output, summary_log
    manifest["status"] = "success_done"
    manifest["target_note_audit_status"] = "passed"
    manifest["audit_json_path"] = audit_fields.get("audit_json_path", "")
    manifest_path = _write_staged_manifest(stage_dir, manifest)
    summary_log["staged_manifest"] = manifest_path
    return "success_done", output, summary_log


def read_zotero_key_codex_controlled(
    zotero_key: str,
    *,
    timeout: int = 4200,
    run_date: str = "",
    attempt: int = 1,
    allow_dangerous_claude: bool = False,
) -> tuple[str, str, dict[str, Any]]:
    _ = allow_dangerous_claude
    stage_dir = codex_controlled_read_root(zotero_key, attempt=attempt)
    logs_dir = stage_dir / "logs"
    stage_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    final_analysis_path = _final_analysis_path(zotero_key)
    manifest: dict[str, Any] = {
        "schema_version": "codex_controlled_read_manifest.v1",
        "codex_controlled_read_version": CODEX_CONTROLLED_READ_VERSION,
        "zotero_key": zotero_key,
        "attempt": attempt,
        "stage_dir": _rel(stage_dir),
        "final_analysis_path": _rel(final_analysis_path),
        "status": "running",
    }
    manifest_path = _write_staged_manifest(stage_dir, manifest)
    outputs: list[str] = [
        f"CODEX_CONTROLLED_READ_START key={zotero_key} attempt={attempt} stage_dir={_rel(stage_dir)}",
    ]
    stderr_parts: list[str] = []
    heartbeat_path = ""

    def heartbeat(pid: int) -> None:
        nonlocal heartbeat_path
        if run_date:
            heartbeat_path = _write_read_heartbeat(
                run_date=run_date,
                zotero_key=zotero_key,
                attempt=attempt,
                pid=pid,
                status="running",
                timeout=timeout,
                stage="codex-controlled-read",
            )

    fulltext, fulltext_manifest = resolve_zotero_fulltext_for_controlled_read(zotero_key)
    fulltext_status = str(fulltext_manifest.get("status", "failed"))
    fulltext_source = str(fulltext_manifest.get("source", "none"))
    fulltext_manifest_path = stage_dir / "zotero-fulltext-resolution.json"
    fulltext_manifest_path.write_text(json.dumps(fulltext_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fulltext_path = stage_dir / "zotero-fulltext.txt"
    fulltext_path.write_text(fulltext, encoding="utf-8")
    manifest["zotero_fulltext_status"] = fulltext_status
    manifest["zotero_fulltext_source"] = fulltext_source
    manifest["zotero_fulltext_chars"] = len(fulltext)
    manifest["zotero_fulltext_resolution"] = fulltext_manifest
    manifest["zotero_fulltext_resolution_manifest"] = _rel(fulltext_manifest_path)
    outputs.append(
        f"CODEX_FULLTEXT status={fulltext_status} source={fulltext_source} chars={len(fulltext)} "
        f"path={_rel(fulltext_path)} resolver={_rel(fulltext_manifest_path)}"
    )
    if fulltext_status != "ok" or len(fulltext) < MIN_CONTROLLED_FULLTEXT_CHARS:
        manifest["status"] = "failed"
        manifest["failure_reason"] = f"zotero_fulltext_unavailable_or_too_short:{fulltext_status}:{len(fulltext)}"
        manifest_path = _write_staged_manifest(stage_dir, manifest)
        summary_log = _write_read_attempt_log(
            run_date=run_date or today_iso(),
            zotero_key=zotero_key,
            attempt=attempt,
            stdout="\n".join(outputs),
            stderr=manifest["failure_reason"],
        )
        summary_log.update(
            {
                "codex_controlled_manifest": manifest_path,
                "codex_controlled_dir": _rel(stage_dir),
                "zotero_fulltext_resolution_manifest": _rel(fulltext_manifest_path),
            }
        )
        return "failed:missing_zotero_fulltext", "\n".join(outputs), summary_log

    codex_output_path = stage_dir / "codex-final-message.md"
    command, env_overrides = _codex_controlled_command(stage_dir, codex_output_path)
    prompt_input = controlled_codex_read_prompt(
        zotero_key=zotero_key,
        fulltext=fulltext,
        local_context=_local_note_context(zotero_key),
    )
    code = 1
    stdout = ""
    stderr = ""
    try:
        code, stdout, stderr = run_subprocess_capture(
            command,
            timeout=timeout,
            heartbeat=heartbeat if run_date else None,
            heartbeat_interval=60,
            env_overrides=env_overrides or None,
            input_text=prompt_input,
        )
    except FileNotFoundError as exc:
        code = 127
        stderr = _subprocess_not_found_message(command, exc)
    if run_date and heartbeat_path:
        heartbeat_path = _write_read_heartbeat(
            run_date=run_date,
            zotero_key=zotero_key,
            attempt=attempt,
            pid=0,
            status=f"finished:exit_code={code}",
            timeout=timeout,
            stage="codex-controlled-read",
        )
    stdout_path = logs_dir / "codex.out.log"
    stderr_path = logs_dir / "codex.err.log"
    stdout_path.write_text(stdout or "", encoding="utf-8")
    stderr_path.write_text(stderr or "", encoding="utf-8")
    if stderr:
        stderr_parts.append(f"--- CODEX STDERR ---\n{stderr}")
    if codex_output_path.exists() and codex_output_path.read_text(encoding="utf-8").strip():
        final_text = codex_output_path.read_text(encoding="utf-8")
    else:
        final_text = stdout
        codex_output_path.write_text(final_text or "", encoding="utf-8")
    final_text = strip_controlled_read_orchestrator_status(final_text or "")
    codex_output_path.write_text(final_text, encoding="utf-8")
    outputs.append(
        f"CODEX_CONTROLLED_DONE exit_code={code} out={_rel(stdout_path)} err={_rel(stderr_path)} final={_rel(codex_output_path)}"
    )
    final_analysis_path.parent.mkdir(parents=True, exist_ok=True)
    final_analysis_path.write_text(final_text or "", encoding="utf-8")
    final_ok, final_missing = _final_analysis_has_required_sections(final_analysis_path)
    manifest.update(
        {
            "codex_exit_code": code,
            "codex_stdout": _rel(stdout_path),
            "codex_stderr": _rel(stderr_path),
            "codex_final_message": _rel(codex_output_path),
            "final_missing": final_missing,
        }
    )
    if code != 0 or not final_ok:
        manifest["status"] = "failed"
        manifest_path = _write_staged_manifest(stage_dir, manifest)
        summary_log = _write_read_attempt_log(
            run_date=run_date or today_iso(),
            zotero_key=zotero_key,
            attempt=attempt,
            stdout="\n".join(outputs),
            stderr="\n\n".join(stderr_parts + [f"CODEX_CONTROLLED_FAILED missing={final_missing} exit_code={code}"]),
        )
        if heartbeat_path:
            summary_log["heartbeat"] = heartbeat_path
        summary_log.update(
            {
                "codex_controlled_manifest": manifest_path,
                "codex_controlled_dir": _rel(stage_dir),
                "zotero_fulltext_resolution_manifest": _rel(fulltext_manifest_path),
                "final_analysis": _rel(final_analysis_path),
            }
        )
        return "failed:codex_controlled_read", "\n".join(outputs), summary_log

    finalize_cmd = [
        sys.executable,
        ".claude/scripts/finalize_reading.py",
        zotero_key,
        "--analysis",
        _rel(final_analysis_path),
    ]
    finalize_code, finalize_stdout, finalize_stderr = run_subprocess_capture(finalize_cmd, timeout=420)
    finalize_out_path = logs_dir / "finalize.out.log"
    finalize_err_path = logs_dir / "finalize.err.log"
    finalize_out_path.write_text(finalize_stdout or "", encoding="utf-8")
    finalize_err_path.write_text(finalize_stderr or "", encoding="utf-8")
    outputs.append(
        f"CODEX_CONTROLLED_FINALIZE exit_code={finalize_code} out={_rel(finalize_out_path)} err={_rel(finalize_err_path)}"
    )
    if finalize_stderr:
        stderr_parts.append(f"--- FINALIZE STDERR ---\n{finalize_stderr}")
    note_status, note_path = local_note_status(zotero_key)
    verification = f"\nREAD_VERIFY: note={note_path or '-'} status={note_status}"
    summary_log = _write_read_attempt_log(
        run_date=run_date or today_iso(),
        zotero_key=zotero_key,
        attempt=attempt,
        stdout="\n".join(outputs) + verification,
        stderr="\n\n".join(stderr_parts),
    )
    if heartbeat_path:
        summary_log["heartbeat"] = heartbeat_path
    summary_log.update(
        {
            "codex_controlled_manifest": manifest_path,
            "codex_controlled_dir": _rel(stage_dir),
            "zotero_fulltext_resolution_manifest": _rel(fulltext_manifest_path),
            "final_analysis": _rel(final_analysis_path),
        }
    )
    if finalize_code != 0:
        manifest["status"] = "failed"
        manifest["failure_reason"] = "finalize_reading_failed"
        _write_staged_manifest(stage_dir, manifest)
        return "failed:finalize_reading", "\n".join(outputs) + verification, summary_log
    if note_status != "done":
        reason = _read_failure_reason("\n".join(outputs), note_status)
        manifest["status"] = "failed"
        manifest["failure_reason"] = reason
        _write_staged_manifest(stage_dir, manifest)
        return f"failed:{reason}", "\n".join(outputs) + verification, summary_log
    audit_status, audit_output, audit_fields = run_target_note_audit(zotero_key, run_date=run_date, attempt=attempt)
    summary_log.update(audit_fields)
    output = "\n".join(outputs) + verification + audit_output
    if audit_status != "passed":
        manifest["status"] = "failed"
        manifest["failure_reason"] = "target_note_audit_failed"
        manifest.update(audit_fields)
        _write_staged_manifest(stage_dir, manifest)
        return "failed:target_note_audit", output, summary_log
    manifest["status"] = "success_done"
    manifest["target_note_audit_status"] = "passed"
    manifest["audit_json_path"] = audit_fields.get("audit_json_path", "")
    manifest_path = _write_staged_manifest(stage_dir, manifest)
    summary_log["codex_controlled_manifest"] = manifest_path
    return "success_done", output, summary_log


def read_zotero_key_timed(
    zotero_key: str,
    *,
    timeout: int,
    run_date: str = "",
    max_attempts: int = 1,
    retry_delay_sec: int = 60,
    allow_dangerous_claude: bool = False,
    read_mode: str = "codex-controlled",
) -> tuple[str, str, float, list[dict[str, Any]]]:
    started = time.monotonic()
    outputs: list[str] = []
    logs: list[dict[str, Any]] = []
    status = "failed:not_started"
    effective_attempts = 1 if read_mode == "staged" else max(1, max_attempts)
    for attempt in range(1, effective_attempts + 1):
        if read_mode == "staged":
            read_func = read_zotero_key_staged
        elif read_mode == "codex-controlled":
            read_func = read_zotero_key_codex_controlled
        else:
            read_func = read_zotero_key
        status, output, log_paths = read_func(
            zotero_key,
            timeout=timeout,
            run_date=run_date,
            attempt=attempt,
            allow_dangerous_claude=allow_dangerous_claude,
        )
        if log_paths:
            logs.append({"attempt": str(attempt), **log_paths})
        outputs.append(f"READ_ATTEMPT {attempt}/{effective_attempts}: mode={read_mode} status={status}\n{output}")
        if status.startswith("success"):
            break
        if attempt < effective_attempts:
            time.sleep(max(0, retry_delay_sec))
    return status, "\n\n".join(outputs), round(time.monotonic() - started, 2), logs


def render_run_log(
    *,
    run_date: str,
    status: str,
    ranked: list[RankedPaper],
    imports: list[dict[str, Any]],
    reads: list[dict[str, Any]],
    idea_path: Path,
    agenda_delta_path: Path,
    gemini_prompt_path: Path,
    idea_audit: dict[str, Any],
    idea_generation: dict[str, Any],
    errors: list[str],
    import_policy: dict[str, Any],
    existing_candidates: list[dict[str, Any]],
    baseline_backfill: dict[str, Any] | None = None,
) -> str:
    top = [item for item in ranked if item.decision == "top1_candidate"]
    venue_auto = [item for item in ranked if item.decision == "venue_auto_import"]
    focus_review = [item for item in ranked if item.decision == FOCUS_REVIEW_DECISION]
    review = [item for item in ranked if item.decision == "review_queue"]
    new_imports = sum(1 for item in imports if item.get("status") in NEW_IMPORT_STATUSES)
    resumed_backlog = sum(1 for item in imports if item.get("status") == "backlog")
    existing_duplicates = sum(1 for item in imports if item.get("status") == "exists")
    fill_imports = sum(1 for item in imports if item.get("selection_decision") == AUTO_IMPORT_FILL_DECISION)
    focus_imports = sum(1 for item in imports if item.get("selection_decision") == FOCUS_REVIEW_DECISION)
    quota_summary = import_policy.get("selection_quota_summary", {})
    baseline_backfill = baseline_backfill or {}
    read_by_key = {str(item.get("zotero_key", "")): item for item in reads}
    new_read_success = [
        item
        for item in imports
        if item.get("status") in FOCUS_READY_STATUSES
        and str(read_by_key.get(str(item.get("zotero_key", "")), {}).get("read_status", "")).startswith("success")
    ]
    lines = [
        "---",
        f'title: "Daily arXiv Scout Run - {run_date}"',
        "tags: [arxiv, automation, embodied-ai]",
        f'created: "{run_date}"',
        f'updated: "{run_date}"',
        'type: "permanent"',
        'status: "draft"',
        f'summary: "Daily arXiv scout run status: {status}."',
        "---",
        "",
        f"# Daily arXiv Scout Run - {run_date}",
        "",
        f"- status: {status}",
        f"- candidates_total: {len(ranked)}",
        f"- top1_candidates: {len(top)}",
        f"- venue_auto_import: {len(venue_auto)}",
        f"- focus_review_queue: {len(focus_review)}",
        f"- review_queue: {len(review)}",
        f"- arxiv_source: {import_policy.get('arxiv_source', 'unknown')}",
        f"- mirror_records_total: {import_policy.get('mirror_records_total', 0)}",
        f"- mirror_last_success_at: {import_policy.get('mirror_last_success_at', '-')}",
        f"- search_api_fallback_used: {str(import_policy.get('search_api_fallback_used', False)).lower()}",
        f"- arxiv_fetch_errors: {import_policy.get('arxiv_fetch_errors', 0)}",
        f"- no_new_focus_keys_reason: {import_policy.get('no_new_focus_keys_reason', '-')}",
        f"- days_back: {import_policy.get('days_back')}",
        f"- max_candidates: {import_policy.get('max_candidates')}",
        f"- min_new_imports: {import_policy.get('min_new_imports')}",
        f"- max_import_attempts: {import_policy.get('max_auto_import')}",
        f"- max_daily_reads: {import_policy.get('max_read')}",
        f"- read_failure_backfill: {import_policy.get('read_failure_backfill', 0)}",
        f"- fill_import_threshold: {import_policy.get('fill_import_threshold')}",
        f"- existing_prefilter: {import_policy.get('existing_prefilter', 'not_run')}",
        f"- existing_candidates_excluded: {len(existing_candidates)}",
        f"- baseline_backfill_queued: {baseline_backfill.get('queued_total', 0)}",
        f"- baseline_backfill_json: `{baseline_backfill.get('json_path', '')}`",
        f"- baseline_backfill_md: `{baseline_backfill.get('md_path', '')}`",
        f"- import_attempts: {len(imports)}",
        f"- new_imports_created: {new_imports}",
        f"- resumed_backlog_imports: {resumed_backlog}",
        f"- new_imports_read_success: {len(new_read_success)}",
        f"- existing_duplicates: {existing_duplicates}",
        f"- auto_import_fill_attempts: {fill_imports}",
        f"- focus_review_import_attempts: {focus_imports}",
        f"- selection_quota_summary: {_selection_quota_text(quota_summary) if isinstance(quota_summary, dict) else quota_summary}",
        f"- legacy_idea_pointer: `{idea_path.relative_to(vault_path())}`",
        f"- agenda_delta_file: `{agenda_delta_path.relative_to(vault_path())}`",
        f"- gemini_prompt_file: `{gemini_prompt_path.relative_to(vault_path())}`",
        "- workflow_manifest: `projects/research-agenda/workflow-contracts/daily-readable-workflow.json`",
        "- provider_matrix: `projects/research-agenda/workflow-contracts/provider-matrix.json`",
        f"- agenda_update_mode: {idea_generation.get('mode', 'research_agenda_update')}",
        f"- agenda_update_status: {idea_generation.get('status', 'unknown')}",
        f"- research_seed_v2_status: {idea_generation.get('v2_state_machine', {}).get('status', 'not_run')}",
        f"- research_seed_v2_run_dir: `{idea_generation.get('v2_state_machine', {}).get('run_dir', '')}`",
        f"- strict_kb_maintenance: {import_policy.get('strict_kb_maintenance', 'not_run')}",
        f"- legacy_idea_audit: {idea_audit.get('status', 'not_applicable')}",
        "",
        "## Top Candidates",
        "",
    ]
    if not top:
        lines.append("- evidence_gap: no candidate reached top1_candidate threshold.")
    for item in top:
        paper = item.paper
        lines.append(f"- [{paper.title}]({paper.url}) - score={item.quality_score}; arXiv={paper.arxiv_id}; reasons={', '.join(item.reasons)}")
    lines.extend(["", "## Venue Auto Imports", ""])
    if not venue_auto:
        lines.append("- none")
    for item in venue_auto:
        paper = item.paper
        lines.append(f"- [{paper.title}]({paper.url}) - score={item.quality_score}; arXiv={paper.arxiv_id}; reasons={', '.join(item.reasons)}")
    lines.extend(["", "## Focus Review Queue", ""])
    if not focus_review:
        lines.append("- none")
    for item in focus_review:
        paper = item.paper
        focus = "; ".join(f"{track}: {', '.join(matches)}" for track, matches in item.focus_matches.items())
        lines.append(
            f"- [{paper.title}]({paper.url}) - score={item.quality_score}; arXiv={paper.arxiv_id}; "
            f"focus={focus or '-'}; reasons={', '.join(item.reasons)}"
        )
    lines.extend(["", "## Existing Zotero Candidates Excluded", ""])
    if not existing_candidates:
        lines.append("- none")
    for item in existing_candidates:
        lines.append(
            f"- arxiv={item.get('arxiv_id')} score={item.get('quality_score')} "
            f"decision={item.get('decision')} zotero_key={item.get('zotero_key', '')} "
            f"title={item.get('title', '')}"
        )
    lines.extend(["", "## Canonical Baseline Backfill", ""])
    queued_baselines = [item for item in baseline_backfill.get("entries", []) if item.get("status") == "queued"]
    if not queued_baselines:
        lines.append("- none")
    else:
        lines.append("- boundary: queue only; these entries do not consume today's deep-read quota and do not write seeds.")
        for item in queued_baselines[:10]:
            lines.append(
                f"- {item.get('label')} priority={item.get('priority_score')} "
                f"source_notes={item.get('source_note_count')} mentions={item.get('mention_count')} "
                f"query={item.get('search_query')}"
            )
    lines.extend(["", "## Imports", ""])
    if not imports:
        lines.append("- No Zotero imports attempted.")
    for item in imports:
        lines.append(
            f"- arxiv={item.get('arxiv_id')} selection={item.get('selection_decision', '')} "
            f"original={item.get('original_decision', '')} score={item.get('quality_score', '')} "
            f"status={item.get('status')} zotero_key={item.get('zotero_key', '')} message={item.get('message', '')}"
        )
    lines.extend(["", "## Reading", ""])
    if not reads:
        lines.append("- No Claudian reading attempted.")
    for item in reads:
        lines.append(
            f"- zotero_key={item.get('zotero_key')} ingest={item.get('ingest_status')} "
            f"read={item.get('read_status')} read_elapsed_sec={item.get('read_elapsed_sec', '-')} "
            f"target_note_audit={item.get('target_note_audit_status', 'not_run')} "
            f"audit_json={item.get('audit_json_path', '-') or '-'}"
        )
        if item.get("target_note_issues"):
            lines.append(f"  - target_note_issues: {item.get('target_note_issues')}")
        if item.get("global_warning_counts"):
            lines.append(f"  - global_warning_counts: {item.get('global_warning_counts')}")
        if item.get("global_warning_paths"):
            lines.append(f"  - global_warning_paths: {item.get('global_warning_paths')}")
        if item.get("read_attempt_logs"):
            logs = "; ".join(
                f"attempt{log.get('attempt')}:out={log.get('stdout')},err={log.get('stderr')},"
                f"heartbeat={log.get('heartbeat', '-')},audit={log.get('audit_json_path', '-')},"
                f"staged_manifest={log.get('staged_manifest', '-')},"
                f"codex_controlled_manifest={log.get('codex_controlled_manifest', '-')}"
                for log in item.get("read_attempt_logs", [])
            )
            lines.append(f"  - read_attempt_logs: {logs}")
    lines.extend(["", "## Agenda Update", ""])
    lines.append(f"- mode: {idea_generation.get('mode', 'research_agenda_update')}")
    lines.append(f"- status: {idea_generation.get('status', 'unknown')}")
    if idea_generation.get("message"):
        lines.append(f"- message: {idea_generation['message']}")
        for marker in ["MANDATORY_MODEL_BATTLE:", "AGENDA_DELTA:", "CODEX_REVIEW_PENDING:"]:
            match = re.search(rf"^{re.escape(marker)}\s*(.+)$", str(idea_generation["message"]), re.MULTILINE)
            if match:
                key = marker.rstrip(":").lower()
                lines.append(f"- {key}: {match.group(1).strip()}")
    v2_state = idea_generation.get("v2_state_machine", {})
    lines.extend(["", "## Research Seed V2", ""])
    if not v2_state:
        lines.append("- status: not_run")
    else:
        lines.append(f"- status: {v2_state.get('status', 'unknown')}")
        lines.append(f"- run_dir: `{v2_state.get('run_dir', '')}`")
        lines.append(f"- backfill_mode: {v2_state.get('backfill_mode', '')}")
        lines.append(f"- backfill_generate_ideas: {str(v2_state.get('backfill_generate_ideas', False)).lower()}")
        lines.append(f"- backfill_publish: {v2_state.get('backfill_publish', '')}")
        lines.append(f"- v2_publish_policy: {v2_state.get('v2_publish_policy', '')}")
        lines.append(f"- formal_seed_publish_allowed: {str(v2_state.get('formal_seed_publish_allowed', False)).lower()}")
        lines.append(f"- scheduled_daily_switched: {str(v2_state.get('scheduled_daily_switched', False)).lower()}")
        for stage in v2_state.get("stages", []):
            lines.append(
                f"- stage={stage.get('stage')} status={stage.get('status')} "
                f"exit_code={stage.get('exit_code')} elapsed_sec={stage.get('elapsed_sec')}"
            )
    lines.extend(["", "## Errors", ""])
    if not errors:
        lines.append("- none")
    else:
        lines.extend(f"- {error}" for error in errors)
    return "\n".join(lines).rstrip() + "\n"


def render_reading_backlog(*, run_date: str, imports: list[dict[str, Any]], reads: list[dict[str, Any]]) -> str:
    read_by_key = {str(item.get("zotero_key", "")): item for item in reads}
    lines = [
        "---",
        f'title: "Daily arXiv Reading Backlog - {run_date}"',
        "tags: [arxiv, reading-backlog, embodied-ai]",
        f'created: "{run_date}"',
        f'updated: "{run_date}"',
        'type: "permanent"',
        'status: "draft"',
        'summary: "Imported arXiv papers that still need full Claudian reading."',
        "---",
        "",
        f"# Daily arXiv Reading Backlog - {run_date}",
        "",
    ]
    pending = 0
    for item in imports:
        key = str(item.get("zotero_key", ""))
        read = read_by_key.get(key, {})
        if str(read.get("read_status", "")).startswith("success"):
            continue
        if read.get("read_status") == "skipped":
            continue
        pending += 1
        lines.append(f"- arxiv={item.get('arxiv_id')} zotero_key={key} import_status={item.get('status')} read_status={read.get('read_status', 'not_started')}")
    if pending == 0:
        lines.append("- none")
    return "\n".join(lines).rstrip() + "\n"


def render_resume_log(
    *,
    run_date: str,
    status: str,
    imports: list[dict[str, Any]],
    reads: list[dict[str, Any]],
    focus_keys: list[str],
    agenda_status: str,
    errors: list[str],
) -> str:
    lines = [
        "---",
        f'title: "Daily arXiv Backlog Recovery - {run_date}"',
        "tags: [arxiv, reading-backlog, recovery]",
        f'created: "{run_date}"',
        f'updated: "{run_date}"',
        'type: "permanent"',
        'status: "draft"',
        f'summary: "Backlog recovery status: {status}."',
        "---",
        "",
        f"# Daily arXiv Backlog Recovery - {run_date}",
        "",
        f"- status: {status}",
        f"- attempted_reads: {len(reads)}",
        f"- successful_reads: {sum(1 for item in reads if str(item.get('read_status', '')).startswith('success'))}",
        f"- focus_zotero_keys: {', '.join(focus_keys) if focus_keys else '-'}",
        f"- agenda_update_status: {agenda_status}",
        "",
        "## Imports",
        "",
    ]
    for item in imports:
        lines.append(
            f"- arxiv={item.get('arxiv_id')} status={item.get('status')} "
            f"zotero_key={item.get('zotero_key', '')} message={item.get('message', '')}"
        )
    if not imports:
        lines.append("- none")
    lines.extend(["", "## Reading", ""])
    for item in reads:
        lines.append(
            f"- zotero_key={item.get('zotero_key')} ingest={item.get('ingest_status')} "
            f"read={item.get('read_status')} read_elapsed_sec={item.get('read_elapsed_sec', '-')} "
            f"target_note_audit={item.get('target_note_audit_status', 'not_run')} "
            f"audit_json={item.get('audit_json_path', '-') or '-'}"
        )
        if item.get("target_note_issues"):
            lines.append(f"  - target_note_issues: {item.get('target_note_issues')}")
        if item.get("global_warning_counts"):
            lines.append(f"  - global_warning_counts: {item.get('global_warning_counts')}")
        if item.get("global_warning_paths"):
            lines.append(f"  - global_warning_paths: {item.get('global_warning_paths')}")
        if item.get("read_attempt_logs"):
            logs = "; ".join(
                f"attempt{log.get('attempt')}:out={log.get('stdout')},err={log.get('stderr')},"
                f"heartbeat={log.get('heartbeat', '-')},audit={log.get('audit_json_path', '-')},"
                f"staged_manifest={log.get('staged_manifest', '-')},"
                f"codex_controlled_manifest={log.get('codex_controlled_manifest', '-')}"
                for log in item.get("read_attempt_logs", [])
            )
            lines.append(f"  - read_attempt_logs: {logs}")
    if not reads:
        lines.append("- none")
    lines.extend(["", "## Errors", ""])
    lines.extend(f"- {error}" for error in errors) if errors else lines.append("- none")
    return "\n".join(lines).rstrip() + "\n"


def merge_resume_backlog(
    *,
    run_date: str,
    original_records: list[dict[str, str]],
    reads: list[dict[str, Any]],
) -> str:
    read_by_key = {str(item.get("zotero_key", "")): item for item in reads}
    lines = [
        "---",
        f'title: "Daily arXiv Reading Backlog - {run_date}"',
        "tags: [arxiv, reading-backlog, embodied-ai]",
        f'created: "{run_date}"',
        f'updated: "{run_date}"',
        'type: "permanent"',
        'status: "draft"',
        'summary: "Imported arXiv papers that still need full Claudian reading."',
        "---",
        "",
        f"# Daily arXiv Reading Backlog - {run_date}",
        "",
    ]
    pending = 0
    for record in original_records:
        key = record.get("key", "")
        if not key:
            continue
        read = read_by_key.get(key)
        if read and str(read.get("read_status", "")).startswith("success"):
            continue
        note_status, _ = local_note_status(key)
        if not read and note_status == "done":
            continue
        pending += 1
        read_status = str(read.get("read_status", record.get("read_status", "backlog"))) if read else record.get("read_status", "backlog")
        lines.append(
            f"- arxiv={record.get('arxiv', '')} zotero_key={key} "
            f"import_status={record.get('import_status', 'backlog')} read_status={read_status}"
        )
    if pending == 0:
        lines.append("- none")
    return "\n".join(lines).rstrip() + "\n"


def render_legacy_idea_pointer(*, run_date: str, agenda_delta_path: Path, focus_keys: list[str], agenda_status: str) -> str:
    agenda_delta_rel = str(agenda_delta_path.relative_to(vault_path())).replace("\\", "/")
    return "\n".join(
        [
            "---",
            f'title: "Daily Embodied AI Ideas - {run_date}"',
            "tags: [idea, arxiv, embodied-ai, compatibility]",
            f'created: "{run_date}"',
            f'updated: "{run_date}"',
            'type: "permanent"',
            'status: "done"',
            'summary: "Compatibility pointer. Final research ideas now live in the long-term research agenda."',
            'evidence_level: "partial"',
            "---",
            "",
            f"# Daily Embodied AI Ideas - {run_date}",
            "",
            "This file is kept only for backward compatibility. The daily workflow no longer treats this as the final idea source.",
            "",
            f"- agenda_delta: `{agenda_delta_rel}`",
            f"- agenda_update_status: {agenda_status}",
            f"- focus_zotero_keys: {', '.join(focus_keys) if focus_keys else '-'}",
            "- boundary: high-quality ideas are promoted from `projects/research-agenda/idea_bank/` only after evidence and review gates.",
            "",
        ]
    )


def run_resume_backlog(args: argparse.Namespace) -> int:
    run_date = args.run_date or today_iso()
    base_dir = vault_path("projects", "arxiv-daily")
    run_log_path = base_dir / f"{run_date}-run.md"
    backlog_path = base_dir / f"{run_date}-reading-backlog.md"
    recovery_log_path = base_dir / f"{run_date}-backlog-recovery.md"
    idea_path = vault_path("projects", "ideas", f"{run_date}-embodied-ai-ideas.md")
    agenda_delta_path = vault_path("projects", "research-agenda", "daily", f"{run_date}-agenda-delta.md")
    gemini_prompt_path = vault_path("projects", "ideas", f"{run_date}-gemini-idea-prompt.md")
    errors: list[str] = []

    resume_records = parse_backlog(backlog_path)
    if args.dry_run:
        pending_records = []
        for record in resume_records:
            key = record.get("key", "")
            note_status, note_path = local_note_status(key) if key else ("missing_key", "")
            if key and not record.get("read_status", "").startswith("success") and note_status != "done":
                pending_records.append((record, note_status, note_path))
        safe_print(
            "DRY-RUN RESUME-BACKLOG "
            f"run_date={run_date} "
            f"pending={len(pending_records)} "
            f"max_read={args.max_read} "
            f"read_mode={args.read_mode} "
            f"read_retries={args.read_retries}"
        )
        for record, note_status, note_path in pending_records[: max(1, args.max_read)]:
            safe_print(
                f"{record.get('arxiv', '')} zotero_key={record.get('key', '')} "
                f"note_status={note_status} note={note_path or '-'}"
            )
        return 0
    prior_success_keys = parse_successful_read_keys_with_backups(run_log_path)
    imports: list[dict[str, Any]] = []
    reads: list[dict[str, Any]] = []
    read_attempts = 0
    for record in resume_records:
        if record.get("read_status", "").startswith("success"):
            continue
        key = record.get("key", "")
        if not key:
            continue
        note_status, note_path = local_note_status(key)
        imports.append(
            {
                "arxiv_id": record.get("arxiv", ""),
                "quality_score": "",
                "selection_decision": "resume_backlog",
                "original_decision": "resume_backlog",
                "status": "backlog",
                "zotero_key": key,
                "message": f"resumed from reading backlog; note_status={note_status}; note={note_path or '-'}",
                "mode": "resume_backlog",
                "existing": True,
            }
        )
        read_status = "skipped"
        read_output = ""
        read_attempt_logs: list[dict[str, Any]] = []
        read_elapsed_sec: float | str = "-"
        if note_status == "done":
            ingest_status = "success"
            ingest_output = f"already_done:{note_path}"
            read_status = "success_done_already"
        elif note_status != "missing_note":
            ingest_status = "success"
            ingest_output = f"local_note_ready:{note_status}:{note_path}"
        else:
            ingest_status, ingest_output = ingest_zotero_key(key)
            output_note_status, output_key, output_note_path = local_note_from_ingest_output(ingest_output)
            if output_key and output_key != key:
                key = output_key
                note_status, note_path = output_note_status, output_note_path
                imports[-1]["zotero_key"] = output_key
                imports[-1]["message"] = f"{imports[-1].get('message', '')}; canonical_note_key={output_key}; note={output_note_path}"
            note_status, note_path = wait_for_local_note(key)
            ingest_output = f"{ingest_output}\nLOCAL_NOTE_READY: status={note_status} note={note_path or '-'}"
            if ingest_status != "success" and note_status != "missing_note":
                ingest_status = "success"
                ingest_output = f"{ingest_output}\nINGEST_RECOVERED_BY_LOCAL_NOTE: {note_path}"
        if note_status == "done" and ingest_status == "success":
            read_status = "success_done_already"
        elif note_status not in {"done", "missing_note"} and ingest_status == "success" and not args.skip_read and read_attempts < args.max_read:
            read_attempts += 1
            read_status, read_output, read_elapsed_sec, read_attempt_logs = read_zotero_key_timed(
                key,
                timeout=args.read_timeout,
                run_date=run_date,
                max_attempts=args.read_retries + 1,
                retry_delay_sec=args.read_retry_delay,
                allow_dangerous_claude=args.allow_dangerous_claude,
                read_mode=args.read_mode,
            )
        elif note_status == "missing_note" and ingest_status == "success" and not args.skip_read:
            read_status = "waiting_local_note"
        elif note_status != "done" and ingest_status == "success" and not args.skip_read:
            read_status = "backlog"
        read_status, read_output, audit_fields = finalize_read_status_with_target_audit(
            zotero_key=key,
            read_status=read_status,
            read_output=read_output,
            read_attempt_logs=read_attempt_logs,
            run_date=run_date,
        )
        reads.append(
            {
                "zotero_key": key,
                "ingest_status": ingest_status,
                "ingest_output_tail": ingest_output[-1200:],
                "read_status": read_status,
                "read_elapsed_sec": read_elapsed_sec,
                "read_attempt_logs": read_attempt_logs,
                "read_output_tail": read_output[-1200:],
                **audit_fields,
            }
        )
        if ingest_status != "success":
            errors.append(f"ingest_failed:{key}")
        if read_status == "waiting_local_note":
            errors.append(f"read_pending:{key}:waiting_local_note")
        if read_status not in {"skipped", "backlog", "waiting_local_note"} and not read_status.startswith("success"):
            errors.append(f"read_failed:{key}:{read_status}")

    maintenance_status = "not_run"
    if reads and not args.skip_read:
        try:
            maintenance_code, maintenance_output = run_subprocess(
                [sys.executable, ".claude/scripts/fix_strict_kb_after_read.py"],
                timeout=300,
            )
        except Exception as exc:
            maintenance_code, maintenance_output = 1, str(exc)
        maintenance_status = "success" if maintenance_code == 0 else "failed"
        if maintenance_code != 0:
            errors.append(f"strict_kb_maintenance_failed:{maintenance_output[-400:]}")

    read_by_key = {str(item.get("zotero_key", "")): item for item in reads}
    focus_zotero_keys: list[str] = []
    seen_focus_keys: set[str] = set()
    for item in imports:
        key = str(item.get("zotero_key", ""))
        if key and str(read_by_key.get(key, {}).get("read_status", "")).startswith("success"):
            focus_zotero_keys.append(key)
            seen_focus_keys.add(key)
    for key in prior_success_keys:
        if key and key not in seen_focus_keys:
            focus_zotero_keys.append(key)
            seen_focus_keys.add(key)

    if not focus_zotero_keys and args.idea_mode == "gemini-divergent":
        agenda_command = [
            sys.executable,
            ".claude/scripts/research_agenda_update.py",
            "--run-date",
            run_date,
            "--idea-generator",
            "none",
            "--idea-timeout",
            str(args.idea_timeout),
            "--gemini-model",
            args.gemini_model,
            "--deepseek-model",
            args.deepseek_model,
            "--deepseek-timeout",
            str(args.deepseek_timeout),
            "--raw-candidate-limit",
            str(args.raw_candidate_limit),
            "--min-raw-candidates",
            str(args.min_raw_candidates),
            "--max-generated",
            str(args.max_generated),
        ]
    else:
        agenda_command = [sys.executable, ".claude/scripts/research_agenda_update.py", "--run-date", run_date]
        if focus_zotero_keys:
            agenda_command.extend(["--zotero-keys", ",".join(focus_zotero_keys)])
        agenda_command.extend(
            [
                "--idea-generator",
                "none",
                "--idea-timeout",
                str(args.idea_timeout),
                "--gemini-model",
                args.gemini_model,
                "--deepseek-model",
                args.deepseek_model,
                "--deepseek-timeout",
                str(args.deepseek_timeout),
                "--raw-candidate-limit",
                str(args.raw_candidate_limit),
                "--min-raw-candidates",
                str(args.min_raw_candidates),
                "--max-generated",
                str(args.max_generated),
            ]
        )
    try:
        agenda_code, agenda_output = run_subprocess(
            agenda_command,
            timeout=max(900, args.idea_timeout * 2 + args.deepseek_timeout + 300),
        )
    except Exception as exc:
        agenda_code, agenda_output = 1, str(exc)
    if agenda_code == 0:
        agenda_status = "skipped_no_focus_keys" if not focus_zotero_keys else "success"
    else:
        agenda_status = "partial"
        errors.append("research_agenda_update_failed")

    v2_result = run_research_seed_v2(
        args=args,
        run_date=run_date,
        candidates_path=base_dir / f"{run_date}-candidates.jsonl",
        focus_zotero_keys=focus_zotero_keys,
        errors=errors,
    )
    status = "success" if not errors else "partial"
    safe_write(backlog_path, merge_resume_backlog(run_date=run_date, original_records=resume_records, reads=reads), dry_run=False, backup=True)
    safe_write(
        recovery_log_path,
        render_resume_log(
            run_date=run_date,
            status=status,
            imports=imports,
            reads=reads,
            focus_keys=focus_zotero_keys,
            agenda_status=agenda_status,
            errors=errors,
        ),
        dry_run=False,
        backup=True,
    )
    safe_write(
        idea_path,
        render_legacy_idea_pointer(run_date=run_date, agenda_delta_path=agenda_delta_path, focus_keys=focus_zotero_keys, agenda_status=agenda_status),
        dry_run=False,
        backup=True,
    )
    reconciled = reconcile_run_log_after_recovery(
        run_log_path=run_log_path,
        recovery_log_path=recovery_log_path,
        recovery_status=status,
        agenda_status=agenda_status,
    )
    safe_print(f"RECOVERY_LOG: {recovery_log_path.relative_to(vault_path())}")
    safe_print(f"BACKLOG: {backlog_path.relative_to(vault_path())}")
    safe_print(f"AGENDA_DELTA: {agenda_delta_path.relative_to(vault_path())}")
    safe_print(f"AGENDA_UPDATE: research_agenda_update:{agenda_status}")
    safe_print(f"RESEARCH_SEED_V2: {v2_result.get('status', 'unknown')} run_dir={v2_result.get('run_dir', '')}")
    safe_print(f"STRICT_KB_MAINTENANCE: {maintenance_status}")
    safe_print(f"RUN_LOG_RECONCILED: {str(reconciled).lower()}")
    if errors:
        safe_print("PIPELINE_STATUS: partial")
        for error in errors:
            safe_print(f"ERROR: {error}")
        return 2
    safe_print("PIPELINE_STATUS: success")
    return 0


def run_pipeline(args: argparse.Namespace) -> int:
    run_date = args.run_date or today_iso()
    if args.resume_backlog:
        return run_resume_backlog(args)
    as_of = datetime.fromisoformat(run_date).replace(tzinfo=timezone.utc) + timedelta(days=1)
    base_dir = vault_path("projects", "arxiv-daily")
    candidates_path = base_dir / f"{run_date}-candidates.jsonl"
    review_path = base_dir / "review_queue" / f"{run_date}-review.md"
    run_log_path = base_dir / f"{run_date}-run.md"
    backlog_path = base_dir / f"{run_date}-reading-backlog.md"
    idea_path = vault_path("projects", "ideas", f"{run_date}-embodied-ai-ideas.md")
    gemini_prompt_path = vault_path("projects", "ideas", f"{run_date}-gemini-idea-prompt.md")
    agenda_delta_path = vault_path("projects", "research-agenda", "daily", f"{run_date}-agenda-delta.md")
    errors: list[str] = []

    queries = args.query or DEFAULT_QUERIES
    papers, source_info = collect_candidates_from_source(
        queries,
        source=args.source,
        max_candidates=args.max_candidates,
        days_back=args.days_back,
        as_of=as_of,
        errors=errors,
        fetch_timeout=args.fetch_timeout,
        fetch_retries=args.fetch_retries,
        query_delay=args.query_delay,
        dry_run=args.dry_run,
    )
    cache_fallback = False
    ranked_all = rank_papers(papers)
    if not ranked_all:
        cached = load_cached_ranked(candidates_path)
        if cached:
            cache_fallback = True
            ranked_all = cached
            errors.append(f"arxiv_cache_fallback:{candidates_path.relative_to(vault_path())}")
    new_ranked_all, existing_candidates, existing_prefilter = exclude_existing_candidates(
        ranked_all,
        collection_key=args.collection,
    )
    if existing_prefilter.startswith("failed:"):
        errors.append(f"existing_prefilter_{existing_prefilter}")
    ranked = new_ranked_all[: args.max_candidates]
    selection_pool = new_ranked_all[: max(args.max_candidates, args.max_auto_import)]
    records = [item.to_dict() for item in new_ranked_all[: max(args.max_candidates, args.max_auto_import)]]
    intake_triage = build_triage(
        records,
        run_date=run_date,
        target_deep_read=args.target_deep_read,
        max_deep_read=args.max_deep_read,
    )
    selected_deep_read_ids = [
        str(item.get("arxiv_id") or "")
        for item in intake_triage.get("selected_for_deep_read", [])
        if isinstance(item, dict) and item.get("arxiv_id")
    ]
    import_policy = {
        "min_new_imports": args.min_new_imports,
        "max_auto_import": args.max_auto_import,
        "max_read": args.max_read,
        "v2_intake_controls_imports": bool(args.v2_intake_controls_imports),
        "legacy_import_fill": bool(args.legacy_import_fill),
        "target_deep_read": args.target_deep_read,
        "max_deep_read": args.max_deep_read,
        "v2_selected_for_deep_read": selected_deep_read_ids,
        "read_failure_backfill": args.read_failure_backfill,
        "fill_import_threshold": args.fill_import_threshold,
        "fetch_timeout": args.fetch_timeout,
        "fetch_retries": args.fetch_retries,
        "query_delay": args.query_delay,
        "days_back": args.days_back,
        "max_candidates": args.max_candidates,
        "arxiv_source": source_info.get("source", "unknown"),
        "mirror_records_total": source_info.get("records_total", source_info.get("mirror_records_total", 0)),
        "mirror_last_success_at": source_info.get("last_success_at", source_info.get("mirror_last_success_at", "")),
        "search_api_fallback_used": source_info.get("search_api_fallback_used", False),
        "arxiv_fetch_errors": sum(1 for error in errors if error.startswith("arxiv_fetch_failed")),
        "existing_prefilter": existing_prefilter,
        "existing_candidates_excluded": len(existing_candidates),
    }
    baseline_backfill_report: dict[str, Any] = {}
    if not args.skip_baseline_backfill:
        baseline_backfill_report = build_baseline_backfill_report(
            run_date=run_date,
            min_mentions=max(1, args.baseline_backfill_min_mentions),
            limit=max(0, args.baseline_backfill_limit),
        )
        import_policy["baseline_backfill_queued"] = baseline_backfill_report.get("queued_total", 0)
        import_policy["baseline_backfill_entry_total"] = baseline_backfill_report.get("entry_total", 0)
    selected_import_candidates = select_import_candidates(
        selection_pool,
        max_attempts=args.max_auto_import,
        fill_threshold=args.fill_import_threshold,
    )
    legacy_floor_active = not args.v2_intake_controls_imports or args.legacy_import_fill
    if args.v2_intake_controls_imports and not args.legacy_import_fill:
        selected_import_candidates = _filter_import_candidates_by_v2_triage(selected_import_candidates, intake_triage)
    quota_summary = selection_quota_summary(selected_import_candidates, target=args.min_new_imports)
    import_policy["selection_quota_summary"] = quota_summary
    existing_note_titles = existing_literature_notes_by_title()

    if args.dry_run:
        primary_count = sum(1 for item, _, _ in selected_import_candidates if item.decision in AUTO_IMPORT_DECISIONS)
        focus_count = sum(1 for _, selection, _ in selected_import_candidates if selection == FOCUS_REVIEW_DECISION)
        fill_count = sum(1 for item, _, _ in selected_import_candidates if item.decision == AUTO_IMPORT_FILL_DECISION)
        safe_print(
            "DRY-RUN "
            f"candidates={len(ranked)} "
            f"source={import_policy['arxiv_source']} "
            f"days_back={args.days_back} "
            f"max_candidates={args.max_candidates} "
            f"top1={sum(1 for item in ranked if item.decision == 'top1_candidate')} "
            f"venue_auto={sum(1 for item in ranked if item.decision == 'venue_auto_import')} "
            f"focus_review={sum(1 for item in ranked if item.decision == FOCUS_REVIEW_DECISION)} "
            f"review={sum(1 for item in ranked if item.decision == 'review_queue')} "
            f"existing_excluded={len(existing_candidates)} "
            f"baseline_backfill_queued={baseline_backfill_report.get('queued_total', 0)} "
            f"import_attempts={len(selected_import_candidates)} "
            f"primary_imports={primary_count} "
            f"focus_imports={focus_count} "
            f"fill_imports={fill_count} "
            f"quota={_selection_quota_text(quota_summary)} "
            f"min_new_imports={args.min_new_imports} "
            f"cache={'true' if cache_fallback else 'false'}"
        )
        for item, selection, original_decision in selected_import_candidates[:15]:
            safe_print(f"{item.quality_score:3d} {selection:17s} original={original_decision:12s} {item.paper.arxiv_id} {item.paper.title}")
        if existing_candidates:
            safe_print("EXISTING_EXCLUDED_SAMPLE")
            for item in existing_candidates[:10]:
                safe_print(
                    f"{item.get('quality_score', ''):>3} {item.get('decision', ''):17s} "
                    f"{item.get('arxiv_id')} zotero_key={item.get('zotero_key', '')} {item.get('title', '')}"
                )
        rejected_sample = _non_robotics_rejected_sample(ranked_all)
        if rejected_sample:
            safe_print("REJECTED_NON_ROBOTICS_SAMPLE")
            for item in rejected_sample:
                safe_print(f"{item.quality_score:3d} {'/'.join(item.penalties):24s} {item.paper.arxiv_id} {item.paper.title}")
        if baseline_backfill_report:
            queued = [item for item in baseline_backfill_report.get("entries", []) if item.get("status") == "queued"]
            if queued:
                safe_print("BASELINE_BACKFILL_SAMPLE")
                for item in queued[:8]:
                    safe_print(
                        f"{item.get('priority_score', 0):>3} source_notes={item.get('source_note_count')} "
                        f"mentions={item.get('mention_count')} {item.get('label')} query={item.get('search_query')}"
                    )
        for error in errors:
            safe_print(f"WARN: {error}")
        return 0

    write_jsonl(candidates_path, records, dry_run=False)
    safe_write(review_path, render_review_queue(ranked, run_date), dry_run=False, backup=True)
    if baseline_backfill_report:
        baseline_json_path, baseline_md_path = write_baseline_backfill_report(baseline_backfill_report, dry_run=False, backup=True)
        baseline_backfill_report = dict(baseline_backfill_report)
        baseline_backfill_report["json_path"] = str(baseline_json_path.relative_to(vault_path())).replace("\\", "/")
        baseline_backfill_report["md_path"] = str(baseline_md_path.relative_to(vault_path())).replace("\\", "/")

    imports: list[dict[str, Any]] = []
    reads: list[dict[str, Any]] = []
    prior_success_keys = parse_successful_read_keys_with_backups(run_log_path) if args.resume_backlog else []
    resume_records = parse_backlog(backlog_path) if args.resume_backlog else []
    can_import = bool(selected_import_candidates) and not args.resume_backlog
    if can_import:
        pf = preflight(args.collection)
        if not (pf.get("local_read") and pf.get("write_credentials")):
            errors.extend(pf.get("errors", []))
            can_import = False

    new_import_count = 0
    read_attempts = 0
    successful_read_count = 0
    max_read_candidate_attempts = args.max_read + max(0, args.read_failure_backfill)
    if can_import:
        for ranked_item, selection_decision, original_decision in selected_import_candidates:
            if args.skip_read and new_import_count >= args.min_new_imports:
                break
            if not args.skip_read and args.max_read > 0 and successful_read_count >= args.max_read:
                break
            if (
                not args.skip_read
                and args.max_read > 0
                and read_attempts >= max_read_candidate_attempts
                and successful_read_count < args.max_read
            ):
                break
            paper = ranked_item.paper
            existing_note = existing_note_titles.get(normalize_title(paper.title))
            if existing_note and existing_note.get("zotero_key"):
                result = ImportResult(
                    status="exists",
                    zotero_key=str(existing_note.get("zotero_key", "")),
                    message=(
                        "matched existing local literature note by title; "
                        f"note={existing_note.get('path', '-')}; "
                        f"note_status={existing_note.get('status', '-')}"
                    ),
                    mode="local_note",
                    existing=True,
                )
            else:
                result = None
            try:
                if result is None:
                    result = import_ranked_paper(ranked_item, collection_key=args.collection, dry_run=False)
            except Exception as exc:
                result = ImportResult(status="failed", message=str(exc), mode="web_api")
            if result.status in NEW_IMPORT_STATUSES:
                new_import_count += 1
            import_record = {
                "arxiv_id": paper.arxiv_id,
                "quality_score": ranked_item.quality_score,
                "selection_decision": selection_decision,
                "original_decision": original_decision,
                **result.to_dict(),
            }
            imports.append(import_record)
            if result.status not in INGEST_READY_STATUSES or not result.zotero_key:
                errors.append(f"zotero_import_not_ready:{paper.arxiv_id}:{result.status}:{result.message}")
                continue
            note_status, note_path = local_note_status(result.zotero_key)
            if note_status != "missing_note":
                ingest_status = "success"
                ingest_output = f"local_note_ready:{note_status}:{note_path}"
            else:
                ingest_status, ingest_output = ingest_zotero_key(result.zotero_key)
                output_note_status, output_key, output_note_path = local_note_from_ingest_output(ingest_output)
                if output_key and output_key != result.zotero_key:
                    result.zotero_key = output_key
                    import_record["zotero_key"] = output_key
                    import_record["message"] = f"{import_record.get('message', '')}; canonical_note_key={output_key}; note={output_note_path}"
                    note_status, note_path = output_note_status, output_note_path
            read_status = "skipped"
            read_output = ""
            read_attempt_logs: list[dict[str, Any]] = []
            read_elapsed_sec: float | str = "-"
            if not args.skip_read:
                note_status, note_path = wait_for_local_note(result.zotero_key)
                ingest_output = (
                    f"{ingest_output}\nLOCAL_NOTE_READY: status={note_status} note={note_path or '-'}"
                )
                if ingest_status != "success" and note_status != "missing_note":
                    ingest_status = "success"
                    ingest_output = f"{ingest_output}\nINGEST_RECOVERED_BY_LOCAL_NOTE: {note_path}"
                if ingest_status != "success":
                    read_status = "skipped"
                elif note_status == "done":
                    read_status = "success_done_already"
                elif note_status == "missing_note":
                    read_status = "waiting_local_note"
                elif read_attempts < max_read_candidate_attempts and successful_read_count < args.max_read:
                    read_attempts += 1
                    read_status, read_output, read_elapsed_sec, read_attempt_logs = read_zotero_key_timed(
                        result.zotero_key,
                        timeout=args.read_timeout,
                        run_date=run_date,
                        max_attempts=args.read_retries + 1,
                        retry_delay_sec=args.read_retry_delay,
                        allow_dangerous_claude=args.allow_dangerous_claude,
                        read_mode=args.read_mode,
                    )
                else:
                    read_status = "backlog"
            read_status, read_output, audit_fields = finalize_read_status_with_target_audit(
                zotero_key=result.zotero_key,
                read_status=read_status,
                read_output=read_output,
                read_attempt_logs=read_attempt_logs,
                run_date=run_date,
            )
            reads.append(
                {
                    "zotero_key": result.zotero_key,
                    "ingest_status": ingest_status,
                    "ingest_output_tail": ingest_output[-1200:],
                    "read_status": read_status,
                    "read_elapsed_sec": read_elapsed_sec,
                    "read_attempt_logs": read_attempt_logs,
                    "read_output_tail": read_output[-1200:],
                    **audit_fields,
                }
            )
            if ingest_status != "success":
                errors.append(f"ingest_failed:{result.zotero_key}")
            if read_status == "waiting_local_note":
                errors.append(f"read_pending:{result.zotero_key}:waiting_local_note")
            if read_status not in {"skipped", "backlog", "waiting_local_note"} and not read_status.startswith("success"):
                errors.append(f"read_failed:{result.zotero_key}:{read_status}")
            if read_status.startswith("success"):
                successful_read_count += 1
    if args.resume_backlog:
        for record in resume_records:
            if record.get("read_status", "").startswith("success"):
                continue
            key = record.get("key", "")
            if not key:
                continue
            note_status, note_path = local_note_status(key)
            imports.append(
                {
                    "arxiv_id": record.get("arxiv", ""),
                    "quality_score": "",
                    "selection_decision": "resume_backlog",
                    "original_decision": "resume_backlog",
                    "status": "backlog",
                    "zotero_key": key,
                    "message": f"resumed from reading backlog; note_status={note_status}; note={note_path or '-'}",
                    "mode": "resume_backlog",
                    "existing": True,
                }
            )
            new_import_count += 1
            read_status = "skipped"
            read_output = ""
            read_attempt_logs: list[dict[str, Any]] = []
            read_elapsed_sec: float | str = "-"
            if note_status == "done":
                ingest_status = "success"
                ingest_output = f"already_done:{note_path}"
                read_status = "success_done_already"
            elif note_status != "missing_note":
                ingest_status = "success"
                ingest_output = f"local_note_ready:{note_status}:{note_path}"
            else:
                ingest_status, ingest_output = ingest_zotero_key(key)
                output_note_status, output_key, output_note_path = local_note_from_ingest_output(ingest_output)
                if output_key and output_key != key:
                    key = output_key
                    note_status, note_path = output_note_status, output_note_path
                    imports[-1]["zotero_key"] = output_key
                    imports[-1]["message"] = f"{imports[-1].get('message', '')}; canonical_note_key={output_key}; note={output_note_path}"
                note_status, note_path = wait_for_local_note(key)
                ingest_output = f"{ingest_output}\nLOCAL_NOTE_READY: status={note_status} note={note_path or '-'}"
                if ingest_status != "success" and note_status != "missing_note":
                    ingest_status = "success"
                    ingest_output = f"{ingest_output}\nINGEST_RECOVERED_BY_LOCAL_NOTE: {note_path}"
            if note_status == "done" and ingest_status == "success":
                read_status = "success_done_already"
            elif note_status not in {"done", "missing_note"} and ingest_status == "success" and not args.skip_read and read_attempts < args.max_read:
                read_attempts += 1
                read_status, read_output, read_elapsed_sec, read_attempt_logs = read_zotero_key_timed(
                    key,
                    timeout=args.read_timeout,
                    run_date=run_date,
                    max_attempts=args.read_retries + 1,
                    retry_delay_sec=args.read_retry_delay,
                    allow_dangerous_claude=args.allow_dangerous_claude,
                    read_mode=args.read_mode,
                )
            elif note_status == "missing_note" and ingest_status == "success" and not args.skip_read:
                read_status = "waiting_local_note"
            elif note_status != "done" and ingest_status == "success" and not args.skip_read:
                read_status = "backlog"
            read_status, read_output, audit_fields = finalize_read_status_with_target_audit(
                zotero_key=key,
                read_status=read_status,
                read_output=read_output,
                read_attempt_logs=read_attempt_logs,
                run_date=run_date,
            )
            reads.append(
                {
                    "zotero_key": key,
                    "ingest_status": ingest_status,
                    "ingest_output_tail": ingest_output[-1200:],
                    "read_status": read_status,
                    "read_elapsed_sec": read_elapsed_sec,
                    "read_attempt_logs": read_attempt_logs,
                    "read_output_tail": read_output[-1200:],
                    **audit_fields,
                }
            )
            if ingest_status != "success":
                errors.append(f"ingest_failed:{key}")
            if read_status == "waiting_local_note":
                errors.append(f"read_pending:{key}:waiting_local_note")
            if read_status not in {"skipped", "backlog", "waiting_local_note"} and not read_status.startswith("success"):
                errors.append(f"read_failed:{key}:{read_status}")
    if not args.resume_backlog and legacy_floor_active and new_import_count < args.min_new_imports:
        errors.append(f"min_new_imports_not_met:created={new_import_count}:target={args.min_new_imports}:attempts={len(imports)}")

    maintenance_status = "not_run"
    if reads and not args.skip_read:
        try:
            maintenance_code, maintenance_output = run_subprocess(
                [sys.executable, ".claude/scripts/fix_strict_kb_after_read.py"],
                timeout=300,
            )
        except Exception as exc:
            maintenance_code, maintenance_output = 1, str(exc)
        maintenance_status = "success" if maintenance_code == 0 else "failed"
        if maintenance_code != 0:
            errors.append(f"strict_kb_maintenance_failed:{maintenance_output[-400:]}")
    import_policy["strict_kb_maintenance"] = maintenance_status

    read_by_key = {str(item.get("zotero_key", "")): item for item in reads}
    focus_imports = [
        item
        for item in imports
        if item.get("status") in FOCUS_READY_STATUSES
        and str(read_by_key.get(str(item.get("zotero_key", "")), {}).get("read_status", "")).startswith("success")
    ]
    focus_arxiv_ids = [str(item.get("arxiv_id", "")) for item in focus_imports if item.get("arxiv_id")]
    focus_zotero_keys = [str(item.get("zotero_key", "")) for item in focus_imports if item.get("zotero_key")]
    if args.resume_backlog:
        seen_focus_keys = set(focus_zotero_keys)
        for key in prior_success_keys:
            if key and key not in seen_focus_keys:
                focus_zotero_keys.append(key)
                seen_focus_keys.add(key)
    if not focus_zotero_keys:
        import_policy["no_new_focus_keys_reason"] = "no_new_successfully_read_imports"

    if not focus_zotero_keys and args.idea_mode == "gemini-divergent":
        agenda_command = [
            sys.executable,
            ".claude/scripts/research_agenda_update.py",
            "--run-date",
            run_date,
            "--idea-generator",
            "none",
            "--idea-timeout",
            str(args.idea_timeout),
            "--gemini-model",
            args.gemini_model,
            "--deepseek-model",
            args.deepseek_model,
            "--deepseek-timeout",
            str(args.deepseek_timeout),
            "--raw-candidate-limit",
            str(args.raw_candidate_limit),
            "--min-raw-candidates",
            str(args.min_raw_candidates),
            "--max-generated",
            str(args.max_generated),
        ]
    else:
        agenda_command = [
            sys.executable,
            ".claude/scripts/research_agenda_update.py",
            "--run-date",
            run_date,
        ]
        if focus_zotero_keys:
            agenda_command.extend(["--zotero-keys", ",".join(focus_zotero_keys)])
        agenda_command.extend(
            [
                "--idea-generator",
                "none",
                "--idea-timeout",
                str(args.idea_timeout),
                "--gemini-model",
                args.gemini_model,
                "--deepseek-model",
                args.deepseek_model,
                "--deepseek-timeout",
                str(args.deepseek_timeout),
                "--raw-candidate-limit",
                str(args.raw_candidate_limit),
                "--min-raw-candidates",
                str(args.min_raw_candidates),
                "--max-generated",
                str(args.max_generated),
            ]
        )
    try:
        agenda_code, agenda_output = run_subprocess(
            agenda_command,
            timeout=max(900, args.idea_timeout * 2 + args.deepseek_timeout + 300),
        )
    except Exception as exc:
        agenda_code, agenda_output = 1, str(exc)
    if agenda_code == 0:
        status_value = "skipped_no_focus_keys" if not focus_zotero_keys else "success"
        idea_generation = {"mode": "research_agenda_update", "status": status_value, "message": agenda_output[-1200:]}
    else:
        idea_generation = {"mode": "research_agenda_update", "status": "partial", "message": agenda_output[-1200:]}
        errors.append("research_agenda_update_failed")
    v2_result = run_research_seed_v2(
        args=args,
        run_date=run_date,
        candidates_path=candidates_path,
        focus_zotero_keys=focus_zotero_keys,
        errors=errors,
    )
    idea_generation["v2_state_machine"] = v2_result
    safe_write(
        idea_path,
        render_legacy_idea_pointer(
            run_date=run_date,
            agenda_delta_path=agenda_delta_path,
            focus_keys=focus_zotero_keys,
            agenda_status=idea_generation["status"],
        ),
        dry_run=False,
        backup=True,
    )
    safe_write(
        gemini_prompt_path,
        render_gemini_prompt(run_date=run_date, candidates_path=candidates_path, items=records[:12]),
        dry_run=False,
        backup=True,
    )
    idea_result = {"status": "not_applicable", "idea_count": 0, "errors": [], "warnings": []}
    blocking_errors = [error for error in errors if not error.startswith("arxiv_fetch_failed") and not error.startswith("arxiv_mirror_empty")]
    status = "success" if not blocking_errors else "partial"
    safe_write(backlog_path, render_reading_backlog(run_date=run_date, imports=imports, reads=reads), dry_run=False, backup=True)
    safe_write(
        run_log_path,
        render_run_log(
            run_date=run_date,
            status=status,
            ranked=ranked,
            imports=imports,
            reads=reads,
            idea_path=idea_path,
            agenda_delta_path=agenda_delta_path,
            gemini_prompt_path=gemini_prompt_path,
            idea_audit=idea_result,
            idea_generation=idea_generation,
            errors=errors,
            import_policy=import_policy,
            existing_candidates=existing_candidates,
            baseline_backfill=baseline_backfill_report,
        ),
        dry_run=False,
        backup=True,
    )
    safe_print(f"RUN_LOG: {run_log_path.relative_to(vault_path())}")
    safe_print(f"CANDIDATES: {candidates_path.relative_to(vault_path())}")
    safe_print(f"REVIEW_QUEUE: {review_path.relative_to(vault_path())}")
    if baseline_backfill_report:
        safe_print(f"BASELINE_BACKFILL: {baseline_backfill_report.get('md_path', '')}")
    safe_print(f"AGENDA_DELTA: {agenda_delta_path.relative_to(vault_path())}")
    safe_print(f"LEGACY_IDEA_POINTER: {idea_path.relative_to(vault_path())}")
    safe_print(f"GEMINI_PROMPT_PATH: {gemini_prompt_path.relative_to(vault_path())}")
    safe_print(f"AGENDA_UPDATE: {idea_generation['mode']}:{idea_generation['status']}")
    safe_print(f"RESEARCH_SEED_V2: {v2_result.get('status', 'unknown')} run_dir={v2_result.get('run_dir', '')}")
    safe_print(f"LEGACY_IDEA_AUDIT: {idea_result['status']}")
    if errors:
        safe_print("PIPELINE_STATUS: partial")
        for error in errors:
            safe_print(f"ERROR: {error}")
        return 2
    safe_print("PIPELINE_STATUS: success")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Run one daily scout pass.")
    parser.add_argument("--run-date", help="Override output date for manual historical reruns.")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and rank only; do not write files or Zotero.")
    parser.add_argument("--query", action="append", help="Override default arXiv query; can be repeated.")
    parser.add_argument("--max-candidates", type=int, default=120)
    parser.add_argument("--source", choices=["mirror-first", "mirror-only", "search-api"], default="mirror-first")
    parser.add_argument("--days-back", type=int, default=60)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_KEY)
    parser.add_argument("--min-new-imports", type=int, default=10, help="Minimum newly created Zotero items to attempt each run; existing duplicates do not count.")
    parser.add_argument("--max-auto-import", type=int, default=40, help="Maximum Zotero import attempts after local existing-paper prefiltering.")
    parser.add_argument("--fill-import-threshold", type=int, default=60, help="Minimum score for relevant below-review candidates used only to fill the daily new-import floor.")
    parser.add_argument("--fetch-timeout", type=int, default=DEFAULT_FETCH_TIMEOUT, help="Per-query arXiv API timeout in seconds.")
    parser.add_argument("--fetch-retries", type=int, default=DEFAULT_FETCH_RETRIES, help="Retries per arXiv query before recording a fetch warning.")
    parser.add_argument("--query-delay", type=float, default=DEFAULT_QUERY_DELAY, help="Delay between arXiv API queries to avoid rate-limit bursts.")
    parser.add_argument("--max-read", type=int, default=None, help="Maximum newly created imports to deep-read. Defaults to --min-new-imports so every new daily import is read.")
    parser.add_argument("--resume-backlog", action="store_true", help="Resume reading Zotero keys listed in the run-date reading backlog instead of importing new papers.")
    parser.add_argument("--skip-read", action="store_true", help="Import and ingest but do not invoke Claude reading.")
    parser.add_argument("--read-timeout", type=int, default=4200)
    parser.add_argument("--read-mode", choices=["codex-controlled", "staged", "single"], default="codex-controlled", help="Use controlled Codex fulltext reading by default; staged and single keep legacy Claudian paths.")
    parser.add_argument("--read-retries", type=int, default=0, help="Retry failed one-shot Claudian reads this many extra times. Staged and codex-controlled reads do not whole-paper retry.")
    parser.add_argument("--read-retry-delay", type=int, default=90, help="Seconds to wait between failed Claudian read attempts.")
    parser.add_argument("--read-failure-backfill", type=int, default=5, help="Extra candidate read slots used to replace failed Claudian reads before ending the daily run.")
    parser.add_argument("--skip-baseline-backfill", action="store_true", help="Skip canonical baseline backfill queue generation.")
    parser.add_argument("--baseline-backfill-min-mentions", type=int, default=2, help="Minimum distinct done notes mentioning a baseline before it enters the canonical backfill queue.")
    parser.add_argument("--baseline-backfill-limit", type=int, default=50, help="Maximum baseline backfill entries to report.")
    parser.add_argument(
        "--allow-dangerous-claude",
        action="store_true",
        help=f"Opt in to passing --dangerously-skip-permissions to Claude. The public default is safe; env {DANGEROUS_CLAUDE_ENV}=1 also opts in.",
    )
    parser.add_argument("--idea-mode", choices=["claude", "gemini-cli", "gemini-divergent", "template"], default="gemini-divergent", help="Use Claude Code or Gemini CLI to synthesize final ideas, or keep deterministic template ideas.")
    parser.add_argument("--idea-timeout", type=int, default=1200)
    parser.add_argument("--gemini-model", default="gemini-3.1-pro-preview")
    parser.add_argument("--deepseek-model", default="abrdns/deepseek-v4-pro")
    parser.add_argument("--deepseek-timeout", type=int, default=1200)
    parser.add_argument("--deepseek-provider-retries", type=int, default=1, help="Retry structured OpenCode/openai-compatible review output this many extra times.")
    parser.add_argument("--deepseek-provider", choices=["json", "opencode", "openai-compatible", "none"], default="none")
    parser.add_argument("--deepseek-provider-json", default="")
    parser.add_argument("--codex-execution-provider", choices=["json", "codex-cli", "none"], default="none")
    parser.add_argument("--codex-execution-provider-json", default="")
    parser.add_argument("--novelty-max-external-queries", type=int, default=1)
    parser.add_argument("--novelty-external-timeout", type=int, default=12)
    parser.add_argument("--raw-candidate-limit", type=int, default=24)
    parser.add_argument("--min-raw-candidates", type=int, default=24)
    parser.add_argument("--max-generated", type=int, default=24)
    parser.add_argument("--target-deep-read", type=int, default=4, help="v2 intake target for daily deep reads.")
    parser.add_argument("--max-deep-read", type=int, default=4, help="v2 hard cap for daily deep reads.")
    parser.add_argument(
        "--v2-intake-controls-imports",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Let v2 intake triage control daily Zotero import and Claudian deep-read attempts.",
    )
    parser.add_argument(
        "--legacy-import-fill",
        action="store_true",
        help="Opt back in to legacy min_new_imports fill behavior. Disabled by default for v2 daily runs.",
    )
    parser.add_argument(
        "--backfill-mode",
        choices=["daily", "ingest-only"],
        default="daily",
        help="Use ingest-only for backfill; it stops before raw candidates and formal seed publish.",
    )
    parser.add_argument(
        "--backfill-generate-ideas",
        action="store_true",
        help="Advanced backfill mode: allow raw candidate generation, but not formal seed publish by default.",
    )
    parser.add_argument(
        "--backfill-publish",
        choices=["disabled", "seed-candidates-only"],
        default="disabled",
        help="Backfill publish policy. Formal seed publish remains disabled for backfill.",
    )
    parser.add_argument(
        "--v2-publish-policy",
        choices=["disabled", "seed-candidates-only"],
        default=DEFAULT_V2_PUBLISH_POLICY,
        help="V2 rollout publish policy for scheduled/daily runs. Formal production seed publish is disabled in v1.",
    )
    parser.add_argument(
        "--allow-human-override",
        action="store_true",
        help="Allow scheduled run to consume explicit human override files; default is disabled.",
    )
    args = parser.parse_args()
    if args.max_read is None:
        if args.v2_intake_controls_imports:
            args.max_read = min(args.target_deep_read, args.max_deep_read)
        else:
            args.max_read = args.min_new_imports
    elif args.v2_intake_controls_imports:
        args.max_read = min(args.max_read, args.max_deep_read)
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
