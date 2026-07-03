from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPTS_DIR.parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from candidate_portfolio_select import select_portfolio
from codex_seed_review import execution_review
from arxiv_ranker import ArxivPaper, RankedPaper
from daily_arxiv_pipeline import _filter_import_candidates_by_v2_triage, append_provider_args, build_v2_review_stages, resolve_v2_publish_policy, run_research_seed_v2
from deepseek_scientific_review import build_payload as build_deepseek_payload
from paper_intake_triage import build_triage
from publish_research_run import publish
from research_agenda_extract import build_paper_primitives
from research_claim_graph import build_edges, build_graph_records, build_nodes
from research_seed_v2_common import (
    artifact_dir,
    duplicate_guard,
    file_sha256,
    init_manifest,
    read_json,
    run_dir,
    schema_validator_available,
    validate_json_file,
    validate_payload,
    write_json,
    write_jsonl,
    write_run_artifact,
)
from survival_decision import decide
from tension_map import build_tensions, validate_tensions
from validate_research_run import validate_run
from weekly_strategy_review import sanitize_overrides
from audit_daily_automation_quality import (
    CONTRACT_DIR,
    _audit_v2_state_machine,
    _daily_read_target_satisfied,
    _failed_or_backlog_read_items,
    _moved_to_v2_missing_artifact_issues,
    _provider_backed_artifact_issue,
    _scheduled_provider_artifacts_expected,
    _scan_text_for_seed_writer,
    _v2_deepseek_review_covers_battle,
)
import research_agenda_review as agenda_review
import research_agenda_ideate as agenda_ideate
import novelty_baseline_scan as novelty_scan
import candidate_portfolio_select as portfolio_select
import deepseek_scientific_review as deepseek_review
import codex_seed_review as codex_review
import opencode_cli_adapter
import active_seed_dashboard as seed_dashboard


RUN_DATE = "2099-01-02"
LEGACY_FORMAL_DISABLED_STATUS = "legacy_formal_publish_disabled_use_formal_rehearsal_packet"


class ResearchSeedV2StateMachineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.old_root = os.environ.get("RESEARCH_SEED_V2_AGENDA_ROOT")
        os.environ["RESEARCH_SEED_V2_AGENDA_ROOT"] = self.tmp.name
        init_manifest(RUN_DATE, backfill_mode="daily")

    def tearDown(self) -> None:
        if self.old_root is None:
            os.environ.pop("RESEARCH_SEED_V2_AGENDA_ROOT", None)
        else:
            os.environ["RESEARCH_SEED_V2_AGENDA_ROOT"] = self.old_root
        self.tmp.cleanup()

    def test_opencode_text_events_preserve_streamed_json_chunks(self) -> None:
        chunks = ['{"reviews":[{"candidate_id":"', "cand-alpha", '","status":"success"}]}']
        stdout = "\n".join(
            json.dumps({"type": "text", "part": {"type": "text", "text": chunk}})
            for chunk in chunks
        )

        text, event_count = opencode_cli_adapter._extract_text_events(stdout)
        payload = deepseek_review._extract_json_object(text)

        self.assertEqual(event_count, 3)
        self.assertEqual(text, "".join(chunks))
        self.assertEqual(payload["reviews"][0]["candidate_id"], "cand-alpha")

    def test_quality_audit_ignores_read_failure_recovered_by_recovery_log(self) -> None:
        run_text = """## Reading

- zotero_key=HD7NP5I3 ingest=success read=failed:finalize_reading read_elapsed_sec=182.47 target_note_audit=not_run audit_json=-
"""
        recovery_text = """- status: success

## Reading

- zotero_key=HD7NP5I3 ingest=success read=success_done_already read_elapsed_sec=- target_note_audit=passed audit_json=projects/arxiv-daily/read-logs/2026-05-27/HD7NP5I3-target-note-audit.json
"""

        self.assertEqual(_failed_or_backlog_read_items(run_text, recovery_text, "success"), [])
        self.assertEqual(_failed_or_backlog_read_items(run_text, recovery_text, "partial"), [])

    def test_quality_audit_read_target_satisfied_allows_optional_import_failures(self) -> None:
        self.assertTrue(_daily_read_target_satisfied(4, 4, []))
        self.assertFalse(_daily_read_target_satisfied(3, 4, []))
        self.assertFalse(_daily_read_target_satisfied(4, 4, [{"key": "A", "read": "failed"}]))

    def test_agenda_extract_json_prefers_candidates_object_after_logs(self) -> None:
        text = '{"event":"thinking"}\n```json\n{"candidates":[{"title":"candidate"}]}\n```'

        payload = agenda_ideate._extract_json_object(text)

        self.assertEqual(payload["candidates"][0]["title"], "candidate")

    def test_agenda_extract_json_handles_trailing_prose(self) -> None:
        text = 'Here is the JSON:\n{"candidates":[{"title":"candidate"}]}\nDone.'

        payload = agenda_ideate._extract_json_object(text)

        self.assertEqual(payload["candidates"][0]["title"], "candidate")

    def test_opencode_json_review_agent_disables_todowrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = opencode_cli_adapter._write_json_review_agent_config(tmp, "deepseek/deepseek-v4-pro")
            config = json.loads(Path(config_path).read_text(encoding="utf-8"))

        tools = config["agent"]["research-json-review"]["tools"]
        self.assertEqual(config["agent"]["research-json-review"]["model"], "deepseek-v4-pro")
        self.assertIs(tools["todowrite"], False)
        self.assertIs(tools["todo_write"], False)
        self.assertIs(tools["webfetch"], False)
        self.assertIs(tools["web_search"], False)
        self.assertIn("todowrite", config["agent"]["research-json-review"]["prompt"])
        self.assertIn("webfetch", config["agent"]["research-json-review"]["prompt"])

    def test_opencode_json_review_agent_preserves_global_provider_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            global_cfg = Path(tmp) / "opencode-global.json"
            global_cfg.write_text(
                json.dumps(
                    {
                        "provider": {
                            "abrdns": {
                                "models": {"deepseek-v4-pro": {"id": "deepseek-v4-pro"}},
                                "options": {"baseURL": "https://example.test/v1", "apiKey": "sk-test"},
                            }
                        },
                        "model": "deepseek-v4-pro",
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(opencode_cli_adapter, "_global_opencode_config_paths", return_value=[global_cfg]):
                config_path = opencode_cli_adapter._write_json_review_agent_config(str(workspace), "abrdns/deepseek-v4-pro")
            config = json.loads(Path(config_path).read_text(encoding="utf-8"))

        self.assertIn("provider", config)
        self.assertIn("abrdns", config["provider"])
        self.assertEqual(config["agent"]["research-json-review"]["model"], "deepseek-v4-pro")

    def test_opencode_json_review_agent_normalizes_relative_plugin_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            plugin_dir = Path(tmp) / "plugins"
            plugin_dir.mkdir()
            (plugin_dir / "session-id-header.mjs").write_text("export default {};\n", encoding="utf-8")
            global_cfg = Path(tmp) / "opencode-global.json"
            global_cfg.write_text(
                json.dumps(
                    {
                        "plugin": ["./plugins/session-id-header.mjs", "./plugins/session-id-header.mjs"],
                        "provider": {"abrdns": {"models": {"deepseek-v4-pro": {"id": "deepseek-v4-pro"}}}},
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(opencode_cli_adapter, "_global_opencode_config_paths", return_value=[global_cfg]):
                config_path = opencode_cli_adapter._write_json_review_agent_config(str(workspace), "abrdns/deepseek-v4-pro")
            config = json.loads(Path(config_path).read_text(encoding="utf-8"))

        self.assertEqual(
            config["plugin"],
            [str((plugin_dir / "session-id-header.mjs").resolve())],
        )

    def test_run_opencode_cli_avoids_pure_mode_by_default(self) -> None:
        captured: dict[str, object] = {}

        class FakeProc:
            returncode = 0

            def __init__(self, command: list[str]) -> None:
                captured["command"] = command
                self.pid = 12345

            def communicate(self, timeout: int | None = None) -> tuple[str, str]:
                return ('{"type":"text","part":{"type":"text","text":"{\\"ok\\":true}"}}\n', "")

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            opencode_cli_adapter,
            "_resolve_opencode_path",
            return_value="opencode.exe",
        ), patch.object(
            opencode_cli_adapter,
            "_write_json_review_agent_config",
            return_value=str(Path(tmp) / "opencode.json"),
        ), patch.object(
            opencode_cli_adapter.subprocess,
            "Popen",
            side_effect=lambda *args, **kwargs: FakeProc(args[0]),
        ), patch.dict(
            os.environ,
            {"OPENCODE_JSON_REVIEW_PURE": ""},
            clear=False,
        ):
            result = opencode_cli_adapter.run_opencode_cli('{"ok":true}', timeout_sec=5)

        self.assertEqual(result["error"], "")
        self.assertNotIn("--pure", captured["command"])

    def assert_legacy_formal_publish_disabled(self, result: dict[str, object]) -> None:
        self.assertEqual(result["status"], LEGACY_FORMAL_DISABLED_STATUS)
        self.assertEqual(result["published"], [])
        self.assertIn("v1_disables_legacy_formal_seed_writing", result["blocked"])
        self.assertEqual(list((Path(self.tmp.name) / "idea_bank" / "seed").glob("*")), [])

    def candidate(self, candidate_id: str = "cand-alpha") -> dict[str, object]:
        return {
            "candidate_id": candidate_id,
            "title": "Anchored Contact Failure Benchmark",
            "problem": "Contact-rich DLO policies fail silently under latent friction shifts.",
            "mechanism": "Instrument the interface boundary and expose recovery triggers.",
            "interface_boundary": "vision-tactile-controller handoff",
            "strongest_baseline": "diffusion policy with fixed recovery threshold",
            "novelty_remaining_delta": "anchored latent failure benchmark",
            "lane": "infrastructure_or_benchmark",
            "supporting_nodes": ["claim-1"],
            "evidence_links": ["wiki/topics/example.md#claim-1"],
        }

    def write_claim_graph_node(
        self,
        *,
        node_id: str = "claim-1",
        confidence: str = "high",
        anchor_type: str = "section",
        requires_human_check: bool = False,
    ) -> None:
        anchor = {
            "source_note": "wiki/topics/example.md",
            "evidence_anchor": "wiki/topics/example.md#Results",
            "anchor": "wiki/topics/example.md#Results",
            "anchor_type": anchor_type,
            "section": "Results",
            "snippet": "Contact rich DLO policies fail under latent friction shifts.",
            "snippet_type": "section_summary",
            "pdf_page": None,
        }
        write_jsonl(
            artifact_dir(RUN_DATE) / "claim-graph-snapshot.jsonl",
            [
                {
                    "schema_version": "research_claim_graph.v1",
                    "record_type": "node",
                    "node_id": node_id,
                    "paper_key": "ABC123",
                    "source_note": "wiki/topics/example.md",
                    "source_title": "Anchored Contact Failure Benchmark",
                    "claim_type": "central_claim",
                    "statement": "Contact rich DLO policies fail under latent friction shifts.",
                    "confidence": confidence,
                    "confidence_reason": "test_fixture",
                    "evidence_anchor": anchor["evidence_anchor"],
                    "anchor_type": anchor_type,
                    "summary_origin": "section_summary",
                    "requires_human_check": requires_human_check,
                    "anchor": anchor,
                    "supporting_node_ids": [],
                }
            ],
        )

    def register_daily_arxiv_dry_run(self, *args: str) -> str:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(SCRIPTS_DIR / "register_daily_arxiv_task.ps1"),
                "-DryRun",
                *args,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            encoding="utf-8",
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        return output

    def review_stage_commands(self, **overrides: object) -> dict[str, list[str]]:
        values = {
            "deepseek_provider": "none",
            "deepseek_provider_json": "",
            "deepseek_model": "deepseek/test-model",
            "deepseek_timeout": 123,
            "deepseek_provider_retries": 1,
            "codex_execution_provider": "none",
            "codex_execution_provider_json": "",
            "idea_timeout": 45,
            "allow_human_override": False,
            "v2_publish_policy": "seed-candidates-only",
            "novelty_max_external_queries": 1,
            "novelty_external_timeout": 12,
        }
        values.update(overrides)
        stages = build_v2_review_stages(Namespace(**values), RUN_DATE)
        return {name: command for name, command, _timeout in stages}

    def run_daily_wrapper_test(
        self,
        *,
        pipeline_exit: int = 0,
        recovery_exit: int = 1,
        quality_exit: int = 0,
        extra_args: tuple[str, ...] = (),
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        root = Path(self.tmp.name) / "wrapper-root"
        script_dir = root / ".claude" / "scripts"
        script_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SCRIPTS_DIR / "run_daily_arxiv_task.ps1", script_dir / "run_daily_arxiv_task.ps1")
        shutil.copy2(SCRIPTS_DIR / "automation_pause_guard.ps1", script_dir / "automation_pause_guard.ps1")
        env = os.environ.copy()
        env["DAILY_ARXIV_WRAPPER_TEST_MODE"] = "1"
        env["DAILY_ARXIV_WRAPPER_TEST_PIPELINE_EXIT"] = str(pipeline_exit)
        env["DAILY_ARXIV_WRAPPER_TEST_RECOVERY_EXIT"] = str(recovery_exit)
        env["DAILY_ARXIV_WRAPPER_TEST_QUALITY_EXIT"] = str(quality_exit)
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_dir / "run_daily_arxiv_task.ps1"),
                *extra_args,
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result, root

    def write_bom_json(self, path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\ufeff" + json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_active_seed_support(self, candidate_id: str = "cand-alpha") -> None:
        write_run_artifact(
            RUN_DATE,
            "manual-prior-art-review.json",
            {
                "schema_version": "manual_prior_art_review.v1",
                "run_date": RUN_DATE,
                "reviews": [
                    {
                        "candidate_id": candidate_id,
                        "review_status": "completed",
                        "is_template": False,
                        "reviewer": "human",
                        "reviewed_at": "2099-01-02T12:00:00+00:00",
                        "searched_sources": ["OpenAlex", "Semantic Scholar", "arXiv", "Google Scholar/manual"],
                        "search_queries": ["anchored contact failure benchmark DLO latent friction"],
                        "review_quality_checklist": {
                            "venue_proceedings_checked": True,
                            "google_scholar_checked": True,
                            "openalex_checked": True,
                            "semantic_scholar_checked": True,
                            "arxiv_checked": True,
                            "lab_specific_sources_checked": True,
                            "negative_search_log_present": True,
                            "strongest_baseline_comparison_present": True,
                            "query_log_present": True,
                        },
                        "negative_search_log": [
                            {
                                "query": "anchored contact failure benchmark DLO latent friction",
                                "source": "OpenAlex",
                                "result_summary": "Near work is adjacent.",
                                "result_count": 8,
                                "why_not_overlap": "No anchored latent contact-failure benchmark.",
                            }
                        ],
                        "venue_search_checklist": [
                            {"venue": "RSS/CoRL/ICRA", "years": "2021-2099", "query": "DLO latent friction failure", "checked": True, "notes": "No direct overlap."}
                        ],
                        "nearest_works": [
                            {
                                "title": "Near work",
                                "source": "manual",
                                "url": "",
                                "overlap_type": "adjacent",
                                "what_is_already_done": "Adjacent benchmark exists.",
                                "remaining_delta": "This candidate targets anchored latent contact failure.",
                            }
                        ],
                        "strongest_baseline_comparison_table": [
                            {
                                "work_title": "Near work",
                                "source": "manual:near-work",
                                "overlap_type": "adjacent",
                                "stronger_than_candidate": False,
                                "why_not_kill_or_kills": "It lacks anchored latent contact-failure evidence.",
                                "remaining_delta": "This candidate targets anchored latent contact failure.",
                                "kill_condition": "Reject if it matches failure detection under latent friction shifts.",
                            }
                        ],
                        "strongest_baseline_judgment": {
                            "status": "known",
                            "baseline_name": "Diffusion policy with fixed recovery threshold",
                            "source_work_id": "manual:near-work",
                            "why_strongest": "It is the direct policy baseline for this contact-failure claim.",
                            "kill_condition": "Reject if it matches failure detection under latent friction shifts.",
                            "metric_or_task": "failure_detection_auc",
                            "implementation_feasibility": "available",
                        },
                        "remaining_delta": "Test latent friction failure with anchored evidence.",
                        "risk_acceptance": "Human reviewer accepts active-seed rehearsal risk after prior-art search.",
                        "decision": "allow_active_seed",
                        "reason": "Remaining delta is explicit and strongest baseline is known.",
                        "limitations": "Manual review is not a publishability proof.",
                        "manual_review_quality_status": "complete",
                        "reviewer_signature": {
                            "reviewer": "human",
                            "signed_at": "2099-01-02T12:00:00+00:00",
                            "statement": "I reviewed nearest works and accept the remaining-delta risk.",
                        },
                        "cannot_weaken": [
                            "deepseek_success_required",
                            "codex_accept_required",
                            "external_novelty_required",
                            "anchored_core_evidence_required",
                            "no_unresolved_fatal_flaw",
                            "fresh_external_cache_required",
                        ],
                    }
                ],
            },
            state="manual_prior_art_reviewed",
        )
        write_run_artifact(
            RUN_DATE,
            "baseline-table.json",
            {
                "schema_version": "baseline_table.v1",
                "run_date": RUN_DATE,
                "candidate_id": candidate_id,
                "baselines": [
                    {
                        "baseline_id": "baseline-nearest",
                        "name": "Near work",
                        "source_work_id": "manual:near-work",
                        "source": "novelty_scan",
                        "baseline_role": "nearest_work",
                        "why_strongest": "",
                        "covered_claim": "prior adjacent benchmark",
                        "kill_condition": "",
                        "implementation_feasibility": "unknown",
                        "metric": "",
                        "known_result": "",
                        "evidence_anchor_ids": [],
                        "confidence": "medium",
                    },
                    {
                        "baseline_id": "baseline-strongest",
                        "name": "Diffusion policy with fixed recovery threshold",
                        "source_work_id": "manual:near-work",
                        "source": "manual_prior_art",
                        "baseline_role": "strongest_baseline_final",
                        "why_strongest": "Direct policy baseline for the failure detection claim.",
                        "covered_claim": "latent friction contact-failure benchmark",
                        "kill_condition": "Reject if it matches failure detection under latent friction shifts.",
                        "implementation_feasibility": "available",
                        "metric": "failure_detection_auc",
                        "known_result": "",
                        "evidence_anchor_ids": ["claim-1"],
                        "confidence": "high",
                    },
                ],
                "strongest_baseline_id": "baseline-strongest",
                "strongest_baseline_final": {
                    "status": "known",
                    "baseline_id": "baseline-strongest",
                    "name": "Diffusion policy with fixed recovery threshold",
                    "source": "manual_prior_art",
                    "kill_condition": "Reject if it matches failure detection under latent friction shifts.",
                    "metric_or_task": "failure_detection_auc",
                    "implementation_feasibility": "available",
                    "why_strongest": "Direct policy baseline for the failure detection claim.",
                },
                "baseline_verification_status": "verified",
                "baseline_execution_readiness": {
                    "status": "ready",
                    "source": "manual_prior_art_review",
                    "implementation_path": "baselines/diffusion-policy-threshold",
                    "dataset_or_sim": "public sim",
                    "compute_budget": "one GPU day",
                    "metric_automation": "pytest evaluator",
                    "not_applicable_reason": "",
                    "blocking_issues": [],
                },
            },
            state="baseline_table_built",
        )
        plan_path = Path(self.tmp.name) / "pilots" / "anchored-contact-failure-benchmark" / "pilot-plan.json"
        write_json(
            plan_path,
            {
                "schema_version": "pilot_plan.v1",
                "seed_slug": "anchored-contact-failure-benchmark",
                "candidate_id": candidate_id,
                "pilot_status": "planned",
                "metric": "failure_detection_auc",
                "metric_automation": "pytest evaluator",
                "baseline_implementation_path": "baselines/diffusion-policy-threshold",
                "resource_budget": "one GPU day",
                "executable": True,
            },
        )

    def fake_urlopen_json(self, payload: dict[str, object], calls: list[object] | None = None):
        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(payload).encode("utf-8")

        def fake(request: object, timeout: int = 0) -> FakeResponse:
            if calls is not None:
                calls.append(request)
            return FakeResponse()

        return fake

    def provider_deepseek_payload(self, candidate: dict[str, object] | None = None) -> dict[str, object]:
        item = candidate or self.candidate()
        cid = str(item["candidate_id"])
        return {
            "provider_status": {"provider": "deepseek", "provider_backed": True, "mode": "bom-test"},
            "reviews": [
                {
                    "candidate_id": cid,
                    "status": "success",
                    "candidate_title": str(item["title"]),
                    "novelty_attack": "nearest work reviewed",
                    "baseline_attack": "baseline checked",
                    "mechanism_attack": "mechanism checked",
                    "evaluation_attack": "evaluation checked",
                    "scope_attack": "scope checked",
                    "a_plus_b_risk": "low",
                    "fatal_flaw": "",
                    "rescue_mutation": "",
                    "survivability_label": "survives",
                    "allowed_next_stage": "novelty_scan",
                }
            ],
        }

    def provider_codex_payload(self, candidate: dict[str, object] | None = None) -> dict[str, object]:
        item = candidate or self.candidate()
        cid = str(item["candidate_id"])
        return {
            "provider_status": {"provider": "codex", "provider_backed": True, "mode": "bom-test"},
            "reviews": [
                {
                    "candidate_id": cid,
                    "status": "success",
                    "candidate_title": str(item["title"]),
                    "action": "accept_for_user_review",
                    "no_hardware_pilot_feasibility": "feasible",
                    "public_dataset_or_sim_availability": "available",
                    "baseline_training_cost": "single GPU day",
                    "metric_automation": "scriptable",
                    "data_leakage_risk": "low",
                    "minimal_repo_plan": "notebook plus evaluator",
                    "real_robot_pilot_complexity": "low",
                    "reproducibility_path": "public sim",
                    "compute_time_budget": "one day",
                    "blocking_issues": [],
                    "nonblocking_risks": [],
                    "rewrite_request": "",
                    "rescue_signal": "",
                    "confidence": "medium",
                    "field_presence_only": False,
                }
            ],
        }

    def write_review_artifacts(
        self,
        *,
        candidate: dict[str, object] | None = None,
        novelty: str = "likely_open",
        codex_action: str = "accept_for_user_review",
        deepseek_label: str = "survives",
        fatal_flaw: bool = False,
        novelty_candidate_id: str | None = None,
        codex_candidate_id: str | None = None,
        mutations: list[dict[str, object]] | None = None,
        verification_scope: str = "local_only",
        external_providers_used: list[str] | None = None,
        formal_promotion_allowed: bool | None = None,
        deepseek_mode: str = "test",
        codex_mode: str = "test",
        ) -> str:
        item = candidate or self.candidate()
        cid = str(item["candidate_id"])
        self.write_claim_graph_node()
        write_run_artifact(
            RUN_DATE,
            "selected-candidates.json",
            {
                "schema_version": "selected_candidates.v1",
                "run_date": RUN_DATE,
                "selected": [item],
                "rejected": [],
                "selection_rules": {},
                "artifact_hashes": {},
            },
            state="portfolio_selected",
        )
        write_run_artifact(
            RUN_DATE,
            "deepseek-review.json",
            {
                "schema_version": "deepseek_review.v1",
                "run_date": RUN_DATE,
                "status": "success",
                "reviews": [
                    {
                        "candidate_id": cid,
                        "status": "success",
                        "novelty_attack": "nearest work reviewed",
                        "baseline_attack": "baseline checked",
                        "mechanism_attack": "mechanism checked",
                        "evaluation_attack": "evaluation checked",
                        "scope_attack": "scope checked",
                        "a_plus_b_risk": "low",
                        "survivability_label": deepseek_label,
                        "fatal_flaw": fatal_flaw,
                        "rescue_mutation": "reframe as a benchmark" if deepseek_label != "survives" else "",
                        "allowed_next_stage": "novelty_scan",
                    }
                ],
                "provider_status": {"provider": "deepseek", "provider_backed": True, "mode": deepseek_mode},
                "artifact_hashes": {},
            },
            state="attacked_by_deepseek",
        )
        if mutations is not None:
            write_run_artifact(
                RUN_DATE,
                "gemini-mutations.json",
                {
                    "schema_version": "gemini_mutations.v1",
                    "run_date": RUN_DATE,
                    "mutations": mutations,
                    "artifact_hashes": {},
                },
                state="rescued_or_mutated",
            )
        if external_providers_used is not None:
            external_providers = external_providers_used
        elif verification_scope == "local_plus_arxiv_api":
            external_providers = ["arxiv_api"]
        elif verification_scope == "local_plus_s2_or_openalex":
            external_providers = ["openalex"]
        elif verification_scope == "strict_multi_provider":
            external_providers = ["arxiv_api", "openalex"]
        else:
            external_providers = []
        if formal_promotion_allowed is None:
            formal_promotion_allowed = (
                novelty in {"likely_open", "partial_overlap"}
                and verification_scope in {"local_plus_s2_or_openalex", "strict_multi_provider"}
                and bool({"openalex", "semantic_scholar"} & set(external_providers))
            )
        provider_results = {
            provider: {"provider": provider, "status": "success", "records_scanned": 1, "cached": False, "queries": 1}
            for provider in external_providers
        }
        write_run_artifact(
            RUN_DATE,
            "novelty-scan.json",
            {
                "schema_version": "novelty_scan.v1",
                "run_date": RUN_DATE,
                "status": "completed" if verification_scope != "local_only" else "completed_local_only",
                "scans": [
                    {
                        "candidate_id": novelty_candidate_id or cid,
                        "candidate_title": str(item["title"]),
                        "status": "completed" if verification_scope != "local_only" else "completed_local_only",
                        "novelty_classification": novelty,
                        "promotion_allowed": novelty in {"likely_open", "partial_overlap"},
                        "formal_promotion_allowed": formal_promotion_allowed,
                        "verification_scope": verification_scope,
                        "external_providers_used": external_providers,
                        "formal_publish_risk": "external_scope_arxiv_only_not_full_prior_art" if verification_scope == "local_plus_arxiv_api" else "",
                        "provider_results": provider_results,
                        "provider_errors": [],
                        "nearest_works": [{"source": "test", "score": 0.1, "title": "Near work"}] if novelty != "unknown" else [],
                        "strongest_baseline": {"source": "test", "score": 0.1, "title": "Near work"} if novelty != "unknown" else {"source": "candidate_field", "score": 0, "title": str(item.get("strongest_baseline", ""))},
                        "local_scan_evidence": {},
                        "external_scan_evidence": provider_results,
                        "duplicate_guard": {"status": "pass", "action": "allow_with_lineage", "nearest_candidates": []},
                    }
                ],
                "provider_policy": {},
                "artifact_hashes": {},
            },
            state="novelty_checked",
        )
        write_run_artifact(
            RUN_DATE,
            "codex-execution-review.json",
            {
                "schema_version": "codex_execution_review.v1",
                "run_date": RUN_DATE,
                "status": "success",
                "reviews": [
                    {
                        "candidate_id": codex_candidate_id or cid,
                        "status": "success",
                        "action": codex_action,
                        "no_hardware_pilot_feasibility": "feasible",
                        "public_dataset_or_sim_availability": "available",
                        "baseline_training_cost": "single GPU day",
                        "metric_automation": "scriptable",
                        "data_leakage_risk": "low",
                        "minimal_repo_plan": "notebook plus evaluator",
                        "real_robot_pilot_complexity": "low",
                        "reproducibility_path": "public sim",
                        "compute_time_budget": "one day",
                    }
                ],
                "provider_status": {"provider": "codex", "provider_backed": True, "mode": codex_mode},
                "artifact_hashes": {},
            },
            state="execution_reviewed",
        )
        return cid

    def ranked_paper(self, index: int, *, score: int = 80) -> RankedPaper:
        paper = ArxivPaper(
            arxiv_id=f"2401.{index:05d}",
            title=f"Robot DLO Benchmark {index}",
            authors=["A. Author"],
            summary="A tactile deformable linear object benchmark for robot learning.",
            published="2099-01-01T00:00:00Z",
            updated="2099-01-01T00:00:00Z",
            url=f"https://arxiv.org/abs/2401.{index:05d}",
            pdf_url=f"https://arxiv.org/pdf/2401.{index:05d}",
            categories=["cs.RO"],
            primary_category="cs.RO",
        )
        return RankedPaper(
            paper=paper,
            quality_score=score,
            decision="top1_candidate",
            reasons=[],
            matched_terms=[],
            penalties=[],
            research_value_score=score,
            diversity_features=["dlo"],
        )

    def test_v2_triage_nested_jsonl_selects_default_three_and_caps_four(self) -> None:
        records = [self.ranked_paper(index, score=90 - index).to_dict() for index in range(8)]
        payload = build_triage(records, run_date=RUN_DATE, target_deep_read=3, max_deep_read=4)
        selected = payload["selected_for_deep_read"]
        self.assertEqual(len(selected), 3)
        self.assertLessEqual(payload["counts"]["deep_read"], 4)
        self.assertEqual(selected[0]["arxiv_id"], "2401.00000")
        self.assertEqual(selected[0]["original_index"], 0)

    def test_v2_intake_filter_controls_import_attempts_not_min_new_imports(self) -> None:
        ranked = [self.ranked_paper(index, score=90 - index) for index in range(10)]
        candidates = [(item, item.decision, item.decision) for item in ranked]
        triage = build_triage([item.to_dict() for item in ranked], run_date=RUN_DATE, target_deep_read=3, max_deep_read=4)
        filtered = _filter_import_candidates_by_v2_triage(candidates, triage)
        self.assertEqual(len(filtered), len(candidates))
        self.assertEqual(
            [item.paper.arxiv_id for item, _selection, _original in filtered[:3]],
            [row["arxiv_id"] for row in triage["selected_for_deep_read"]],
        )
        self.assertEqual(
            [item.paper.arxiv_id for item, _selection, _original in filtered[3:]],
            [item.paper.arxiv_id for item in ranked if item.paper.arxiv_id not in {row["arxiv_id"] for row in triage["selected_for_deep_read"]}],
        )

    def test_backfill_ingest_only_stops_before_candidates_and_seed(self) -> None:
        candidates = Path(self.tmp.name) / "candidates.jsonl"
        candidates.write_text("", encoding="utf-8")
        args = Namespace(
            backfill_mode="ingest-only",
            backfill_generate_ideas=False,
            backfill_publish="disabled",
            target_deep_read=10,
            max_deep_read=10,
            idea_mode="template",
            idea_timeout=10,
            gemini_model="test",
            raw_candidate_limit=24,
            min_raw_candidates=24,
            max_generated=24,
            deepseek_timeout=10,
            allow_human_override=False,
        )
        errors: list[str] = []
        result = run_research_seed_v2(
            args=args,
            run_date=RUN_DATE,
            candidates_path=candidates,
            focus_zotero_keys=[],
            errors=errors,
        )
        self.assertEqual(result["status"], "success_ingest_only")
        self.assertFalse((artifact_dir(RUN_DATE) / "raw-candidates.json").exists())
        self.assertEqual(list((Path(self.tmp.name) / "idea_bank" / "seed").glob("*")), [])
        self.assertEqual(errors, [])

    def test_paper_primitives_note_only_downgrades_confidence(self) -> None:
        records = [
            {
                "source_note": "wiki/topics/example.md",
                "source_title": "Example",
                "zotero_key": "ABC123",
                "claim_type": "metric",
                "statement": "Baseline reaches 61 percent success.",
                "source_snippet_type": "claim_statement",
                "source_snippet": "",
            }
        ]
        primitive = build_paper_primitives(records)[0]
        self.assertEqual(primitive["confidence"]["strongest_baseline"], "low")
        self.assertEqual(primitive["confidence"]["actual_baseline_result"], "unusable")

    def test_legacy_structured_field_anchor_downgraded_low(self) -> None:
        records = [
            {
                "source_note": "wiki/topics/example.md",
                "source_title": "Example",
                "zotero_key": "ABC123",
                "claim_type": "method",
                "statement": "A structured legacy field claim.",
                "source_snippet_type": "structured_field",
                "source_snippet": "A structured legacy field claim.",
                "anchor_type": "note_only",
                "evidence_anchor": "wiki/topics/example.md#结构化提取",
                "evidence_section": "结构化提取",
            }
        ]
        primitive = build_paper_primitives(records)[0]
        self.assertEqual(primitive["confidence"]["method_assumption"], "low")
        node = build_nodes([{**primitive, "confidence": {"method_assumption": "high"}}])[0]
        self.assertEqual(node["confidence"], "low")

    def test_note_only_cannot_become_high_confidence(self) -> None:
        primitive = {
            "schema_version": "paper_primitives.v1",
            "paper_key": "ABC123",
            "source_note": "wiki/topics/example.md",
            "source_title": "Example",
            "claims": [
                {
                    "claim_id": "claim-note-only",
                    "claim_type": "central_claim",
                    "statement": "A model summary claim without source anchor.",
                    "evidence_anchor": "wiki/topics/example.md",
                    "anchor_type": "note_only",
                    "anchor": {"anchor_type": "note_only", "evidence_anchor": "wiki/topics/example.md"},
                    "confidence": "high",
                    "confidence_reason": "bad_fixture",
                    "summary_origin": "model_summary",
                    "requires_human_check": False,
                }
            ],
        }
        node = build_nodes([primitive])[0]
        self.assertEqual(node["confidence"], "low")
        self.assertTrue(node["requires_human_check"])

    def test_anchored_baseline_result_can_be_high_confidence(self) -> None:
        records = [
            {
                "source_note": "wiki/topics/example.md",
                "source_title": "Example",
                "zotero_key": "ABC123",
                "claim_type": "metric",
                "statement": "Table 2 reports 71 percent success against the strongest baseline.",
                "source_snippet_type": "section_summary",
                "source_snippet": "Table 2 reports 71 percent success against the strongest baseline.",
                "anchor_type": "table",
                "evidence_anchor": "wiki/topics/example.md#Results",
                "evidence_section": "Results",
            }
        ]
        primitive = build_paper_primitives(records)[0]
        self.assertIn(primitive["confidence"]["actual_baseline_result"], {"medium", "high"})
        self.assertIn(primitive["confidence"]["strongest_baseline"], {"medium", "high"})

    def test_claim_graph_writes_node_and_edge_records(self) -> None:
        records = [
            {
                "source_note": "wiki/topics/example.md",
                "source_title": "Example",
                "zotero_key": "ABC123",
                "claim_type": claim_type,
                "statement": statement,
                "source_snippet_type": "section_summary",
                "source_snippet": statement,
                "anchor_type": "section",
                "evidence_anchor": f"wiki/topics/example.md#{section}",
                "evidence_section": section,
            }
            for claim_type, statement, section in [
                ("problem", "DLO policies fail under latent friction shifts.", "Problem"),
                ("method", "The method assumes stable tactile handoff.", "Method"),
                ("metric", "Results compare against a diffusion policy baseline.", "Results"),
                ("limitation", "Missing ablation on tactile latency.", "Limitations"),
            ]
        ]
        primitive = build_paper_primitives(records)[0]
        graph = build_graph_records([primitive])
        nodes = [item for item in graph if item["record_type"] == "node"]
        edges = [item for item in graph if item["record_type"] == "edge"]
        node_ids = {item["node_id"] for item in nodes}
        self.assertTrue(nodes)
        self.assertTrue(edges)
        self.assertTrue(all(edge["source_node_id"] in node_ids and edge["target_node_id"] in node_ids for edge in edges))

    def test_contradiction_edge_requires_two_existing_nodes(self) -> None:
        contradiction_only = [
            {
                "schema_version": "research_claim_graph.v1",
                "record_type": "node",
                "node_id": "claim-contradiction",
                "paper_key": "ABC123",
                "claim_type": "contradiction",
                "statement": "The reported failure contradicts the assumed interface stability.",
                "confidence": "medium",
                "anchor": {"anchor_type": "section", "evidence_anchor": "wiki/topics/example.md#Limitations"},
                "requires_human_check": False,
            }
        ]
        self.assertFalse([edge for edge in build_edges(contradiction_only) if edge["relation"] == "contradicts"])

        with_method = contradiction_only + [
            {
                **contradiction_only[0],
                "node_id": "claim-method",
                "claim_type": "method_assumption",
                "statement": "The method assumes stable tactile handoff.",
            }
        ]
        self.assertTrue([edge for edge in build_edges(with_method) if edge["relation"] == "contradicts"])

    def test_tension_map_rejects_high_confidence_without_high_anchor(self) -> None:
        node = {"node_id": "claim-1", "confidence": "low"}
        errors = validate_tensions(
            [{"tension_id": "t1", "supporting_nodes": ["claim-1"], "confidence": "high"}],
            {"claim-1"},
            {"claim-1": node},
        )
        self.assertIn("t1:high_confidence_without_high_anchored_node", errors)

    def test_tension_map_rejects_high_confidence_llm_only_tension(self) -> None:
        errors = validate_tensions(
            [{"tension_id": "t-llm", "tension_type": "baseline_contradiction", "supporting_nodes": [], "confidence": "high"}],
            set(),
            {},
        )
        self.assertIn("t-llm:high_confidence_llm_only_tension", errors)

    def test_tension_map_uses_existing_edges(self) -> None:
        nodes = [
            {
                "schema_version": "research_claim_graph.v1",
                "record_type": "node",
                "node_id": "claim-gap",
                "paper_key": "ABC123",
                "claim_type": "missing_ablation",
                "statement": "Missing tactile latency ablation.",
                "confidence": "medium",
                "anchor": {"anchor_type": "section", "evidence_anchor": "wiki/topics/example.md#Limitations"},
                "requires_human_check": False,
            },
            {
                "schema_version": "research_claim_graph.v1",
                "record_type": "node",
                "node_id": "claim-central",
                "paper_key": "ABC123",
                "claim_type": "central_claim",
                "statement": "DLO policies fail under latent shifts.",
                "confidence": "high",
                "anchor": {"anchor_type": "section", "evidence_anchor": "wiki/topics/example.md#Problem"},
                "requires_human_check": False,
            },
        ]
        edge = build_edges(nodes)[0]
        tensions, speculative = build_tensions(nodes, [edge])
        self.assertFalse(speculative)
        self.assertEqual(tensions[0]["tension_type"], "negative_result_opportunity")
        self.assertEqual(tensions[0]["supporting_edges"], [edge["edge_id"]])

    def test_research_agenda_review_apply_cannot_modify_seed(self) -> None:
        idea_bank = Path(self.tmp.name) / "idea_bank"
        seed = idea_bank / "seed" / "legacy-seed"
        seed.mkdir(parents=True, exist_ok=True)
        idea = seed / "idea.md"
        review_log = seed / "review_log.md"
        idea.write_text("idea_state: seed\n- decision_status: seed_pending_review\n", encoding="utf-8")
        review_log.write_text("# Review Log\n", encoding="utf-8")
        old_root = agenda_review.IDEA_BANK_DIR
        old_map = agenda_review.MUTABLE_IDEA_STATE_DIRS
        try:
            agenda_review.IDEA_BANK_DIR = idea_bank
            agenda_review.MUTABLE_IDEA_STATE_DIRS = {
                state: idea_bank / state for state in agenda_review.IDEA_STATES if state != agenda_review.FORMAL_SEED_STATE
            }
            agenda_review.apply_recommendations(
                [
                    {
                        "path": str(seed),
                        "state": "seed",
                        "recommended_state": "rejected",
                        "quality_flags": ["test_flag"],
                    }
                ]
            )
        finally:
            agenda_review.IDEA_BANK_DIR = old_root
            agenda_review.MUTABLE_IDEA_STATE_DIRS = old_map
        self.assertTrue(seed.exists())
        self.assertEqual(idea.read_text(encoding="utf-8"), "idea_state: seed\n- decision_status: seed_pending_review\n")
        self.assertEqual(review_log.read_text(encoding="utf-8"), "# Review Log\n")
        reports = list((idea_bank.parent / "seed-candidates" / "legacy-review").rglob("legacy-seed.json"))
        self.assertEqual(len(reports), 1)

    def test_portfolio_select_keeps_outside_analogy_slot(self) -> None:
        candidates = [
            {**self.candidate(f"cand-{index}"), "lane": "grounded_mechanism", "mechanism_family": f"m{index}"}
            for index in range(4)
        ]
        candidates.append({**self.candidate("cand-outside"), "lane": "outside_analogy", "mechanism_family": "analogy"})
        selected, _ = select_portfolio(candidates, max_selected=4)
        self.assertTrue(any(item.get("lane") == "outside_analogy" for item in selected))

    def test_bom_raw_candidates_json_can_be_portfolio_selected(self) -> None:
        self.write_bom_json(
            artifact_dir(RUN_DATE) / "raw-candidates.json",
            {
                "schema_version": "raw_candidate.v1",
                "run_date": RUN_DATE,
                "candidates": [self.candidate()],
                "generation_config": {},
                "artifact_hashes": {},
            },
        )
        old_argv = sys.argv[:]
        try:
            sys.argv = ["candidate_portfolio_select.py", "--run-date", RUN_DATE, "--max-selected", "8"]
            self.assertEqual(portfolio_select.main(), 0)
        finally:
            sys.argv = old_argv
        selected = read_json(artifact_dir(RUN_DATE) / "selected-candidates.json")
        self.assertEqual(selected["selected"][0]["candidate_id"], "cand-alpha")

    def test_bom_invalid_schema_still_fails_validation(self) -> None:
        path = artifact_dir(RUN_DATE) / "raw-candidates.json"
        self.write_bom_json(path, {"schema_version": "wrong.v1", "run_date": RUN_DATE, "candidates": []})
        errors = validate_json_file(path, "raw_candidate.v1")
        self.assertTrue(any(error.startswith("schema_version_mismatch") for error in errors))

    def test_invalid_nested_codex_action_fails_schema(self) -> None:
        path = artifact_dir(RUN_DATE) / "codex-execution-review.json"
        self.write_bom_json(
            path,
            {
                "schema_version": "codex_execution_review.v1",
                "run_date": RUN_DATE,
                "status": "success",
                "provider_status": {"provider": "codex", "provider_backed": True, "mode": "codex-cli"},
                "reviews": [{**self.provider_codex_payload()["reviews"][0], "action": "accept"}],
            },
        )
        errors = validate_json_file(path, "codex_execution_review.v1")
        self.assertTrue(any("action" in error for error in errors))

    def test_invalid_deepseek_survivability_label_fails_schema(self) -> None:
        path = artifact_dir(RUN_DATE) / "deepseek-review.json"
        payload = self.provider_deepseek_payload()
        payload["schema_version"] = "deepseek_review.v1"
        payload["run_date"] = RUN_DATE
        payload["status"] = "success"
        payload["reviews"][0]["survivability_label"] = "maybe"  # type: ignore[index]
        self.write_bom_json(path, payload)
        errors = validate_json_file(path, "deepseek_review.v1")
        self.assertTrue(any("survivability_label" in error for error in errors))

    def test_missing_nested_candidate_id_fails_schema(self) -> None:
        path = artifact_dir(RUN_DATE) / "novelty-scan.json"
        self.write_bom_json(
            path,
            {
                "schema_version": "novelty_scan.v1",
                "run_date": RUN_DATE,
                "status": "completed_local_only",
                "scans": [
                    {
                        "novelty_classification": "likely_open",
                        "promotion_allowed": True,
                        "formal_promotion_allowed": False,
                        "verification_scope": "local_only",
                        "external_providers_used": [],
                        "nearest_works": [],
                        "duplicate_guard": {},
                    }
                ],
            },
        )
        errors = validate_json_file(path, "novelty_scan.v1")
        self.assertTrue(any("candidate_id" in error for error in errors))

    def test_invalid_novelty_classification_fails_schema(self) -> None:
        path = artifact_dir(RUN_DATE) / "novelty-scan.json"
        self.write_bom_json(
            path,
            {
                "schema_version": "novelty_scan.v1",
                "run_date": RUN_DATE,
                "status": "completed_local_only",
                "scans": [
                    {
                        "candidate_id": "cand-alpha",
                        "novelty_classification": "new",
                        "promotion_allowed": True,
                        "formal_promotion_allowed": False,
                        "verification_scope": "local_only",
                        "external_providers_used": [],
                        "nearest_works": [],
                        "duplicate_guard": {},
                    }
                ],
            },
        )
        errors = validate_json_file(path, "novelty_scan.v1")
        self.assertTrue(any("novelty_classification" in error for error in errors))

    def test_invalid_human_override_cannot_be_consumed(self) -> None:
        cid = self.write_review_artifacts(novelty="unknown")
        override_dir = Path(self.tmp.name) / "overrides" / "human-overrides" / RUN_DATE
        self.write_bom_json(
            override_dir / f"{cid}.json",
            {
                "schema_version": "human_override.v1",
                "candidate_id": cid,
                "run_id": RUN_DATE,
                "override_type": "bad_override",
                "reviewer": "human",
                "created_at": "2099-01-02T00:00:00Z",
                "reason": "test",
                "risk_acceptance": "test",
                "expires_after_days": 7,
                "allowed_publish_target": "seed",
                "cannot_weaken": [],
            },
        )
        payload = decide(run_date=RUN_DATE, allow_human_override=True)
        self.assertEqual(payload["human_overrides_consumed"], [])
        self.assertTrue(payload["human_override_errors"])

    def test_bom_deepseek_provider_json_can_be_loaded_but_invalid_still_blocks(self) -> None:
        item = self.candidate()
        write_run_artifact(
            RUN_DATE,
            "selected-candidates.json",
            {
                "schema_version": "selected_candidates.v1",
                "run_date": RUN_DATE,
                "selected": [item],
                "rejected": [],
                "selection_rules": {},
                "artifact_hashes": {},
            },
            state="portfolio_selected",
        )
        provider_path = Path(self.tmp.name) / "provider-deepseek.json"
        self.write_bom_json(provider_path, self.provider_deepseek_payload(item))
        old_argv = sys.argv[:]
        try:
            sys.argv = ["deepseek_scientific_review.py", "--run-date", RUN_DATE, "--provider-review-json", str(provider_path)]
            self.assertEqual(deepseek_review.main(), 0)
        finally:
            sys.argv = old_argv
        payload = read_json(artifact_dir(RUN_DATE) / "deepseek-review.json")
        self.assertEqual(payload["status"], "success")
        self.assertTrue(payload["provider_status"]["provider_backed"])

        invalid_path = Path(self.tmp.name) / "provider-deepseek-invalid.json"
        self.write_bom_json(
            invalid_path,
            {
                "provider_status": {"provider": "deepseek", "provider_backed": True},
                "reviews": [{"candidate_id": "cand-alpha", "status": "success"}],
            },
        )
        invalid_payload = deepseek_review._load_provider_payload(str(invalid_path))
        blocked, exit_code = build_deepseek_payload([item], run_date=RUN_DATE, provider_payload=invalid_payload)
        self.assertEqual(exit_code, 2)
        self.assertEqual(blocked["status"], "partial_provider_invalid")
        self.assertFalse(blocked["provider_status"]["provider_backed"])

    def test_deepseek_empty_selection_is_success_noop(self) -> None:
        payload, exit_code = build_deepseek_payload([], run_date=RUN_DATE, provider_payload={})
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "success_empty_selection")
        self.assertEqual(payload["reviews"], [])
        self.assertFalse(payload["provider_status"]["provider_backed"])
        self.assertEqual(payload["provider_status"]["mode"], "no_selection")
        self.assertEqual(validate_payload(payload, "deepseek_review.v1"), [])

    def test_deepseek_opencode_provider_success_with_mocked_cli(self) -> None:
        output = {
            "schema_version": "deepseek_review.v1",
            "run_date": RUN_DATE,
            "status": "success",
            "reviews": self.provider_deepseek_payload()["reviews"],
        }
        with patch.object(
            deepseek_review,
            "run_opencode_cli",
            return_value={"exit_code": 0, "timed_out": False, "clean_output": json.dumps(output), "effective_model": "deepseek/deepseek-v4-pro", "event_count": 1},
        ):
            provider_payload = deepseek_review._opencode_provider_payload([self.candidate()], run_date=RUN_DATE, model="deepseek/deepseek-v4-pro", timeout_sec=5)
        payload, exit_code = build_deepseek_payload([self.candidate()], run_date=RUN_DATE, provider_payload=provider_payload)
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "success")
        self.assertTrue(payload["provider_status"]["provider_backed"])

    def test_deepseek_openai_compatible_provider_success_with_mocked_api(self) -> None:
        output = {
            "schema_version": "deepseek_review.v1",
            "run_date": RUN_DATE,
            "status": "success",
            "reviews": self.provider_deepseek_payload()["reviews"],
        }
        with patch.object(
            deepseek_review,
            "_openai_compatible_chat_completion",
            return_value={
                "exit_code": 0,
                "timed_out": False,
                "clean_output": json.dumps(output),
                "effective_model": "deepseek/deepseek-v4-pro",
                "base_url_origin": "https://example.test",
                "response_model": "deepseek/deepseek-v4-pro",
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            },
        ):
            provider_payload = deepseek_review._openai_compatible_provider_payload([self.candidate()], run_date=RUN_DATE, model="abrdns/deepseek-v4-pro", timeout_sec=5)
        payload, exit_code = build_deepseek_payload([self.candidate()], run_date=RUN_DATE, provider_payload=provider_payload)
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "success")
        self.assertTrue(payload["provider_status"]["provider_backed"])
        self.assertEqual(payload["provider_status"]["mode"], "openai-compatible")
        self.assertEqual(payload["provider_status"]["response_model"], "deepseek/deepseek-v4-pro")

    def test_deepseek_openai_compatible_missing_config_fails_closed(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_OPENAI_BASE_URL": "", "DEEPSEEK_OPENAI_API_KEY": ""}, clear=False), patch.object(
            deepseek_review,
            "_global_opencode_config_paths",
            return_value=[],
        ):
            result = deepseek_review._openai_compatible_chat_completion("{}", model="abrdns/deepseek-v4-pro", timeout_sec=5)

        self.assertEqual(result["exit_code"], 1)
        self.assertIn("openai_compatible_config_missing", result["error"])

    def test_deepseek_opencode_single_review_object_is_wrapped(self) -> None:
        output = dict(self.provider_deepseek_payload()["reviews"][0])
        output.pop("candidate_title", None)
        with patch.object(
            deepseek_review,
            "run_opencode_cli",
            return_value={"exit_code": 0, "timed_out": False, "clean_output": json.dumps(output), "effective_model": "deepseek/deepseek-v4-pro", "event_count": 1},
        ):
            provider_payload = deepseek_review._opencode_provider_payload([self.candidate()], run_date=RUN_DATE, model="deepseek/deepseek-v4-pro", timeout_sec=5)
        payload, exit_code = build_deepseek_payload([self.candidate()], run_date=RUN_DATE, provider_payload=provider_payload)
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["reviews"][0]["candidate_title"], self.candidate()["title"])

    def test_deepseek_single_candidate_batch_hydrates_wrong_candidate_id(self) -> None:
        output = dict(self.provider_deepseek_payload()["reviews"][0])
        output["candidate_id"] = "wrong-short-id"
        with patch.object(
            deepseek_review,
            "run_opencode_cli",
            return_value={
                "exit_code": 0,
                "timed_out": False,
                "clean_output": json.dumps({"schema_version": "deepseek_review.v1", "run_date": RUN_DATE, "status": "success", "reviews": [output]}),
                "effective_model": "deepseek/deepseek-v4-pro",
                "event_count": 1,
            },
        ):
            provider_payload = deepseek_review._opencode_provider_payload([self.candidate()], run_date=RUN_DATE, model="deepseek/deepseek-v4-pro", timeout_sec=5)
        payload, exit_code = build_deepseek_payload([self.candidate()], run_date=RUN_DATE, provider_payload=provider_payload)
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["reviews"][0]["candidate_id"], "cand-alpha")
        self.assertTrue(payload["provider_status"]["candidate_id_hydrated_from_single_candidate_batch"])

    def test_deepseek_single_candidate_batch_hydrates_review_without_candidate_id(self) -> None:
        output = dict(self.provider_deepseek_payload()["reviews"][0])
        output.pop("candidate_id", None)
        with patch.object(
            deepseek_review,
            "run_opencode_cli",
            return_value={
                "exit_code": 0,
                "timed_out": False,
                "clean_output": json.dumps(output),
                "effective_model": "deepseek/deepseek-v4-pro",
                "event_count": 1,
            },
        ):
            provider_payload = deepseek_review._opencode_provider_payload([self.candidate()], run_date=RUN_DATE, model="deepseek/deepseek-v4-pro", timeout_sec=5)
        payload, exit_code = build_deepseek_payload([self.candidate()], run_date=RUN_DATE, provider_payload=provider_payload)
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["reviews"][0]["candidate_id"], "cand-alpha")
        self.assertTrue(payload["provider_status"]["candidate_id_hydrated_from_single_candidate_batch"])

    def test_deepseek_single_candidate_batch_normalizes_review_dict(self) -> None:
        output = dict(self.provider_deepseek_payload()["reviews"][0])
        output.pop("candidate_id", None)
        with patch.object(
            deepseek_review,
            "run_opencode_cli",
            return_value={
                "exit_code": 0,
                "timed_out": False,
                "clean_output": json.dumps({"review": output}),
                "effective_model": "deepseek/deepseek-v4-pro",
                "event_count": 1,
            },
        ):
            provider_payload = deepseek_review._opencode_provider_payload([self.candidate()], run_date=RUN_DATE, model="deepseek/deepseek-v4-pro", timeout_sec=5)
        payload, exit_code = build_deepseek_payload([self.candidate()], run_date=RUN_DATE, provider_payload=provider_payload)
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["reviews"][0]["candidate_id"], "cand-alpha")

    def test_deepseek_single_candidate_batch_hydrates_missing_review_status(self) -> None:
        output = dict(self.provider_deepseek_payload()["reviews"][0])
        output.pop("status", None)
        with patch.object(
            deepseek_review,
            "run_opencode_cli",
            return_value={
                "exit_code": 0,
                "timed_out": False,
                "clean_output": json.dumps({"reviews": [output]}),
                "effective_model": "deepseek/deepseek-v4-pro",
                "event_count": 1,
            },
        ):
            provider_payload = deepseek_review._opencode_provider_payload([self.candidate()], run_date=RUN_DATE, model="deepseek/deepseek-v4-pro", timeout_sec=5)
        payload, exit_code = build_deepseek_payload([self.candidate()], run_date=RUN_DATE, provider_payload=provider_payload)
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["reviews"][0]["status"], "success")
        self.assertTrue(payload["provider_status"]["review_status_hydrated_from_single_candidate_batch"])

    def test_deepseek_provider_list_enum_field_fails_closed_without_type_error(self) -> None:
        provider = self.provider_deepseek_payload()
        provider["reviews"][0]["a_plus_b_risk"] = ["medium"]  # type: ignore[index]

        payload, exit_code = build_deepseek_payload([self.candidate()], run_date=RUN_DATE, provider_payload=provider)

        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["status"], "partial_provider_invalid")
        self.assertFalse(payload["provider_status"]["provider_backed"])
        self.assertEqual(payload["reviews"][0]["a_plus_b_risk"], "fatal")
        self.assertIn(
            "provider_review_invalid_field_type:cand-alpha:a_plus_b_risk:list",
            payload["provider_status"]["validation_errors"],
        )

    def test_opencode_redaction_does_not_mangle_task_candidate_id(self) -> None:
        candidate_id_value = "cand-observation-prior-sparsification-task-aligne-bbef39186e"
        fake_secret = "sk-" + "abcdefghijklmnopqrstuvwxyz123456"
        self.assertEqual(opencode_cli_adapter._redact(candidate_id_value), candidate_id_value)
        self.assertEqual(opencode_cli_adapter._redact(fake_secret), "[REDACTED]")

    def test_opencode_resolver_prefers_bundled_exe_over_cmd_shim(self) -> None:
        def fake_which(name: str) -> str | None:
            if name == "opencode":
                return r"C:\nvm4w\nodejs\opencode.CMD"
            if name == "opencode.exe":
                return None
            return None

        with patch.object(opencode_cli_adapter.shutil, "which", side_effect=fake_which), patch.object(
            opencode_cli_adapter.os.path,
            "exists",
            side_effect=lambda path: str(path).endswith(r"node_modules\opencode-ai\bin\opencode.exe"),
        ):
            self.assertEqual(
                opencode_cli_adapter._resolve_opencode_path(),
                r"C:\nvm4w\nodejs\node_modules\opencode-ai\bin\opencode.exe",
            )

    def test_opencode_writes_json_review_agent_config(self) -> None:
        workspace = Path(self.tmp.name) / "opencode-work"
        workspace.mkdir()
        config_path = opencode_cli_adapter._write_json_review_agent_config(str(workspace), "deepseek/deepseek-v4-pro")
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.assertEqual(config["default_agent"], opencode_cli_adapter.JSON_REVIEW_AGENT_NAME)
        agent = config["agent"][opencode_cli_adapter.JSON_REVIEW_AGENT_NAME]
        self.assertEqual(agent["model"], "deepseek-v4-pro")
        self.assertIn("RFC 8259 JSON only", agent["prompt"])
        self.assertTrue(agent["tools"])
        self.assertTrue(all(value is False for value in agent["tools"].values()))

    def test_opencode_extracts_nested_assistant_message_text(self) -> None:
        stdout = json.dumps(
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": '{"reviews": [{"candidate_id": "cand-alpha"}]}'}],
                },
            }
        )
        text, event_count = opencode_cli_adapter._extract_text_events(stdout)
        self.assertEqual(event_count, 1)
        self.assertIn('"candidate_id": "cand-alpha"', text)

    def test_opencode_extract_ignores_nested_user_message_text(self) -> None:
        stdout = "\n".join(
            [
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": '{"selected_candidates": ["prompt"]}'}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": '{"reviews": [{"candidate_id": "cand-alpha"}]}'}],
                        },
                    }
                ),
            ]
        )
        text, event_count = opencode_cli_adapter._extract_text_events(stdout)
        self.assertEqual(event_count, 2)
        self.assertNotIn("selected_candidates", text)
        self.assertIn('"candidate_id": "cand-alpha"', text)

    def test_opencode_extract_text_ignores_step_only_events(self) -> None:
        stdout = "\n".join(
            [
                json.dumps({"type": "step_start", "part": {"type": "step-start"}}),
                json.dumps({"type": "tool_use", "part": {"type": "tool", "tool": "read"}}),
            ]
        )
        text, event_count = opencode_cli_adapter._extract_text_events(stdout)
        self.assertEqual(event_count, 2)
        self.assertEqual(text, "")

    def test_opencode_inline_limit_routes_single_candidate_review_to_file_mode(self) -> None:
        prompt = deepseek_review._render_opencode_prompt([self.candidate()], run_date=RUN_DATE)
        self.assertGreater(len(prompt), opencode_cli_adapter.INLINE_PROMPT_CHAR_LIMIT)

    def test_deepseek_extract_json_repairs_unquoted_keys(self) -> None:
        payload = deepseek_review._extract_json_object(
            '{status: "success", reviews: [{candidate_id: "cand-alpha", status: "success"}]}'
        )
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["reviews"][0]["candidate_id"], "cand-alpha")

    def test_deepseek_first_opencode_prompt_uses_compact_view_when_requested(self) -> None:
        prompt = deepseek_review._render_opencode_prompt([self.candidate()], run_date=RUN_DATE, retry_compact=True)
        request = json.loads(prompt)
        self.assertEqual(request["input_mode"], "compact_provider_review")
        self.assertNotIn("evidence_excerpt", request["selected_candidates"][0])
        self.assertEqual(request["exact_candidate_ids"], ["cand-alpha"])
        self.assertEqual(request["required_output_shape"]["reviews"][0]["candidate_id"], "cand-alpha")
        self.assertEqual(request["required_output_shape"]["reviews"][0]["status"], "success")

    def test_deepseek_retry_prompt_uses_error_kind_not_event_payload(self) -> None:
        event = json.dumps({"type": "tool_use", "part": {"type": "tool", "tool": "read"}})
        prompt = deepseek_review._render_opencode_prompt(
            [self.candidate()],
            run_date=RUN_DATE,
            retry_reason="opencode_invalid_json",
            previous_output=event,
            retry_details=["provider_output_json_object_not_found"],
            retry_compact=True,
        )
        request = json.loads(prompt)
        self.assertEqual(request["previous_invalid_output_kind"], "opencode_event:tool_use")
        self.assertEqual(request["previous_failure_details"], ["provider_output_json_object_not_found"])
        self.assertNotIn("previous_invalid_output", request)

    def test_deepseek_extract_json_skips_opencode_event_objects(self) -> None:
        event = json.dumps(
            {
                "type": "step_start",
                "timestamp": 1779451492067,
                "sessionID": "ses-test",
                "part": {"type": "step-start"},
            }
        )
        review = json.dumps(
            {
                "schema_version": "deepseek_review.v1",
                "run_date": RUN_DATE,
                "status": "success",
                "reviews": [{"candidate_id": "cand-alpha", "status": "success"}],
            }
        )
        payload = deepseek_review._extract_json_object(event + "\n" + review)
        self.assertEqual(payload["reviews"][0]["candidate_id"], "cand-alpha")

    def test_deepseek_extract_json_rejects_tool_event_only_output(self) -> None:
        event = json.dumps(
            {
                "type": "tool_use",
                "timestamp": 1779451499377,
                "sessionID": "ses-test",
                "part": {
                    "type": "tool",
                    "tool": "read",
                    "state": {"status": "error", "input": {"filePath": "C:/Temp/request.json"}},
                },
            }
        )
        with self.assertRaises(ValueError):
            deepseek_review._extract_json_object(event)

    def test_opencode_reads_tool_written_json_output(self) -> None:
        output_dir = Path(self.tmp.name) / "opencode-work" / ".sisyphus"
        output_dir.mkdir(parents=True)
        output_path = output_dir / "review-output.json"
        output_path.write_text(
            json.dumps({"reviews": [{"candidate_id": "cand-alpha", "status": "success"}]}),
            encoding="utf-8",
        )
        text = opencode_cli_adapter._read_tool_json_output(str(output_dir.parent))
        self.assertIn('"candidate_id": "cand-alpha"', text)

    def test_opencode_tool_output_reader_ignores_agent_config(self) -> None:
        workspace = Path(self.tmp.name) / "opencode-work"
        workspace.mkdir()
        (workspace / "opencode.json").write_text(
            json.dumps({"agent": {"research-json-review": {"prompt": "not a review"}}}),
            encoding="utf-8",
        )
        text = opencode_cli_adapter._read_tool_json_output(str(workspace))
        self.assertEqual(text, "")

    def test_deepseek_opencode_batches_multi_candidate_with_mocked_cli(self) -> None:
        candidates = [self.candidate("cand-alpha"), self.candidate("cand-beta")]
        call_sizes: list[int] = []

        def fake_opencode(prompt: str, **_kwargs: object) -> dict[str, object]:
            request = json.loads(prompt)
            selected = request["selected_candidates"]
            call_sizes.append(len(selected))
            reviews = []
            for item in selected:
                reviews.append(
                    {
                        "candidate_id": item["candidate_id"],
                        "candidate_title": item["title"],
                        "status": "success",
                        "novelty_attack": "nearest work reviewed",
                        "baseline_attack": "baseline checked",
                        "mechanism_attack": "mechanism checked",
                        "evaluation_attack": "evaluation checked",
                        "scope_attack": "scope checked",
                        "a_plus_b_risk": "low",
                        "fatal_flaw": "",
                        "rescue_mutation": "",
                        "survivability_label": "survives",
                        "allowed_next_stage": "novelty_scan",
                    }
                )
            return {
                "exit_code": 0,
                "timed_out": False,
                "clean_output": json.dumps({"schema_version": "deepseek_review.v1", "run_date": RUN_DATE, "status": "success", "reviews": reviews}),
                "effective_model": "deepseek/deepseek-v4-pro",
                "event_count": 1,
            }

        with patch.object(deepseek_review, "run_opencode_cli", side_effect=fake_opencode):
            provider_payload = deepseek_review._opencode_provider_payload(
                candidates,
                run_date=RUN_DATE,
                model="deepseek/deepseek-v4-pro",
                timeout_sec=5,
                batch_size=1,
            )
        payload, exit_code = build_deepseek_payload(candidates, run_date=RUN_DATE, provider_payload=provider_payload)
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "success")
        self.assertTrue(payload["provider_status"]["provider_backed"])
        self.assertEqual(payload["provider_status"]["batch_count"], 2)
        self.assertEqual(call_sizes, [1, 1])
        self.assertEqual({review["candidate_id"] for review in payload["reviews"]}, {"cand-alpha", "cand-beta"})

    def test_deepseek_opencode_reuses_cached_successful_batch(self) -> None:
        candidates = [self.candidate("cand-alpha"), self.candidate("cand-beta")]
        cached_review = {
            "candidate_id": "cand-alpha",
            "candidate_title": "Anchored Contact Failure Benchmark",
            "status": "success",
            "novelty_attack": "nearest work reviewed",
            "baseline_attack": "baseline checked",
            "mechanism_attack": "mechanism checked",
            "evaluation_attack": "evaluation checked",
            "scope_attack": "scope checked",
            "a_plus_b_risk": "low",
            "fatal_flaw": "",
            "rescue_mutation": "",
            "survivability_label": "survives",
            "allowed_next_stage": "novelty_scan",
        }
        deepseek_review._write_opencode_batch_cache(
            {
                "schema_version": "deepseek_review.v1",
                "run_date": RUN_DATE,
                "status": "success",
                "provider_status": {
                    "provider": "deepseek",
                    "provider_backed": True,
                    "mode": "opencode",
                    "batch_index": 1,
                    "batch_size": 1,
                    "batch_candidate_ids": ["cand-alpha"],
                },
                "reviews": [cached_review],
            },
            run_date=RUN_DATE,
            batch_index=1,
        )
        calls = 0

        def fake_opencode(prompt: str, **_kwargs: object) -> dict[str, object]:
            nonlocal calls
            calls += 1
            request = json.loads(prompt)
            selected = request["selected_candidates"]
            self.assertEqual([item["candidate_id"] for item in selected], ["cand-beta"])
            review = {
                **cached_review,
                "candidate_id": "cand-beta",
            }
            return {
                "exit_code": 0,
                "timed_out": False,
                "clean_output": json.dumps({"schema_version": "deepseek_review.v1", "run_date": RUN_DATE, "status": "success", "reviews": [review]}),
                "effective_model": "deepseek/deepseek-v4-pro",
                "event_count": 1,
            }

        with patch.object(deepseek_review, "run_opencode_cli", side_effect=fake_opencode):
            provider_payload = deepseek_review._opencode_provider_payload(
                candidates,
                run_date=RUN_DATE,
                model="deepseek/deepseek-v4-pro",
                timeout_sec=5,
                batch_size=1,
            )
        payload, exit_code = build_deepseek_payload(candidates, run_date=RUN_DATE, provider_payload=provider_payload)
        self.assertEqual(exit_code, 0)
        self.assertEqual(calls, 1)
        self.assertEqual(payload["status"], "success")
        self.assertTrue(payload["provider_status"]["batch_statuses"][0]["cache_hit"])
        self.assertEqual({review["candidate_id"] for review in payload["reviews"]}, {"cand-alpha", "cand-beta"})

    def test_deepseek_opencode_batch_failure_fails_closed(self) -> None:
        candidates = [self.candidate("cand-alpha"), self.candidate("cand-beta")]
        calls = 0

        def fake_opencode(prompt: str, **_kwargs: object) -> dict[str, object]:
            nonlocal calls
            calls += 1
            request = json.loads(prompt)
            selected = request["selected_candidates"]
            if calls == 2:
                return {"exit_code": 1, "timed_out": False, "clean_output": "", "effective_model": "deepseek/deepseek-v4-pro", "event_count": 1, "error": "nonzero_exit:1"}
            item = selected[0]
            review = {
                "candidate_id": item["candidate_id"],
                "candidate_title": item["title"],
                "status": "success",
                "novelty_attack": "nearest work reviewed",
                "baseline_attack": "baseline checked",
                "mechanism_attack": "mechanism checked",
                "evaluation_attack": "evaluation checked",
                "scope_attack": "scope checked",
                "a_plus_b_risk": "low",
                "fatal_flaw": "",
                "rescue_mutation": "",
                "survivability_label": "survives",
                "allowed_next_stage": "novelty_scan",
            }
            return {
                "exit_code": 0,
                "timed_out": False,
                "clean_output": json.dumps({"schema_version": "deepseek_review.v1", "run_date": RUN_DATE, "status": "success", "reviews": [review]}),
                "effective_model": "deepseek/deepseek-v4-pro",
                "event_count": 1,
            }

        with patch.object(deepseek_review, "run_opencode_cli", side_effect=fake_opencode):
            provider_payload = deepseek_review._opencode_provider_payload(
                candidates,
                run_date=RUN_DATE,
                model="deepseek/deepseek-v4-pro",
                timeout_sec=5,
                batch_size=1,
                retries=0,
            )
        payload, exit_code = build_deepseek_payload(candidates, run_date=RUN_DATE, provider_payload=provider_payload)
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "partial_provider_unavailable")
        self.assertFalse(payload["provider_status"]["provider_backed"])
        self.assertTrue(payload["provider_status"]["candidate_level_fail_closed"])
        self.assertEqual(payload["provider_status"]["provider_backed_success_count"], 1)
        self.assertEqual(payload["provider_status"]["provider_fallback_count"], 1)
        self.assertIn("nonzero_exit:1", payload["provider_status"]["provider_error"])
        by_id = {review["candidate_id"]: review for review in payload["reviews"]}
        self.assertEqual(by_id["cand-alpha"]["status"], "success")
        self.assertEqual(by_id["cand-beta"]["status"], "failed_fallback_only")
        self.assertEqual(by_id["cand-beta"]["survivability_label"], "reject_fatal")
        self.assertEqual(by_id["cand-beta"]["allowed_next_stage"], "stop")

    def test_deepseek_opencode_batch_failure_marks_each_candidate_failed_fallback(self) -> None:
        candidates = [self.candidate("cand-alpha"), self.candidate("cand-beta")]
        calls = 0

        def fake_opencode(_prompt: str, **_kwargs: object) -> dict[str, object]:
            nonlocal calls
            calls += 1
            return {
                "exit_code": 1,
                "timed_out": False,
                "clean_output": "",
                "effective_model": "deepseek/deepseek-v4-pro",
                "event_count": 1,
                "error": "nonzero_exit:1",
            }

        with patch.object(deepseek_review, "run_opencode_cli", side_effect=fake_opencode):
            provider_payload = deepseek_review._opencode_provider_payload(
                candidates,
                run_date=RUN_DATE,
                model="deepseek/deepseek-v4-pro",
                timeout_sec=5,
                batch_size=1,
                retries=0,
            )
        payload, exit_code = build_deepseek_payload(candidates, run_date=RUN_DATE, provider_payload=provider_payload)
        self.assertEqual(calls, 1)
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "partial_provider_unavailable")
        self.assertTrue(payload["provider_status"]["candidate_level_fail_closed"])
        self.assertEqual(payload["provider_status"]["provider_backed_success_count"], 0)
        self.assertEqual(payload["provider_status"]["provider_fallback_count"], 2)
        self.assertTrue(payload["provider_status"]["batch_statuses"][0]["batch_fail_fast_after_provider_failure"])
        self.assertEqual(payload["provider_status"]["batch_statuses"][0]["unattempted_batch_count"], 1)
        self.assertEqual({review["status"] for review in payload["reviews"]}, {"failed_fallback_only"})
        self.assertEqual({review["allowed_next_stage"] for review in payload["reviews"]}, {"stop"})

    def test_deepseek_opencode_invalid_json_retries_and_succeeds(self) -> None:
        calls = 0
        item = self.candidate()

        def fake_opencode(prompt: str, **_kwargs: object) -> dict[str, object]:
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"exit_code": 0, "timed_out": False, "clean_output": "plain prose without json", "effective_model": "deepseek/deepseek-v4-pro", "event_count": 1}
            request = json.loads(prompt)
            selected = request["selected_candidates"]
            self.assertEqual(request["retry_reason"], "opencode_invalid_json")
            self.assertEqual(request["input_mode"], "compact_retry")
            self.assertEqual(request["previous_invalid_output_kind"], "invalid_json")
            self.assertNotIn("previous_invalid_output", request)
            self.assertNotIn("evidence_excerpt", selected[0])
            review = {
                "candidate_id": selected[0]["candidate_id"],
                "status": "success",
                "novelty_attack": "nearest work reviewed",
                "baseline_attack": "baseline checked",
                "mechanism_attack": "mechanism checked",
                "evaluation_attack": "evaluation checked",
                "scope_attack": "scope checked",
                "a_plus_b_risk": "low",
                "fatal_flaw": "",
                "rescue_mutation": "",
                "survivability_label": "survives",
                "allowed_next_stage": "novelty_scan",
            }
            return {
                "exit_code": 0,
                "timed_out": False,
                "clean_output": json.dumps({"schema_version": "deepseek_review.v1", "run_date": RUN_DATE, "status": "success", "reviews": [review]}),
                "effective_model": "deepseek/deepseek-v4-pro",
                "event_count": 1,
            }

        with patch.object(deepseek_review, "run_opencode_cli", side_effect=fake_opencode):
            provider_payload = deepseek_review._opencode_provider_payload(
                [item],
                run_date=RUN_DATE,
                model="deepseek/deepseek-v4-pro",
                timeout_sec=5,
                batch_size=1,
                retries=1,
            )
        payload, exit_code = build_deepseek_payload([item], run_date=RUN_DATE, provider_payload=provider_payload)
        self.assertEqual(exit_code, 0)
        self.assertEqual(calls, 2)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["reviews"][0]["candidate_title"], item["title"])
        self.assertTrue(payload["provider_status"]["provider_backed"])
        self.assertEqual(len(payload["provider_status"]["attempt_statuses"]), 2)

    def test_deepseek_opencode_invalid_review_payload_retries_and_succeeds(self) -> None:
        calls = 0
        item = self.candidate()

        def fake_opencode(prompt: str, **_kwargs: object) -> dict[str, object]:
            nonlocal calls
            calls += 1
            if calls == 1:
                return {
                    "exit_code": 0,
                    "timed_out": False,
                    "clean_output": json.dumps({"schema_version": "deepseek_review.v1", "run_date": RUN_DATE, "status": "success", "reviews": []}),
                    "effective_model": "deepseek/deepseek-v4-pro",
                    "event_count": 1,
                }
            request = json.loads(prompt)
            selected = request["selected_candidates"]
            self.assertEqual(request["retry_reason"], "opencode_invalid_review_payload")
            self.assertEqual(request["input_mode"], "compact_retry")
            self.assertEqual(request["previous_invalid_output_kind"], "json_not_matching_review_schema")
            self.assertNotIn("previous_invalid_output", request)
            self.assertNotIn("evidence_excerpt", selected[0])
            review = {
                "candidate_id": selected[0]["candidate_id"],
                "candidate_title": selected[0]["title"],
                "status": "success",
                "novelty_attack": "nearest work reviewed",
                "baseline_attack": "baseline checked",
                "mechanism_attack": "mechanism checked",
                "evaluation_attack": "evaluation checked",
                "scope_attack": "scope checked",
                "a_plus_b_risk": "low",
                "fatal_flaw": "",
                "rescue_mutation": "",
                "survivability_label": "survives",
                "allowed_next_stage": "novelty_scan",
            }
            return {
                "exit_code": 0,
                "timed_out": False,
                "clean_output": json.dumps({"schema_version": "deepseek_review.v1", "run_date": RUN_DATE, "status": "success", "reviews": [review]}),
                "effective_model": "deepseek/deepseek-v4-pro",
                "event_count": 1,
            }

        with patch.object(deepseek_review, "run_opencode_cli", side_effect=fake_opencode):
            provider_payload = deepseek_review._opencode_provider_payload(
                [item],
                run_date=RUN_DATE,
                model="deepseek/deepseek-v4-pro",
                timeout_sec=5,
                batch_size=1,
                retries=1,
            )
        payload, exit_code = build_deepseek_payload([item], run_date=RUN_DATE, provider_payload=provider_payload)
        self.assertEqual(exit_code, 0)
        self.assertEqual(calls, 2)
        self.assertEqual(payload["status"], "success")
        self.assertTrue(payload["provider_status"]["provider_backed"])
        self.assertEqual(len(payload["provider_status"]["attempt_statuses"]), 2)

    def test_deepseek_opencode_invalid_json_fails_closed(self) -> None:
        with patch.object(
            deepseek_review,
            "run_opencode_cli",
            return_value={"exit_code": 0, "timed_out": False, "clean_output": "not json", "effective_model": "deepseek/deepseek-v4-pro", "event_count": 1},
        ):
            provider_payload = deepseek_review._opencode_provider_payload([self.candidate()], run_date=RUN_DATE, model="deepseek/deepseek-v4-pro", timeout_sec=5, retries=0)
        payload, exit_code = build_deepseek_payload([self.candidate()], run_date=RUN_DATE, provider_payload=provider_payload)
        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["status"], "partial_provider_invalid")
        self.assertFalse(payload["provider_status"]["provider_backed"])

    def test_deepseek_opencode_provider_event_error_fails_closed_as_unavailable(self) -> None:
        with patch.object(
            deepseek_review,
            "run_opencode_cli",
            return_value={
                "exit_code": 0,
                "timed_out": False,
                "clean_output": "",
                "effective_model": "abrdns/DeepSeek-V4-Pro-think",
                "event_count": 1,
                "error": "provider_event_error:APIError:Not Found:status=404",
                "event_errors": ["provider_event_error:APIError:Not Found:status=404"],
                "raw_stdout": '{"type":"error"}',
                "raw_stderr": "",
            },
        ):
            provider_payload = deepseek_review._opencode_provider_payload(
                [self.candidate()],
                run_date=RUN_DATE,
                model="abrdns/DeepSeek-V4-Pro-think",
                timeout_sec=5,
                retries=0,
            )
        payload, exit_code = build_deepseek_payload([self.candidate()], run_date=RUN_DATE, provider_payload=provider_payload)
        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["status"], "partial_provider_unavailable")
        self.assertFalse(payload["provider_status"]["provider_backed"])
        self.assertIn("provider_event_error:APIError:Not Found:status=404", payload["provider_status"]["provider_error"])

    def test_deepseek_opencode_auth_error_can_use_explicit_fallback_model(self) -> None:
        calls: list[str] = []
        output = {
            "schema_version": "deepseek_review.v1",
            "run_date": RUN_DATE,
            "status": "success",
            "reviews": [
                {
                    "candidate_id": "cand-alpha",
                    "candidate_title": "Anchored Contact Failure Benchmark",
                    "status": "success",
                    "novelty_attack": "nearest work reviewed",
                    "baseline_attack": "baseline checked",
                    "mechanism_attack": "mechanism checked",
                    "evaluation_attack": "evaluation checked",
                    "scope_attack": "scope checked",
                    "a_plus_b_risk": "low",
                    "fatal_flaw": "",
                    "rescue_mutation": "",
                    "survivability_label": "survives",
                    "allowed_next_stage": "novelty_scan",
                }
            ],
        }

        def fake_opencode(_prompt: str, **kwargs: object) -> dict[str, object]:
            model = str(kwargs.get("model"))
            calls.append(model)
            if model == "abrdns/deepseek-v4-pro":
                return {
                    "exit_code": 0,
                    "timed_out": False,
                    "clean_output": "",
                    "effective_model": model,
                    "event_count": 1,
                    "error": "provider_event_error:APIError:Invalid token:status=401",
                    "event_errors": ["provider_event_error:APIError:Invalid token:status=401"],
                    "raw_stdout": '{"type":"error"}',
                    "raw_stderr": "",
                }
            return {
                "exit_code": 0,
                "timed_out": False,
                "clean_output": json.dumps(output),
                "effective_model": model,
                "event_count": 1,
            }

        with patch.dict(os.environ, {"DEEPSEEK_OPENCODE_FALLBACK_MODELS": "dwai/gpt-5.5"}), patch.object(deepseek_review, "run_opencode_cli", side_effect=fake_opencode):
            provider_payload = deepseek_review._opencode_provider_payload(
                [self.candidate()],
                run_date=RUN_DATE,
                model="abrdns/deepseek-v4-pro",
                timeout_sec=5,
                retries=0,
            )
        payload, exit_code = build_deepseek_payload([self.candidate()], run_date=RUN_DATE, provider_payload=provider_payload)
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(calls, ["abrdns/deepseek-v4-pro", "dwai/gpt-5.5"])
        self.assertTrue(payload["provider_status"]["provider_backed"])
        self.assertTrue(payload["provider_status"]["model_fallback_used"])
        self.assertEqual(payload["provider_status"]["primary_requested_model"], "abrdns/deepseek-v4-pro")
        self.assertEqual(payload["provider_status"]["requested_model"], "dwai/gpt-5.5")

    def test_deepseek_opencode_auth_error_does_not_fallback_by_default(self) -> None:
        calls: list[str] = []

        def fake_opencode(_prompt: str, **kwargs: object) -> dict[str, object]:
            model = str(kwargs.get("model"))
            calls.append(model)
            return {
                "exit_code": 0,
                "timed_out": False,
                "clean_output": "",
                "effective_model": model,
                "event_count": 1,
                "error": "provider_event_error:APIError:Invalid token:status=401",
                "event_errors": ["provider_event_error:APIError:Invalid token:status=401"],
                "raw_stdout": '{"type":"error"}',
                "raw_stderr": "",
            }

        with patch.dict(os.environ, {"DEEPSEEK_OPENCODE_FALLBACK_MODELS": ""}), patch.object(deepseek_review, "run_opencode_cli", side_effect=fake_opencode):
            provider_payload = deepseek_review._opencode_provider_payload(
                [self.candidate()],
                run_date=RUN_DATE,
                model="abrdns/deepseek-v4-pro",
                timeout_sec=5,
                retries=0,
            )
        payload, exit_code = build_deepseek_payload([self.candidate()], run_date=RUN_DATE, provider_payload=provider_payload)
        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["status"], "partial_provider_unavailable")
        self.assertEqual(calls, ["abrdns/deepseek-v4-pro"])
        self.assertFalse(payload["provider_status"]["provider_backed"])

    def test_deepseek_opencode_missing_candidate_review_fails_closed(self) -> None:
        output = {
            "schema_version": "deepseek_review.v1",
            "run_date": RUN_DATE,
            "status": "success",
            "reviews": [],
        }
        with patch.object(
            deepseek_review,
            "run_opencode_cli",
            return_value={"exit_code": 0, "timed_out": False, "clean_output": json.dumps(output), "effective_model": "deepseek/deepseek-v4-pro", "event_count": 1},
        ):
            provider_payload = deepseek_review._opencode_provider_payload([self.candidate()], run_date=RUN_DATE, model="deepseek/deepseek-v4-pro", timeout_sec=5)
        payload, exit_code = build_deepseek_payload([self.candidate()], run_date=RUN_DATE, provider_payload=provider_payload)
        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["status"], "partial_provider_invalid")
        self.assertFalse(payload["provider_status"]["provider_backed"])

    def test_bom_codex_provider_json_can_be_loaded(self) -> None:
        item = self.candidate()
        write_run_artifact(
            RUN_DATE,
            "selected-candidates.json",
            {
                "schema_version": "selected_candidates.v1",
                "run_date": RUN_DATE,
                "selected": [item],
                "rejected": [],
                "selection_rules": {},
                "artifact_hashes": {},
            },
            state="portfolio_selected",
        )
        provider_path = Path(self.tmp.name) / "provider-codex.json"
        self.write_bom_json(provider_path, self.provider_codex_payload(item))
        result = execution_review(RUN_DATE, dry_run=False, provider_review_json=str(provider_path))
        self.assertEqual(result["status"], "success")
        payload = read_json(artifact_dir(RUN_DATE) / "codex-execution-review.json")
        self.assertTrue(payload["provider_status"]["provider_backed"])
        self.assertEqual(payload["reviews"][0]["action"], "accept_for_user_review")

    def _fake_codex_popen(self, output: str, *, exit_code: int = 0):
        class FakeProc:
            def __init__(self, command: list[str], **_kwargs: object) -> None:
                self.command = command
                self.pid = 999999
                self.returncode = exit_code

            def communicate(self, _stdin: str | None = None, timeout: int | None = None) -> tuple[str, str]:
                if "--output-last-message" in self.command:
                    path = Path(self.command[self.command.index("--output-last-message") + 1])
                    path.write_text(output, encoding="utf-8")
                return "", ""

        return FakeProc

    def test_codex_cli_provider_success_with_mocked_subprocess(self) -> None:
        self.write_review_artifacts()
        output = {
            "schema_version": "codex_execution_review.v1",
            "run_date": RUN_DATE,
            "status": "success",
            "reviews": self.provider_codex_payload()["reviews"],
        }
        with patch.object(codex_review.shutil, "which", return_value="codex"), patch.object(
            codex_review.subprocess,
            "Popen",
            self._fake_codex_popen(json.dumps(output)),
        ):
            result = execution_review(RUN_DATE, dry_run=False, provider="codex-cli", timeout_sec=5)
        self.assertEqual(result["status"], "success")
        payload = read_json(artifact_dir(RUN_DATE) / "codex-execution-review.json")
        self.assertTrue(payload["provider_status"]["provider_backed"])

    def test_codex_cli_provider_ignores_user_config_to_avoid_local_otel_hang(self) -> None:
        self.write_review_artifacts()
        output = {
            "schema_version": "codex_execution_review.v1",
            "run_date": RUN_DATE,
            "status": "success",
            "reviews": self.provider_codex_payload()["reviews"],
        }
        captured: dict[str, object] = {}

        class FakeProc:
            def __init__(self, command: list[str], **_kwargs: object) -> None:
                captured["command"] = command
                self.command = command
                self.pid = 424242
                self.returncode = 0

            def communicate(self, _stdin: str | None = None, timeout: int | None = None) -> tuple[str, str]:
                path = Path(self.command[self.command.index("--output-last-message") + 1])
                path.write_text(json.dumps(output), encoding="utf-8")
                return "", ""

        with patch.object(codex_review.shutil, "which", return_value="codex"), patch.object(
            codex_review.subprocess,
            "Popen",
            FakeProc,
        ):
            result = execution_review(RUN_DATE, dry_run=False, provider="codex-cli", timeout_sec=5)
        self.assertEqual(result["status"], "success")
        command = captured["command"]
        self.assertIsInstance(command, list)
        assert isinstance(command, list)
        self.assertIn("--ignore-user-config", command)

    def test_codex_execution_review_hydrates_missing_status_from_substantive_provider_review(self) -> None:
        self.write_review_artifacts()
        provider = self.provider_codex_payload()
        del provider["reviews"][0]["status"]  # type: ignore[index]
        provider_path = Path(self.tmp.name) / "provider-codex-missing-status.json"
        self.write_bom_json(provider_path, provider)

        result = execution_review(RUN_DATE, dry_run=False, provider_review_json=str(provider_path))

        self.assertEqual(result["status"], "success")
        payload = read_json(artifact_dir(RUN_DATE) / "codex-execution-review.json")
        self.assertEqual(payload["reviews"][0]["status"], "success")
        self.assertTrue(payload["provider_status"]["review_status_hydrated_from_execution_fields"])

    def test_codex_execution_review_coerces_string_false_field_presence_only(self) -> None:
        self.write_review_artifacts()
        provider = self.provider_codex_payload()
        provider["reviews"][0]["field_presence_only"] = "false"  # type: ignore[index]
        provider_path = Path(self.tmp.name) / "provider-codex-string-false.json"
        self.write_bom_json(provider_path, provider)

        result = execution_review(RUN_DATE, dry_run=False, provider_review_json=str(provider_path))

        self.assertEqual(result["status"], "success")
        payload = read_json(artifact_dir(RUN_DATE) / "codex-execution-review.json")
        self.assertFalse(payload["reviews"][0]["field_presence_only"])
        self.assertTrue(payload["provider_status"]["provider_backed"])

    def test_codex_execution_review_invalid_provider_review_writes_partial_artifact(self) -> None:
        self.write_review_artifacts()
        provider = {
            "provider_status": {"provider": "codex", "provider_backed": True, "mode": "codex-cli"},
            "reviews": [{"candidate_id": "cand-alpha"}],
        }
        provider_path = Path(self.tmp.name) / "provider-codex-invalid-review.json"
        self.write_bom_json(provider_path, provider)

        result = execution_review(RUN_DATE, dry_run=False, provider_review_json=str(provider_path))

        self.assertEqual(result["status"], "partial_provider_invalid")
        payload = read_json(artifact_dir(RUN_DATE) / "codex-execution-review.json")
        self.assertFalse(payload["provider_status"]["provider_backed"])
        self.assertEqual(payload["reviews"][0]["status"], "failed_fallback_only")
        self.assertIn(
            "execution_review_missing_field:cand-alpha:no_hardware_pilot_feasibility",
            payload["provider_status"]["validation_errors"],
        )

    def test_codex_cli_invalid_json_fails_closed(self) -> None:
        self.write_review_artifacts()
        with patch.object(codex_review.shutil, "which", return_value="codex"), patch.object(
            codex_review.subprocess,
            "Popen",
            self._fake_codex_popen("not json"),
        ):
            result = execution_review(RUN_DATE, dry_run=False, provider="codex-cli", timeout_sec=5)
        self.assertEqual(result["status"], "partial_provider_invalid")
        payload = read_json(artifact_dir(RUN_DATE) / "codex-execution-review.json")
        self.assertFalse(payload["provider_status"]["provider_backed"])

    def test_codex_cli_nonzero_exit_writes_fail_closed_fallback_and_continues(self) -> None:
        self.write_review_artifacts()
        with patch.object(codex_review.shutil, "which", return_value="codex"), patch.object(
            codex_review.subprocess,
            "Popen",
            self._fake_codex_popen("", exit_code=1),
        ):
            result = execution_review(RUN_DATE, dry_run=False, provider="codex-cli", timeout_sec=5)
        self.assertEqual(result["status"], "partial_provider_unavailable")
        self.assertEqual(result["fallback_reviews_fail_closed"], "true")
        payload = read_json(artifact_dir(RUN_DATE) / "codex-execution-review.json")
        self.assertFalse(payload["provider_status"]["provider_backed"])
        self.assertTrue(payload["provider_status"]["fallback_reviews_fail_closed"])
        self.assertEqual(payload["provider_status"]["provider_fallback_count"], 1)
        self.assertEqual(payload["reviews"][0]["status"], "failed_fallback_only")
        self.assertEqual(payload["reviews"][0]["action"], "reject_with_rescue")

    def test_codex_execution_review_main_returns_zero_for_fail_closed_fallback(self) -> None:
        self.write_review_artifacts()
        old_argv = sys.argv[:]
        try:
            sys.argv = ["codex_seed_review.py", "execution-review", "--run-date", RUN_DATE, "--provider", "codex-cli", "--timeout", "5"]
            with patch.object(codex_review.shutil, "which", return_value="codex"), patch.object(
                codex_review.subprocess,
                "Popen",
                self._fake_codex_popen("", exit_code=1),
            ):
                self.assertEqual(codex_review.main(), 0)
        finally:
            sys.argv = old_argv

    def test_codex_cli_field_presence_accept_fails_closed(self) -> None:
        self.write_review_artifacts()
        provider = self.provider_codex_payload()
        provider["reviews"][0]["field_presence_only"] = True  # type: ignore[index]
        with patch.object(codex_review.shutil, "which", return_value="codex"), patch.object(
            codex_review.subprocess,
            "Popen",
            self._fake_codex_popen(json.dumps(provider)),
        ):
            result = execution_review(RUN_DATE, dry_run=False, provider="codex-cli", timeout_sec=5)
        self.assertEqual(result["status"], "partial_provider_invalid")
        payload = read_json(artifact_dir(RUN_DATE) / "codex-execution-review.json")
        self.assertFalse(payload["provider_status"]["provider_backed"])

    def test_unknown_novelty_cannot_auto_promote(self) -> None:
        self.write_review_artifacts(novelty="unknown")
        payload = decide(run_date=RUN_DATE, allow_human_override=False)
        self.assertEqual(payload["decisions"][0]["decision"], "killed")
        self.assertIn("unknown_novelty_without_human_override", payload["decisions"][0]["blocks"])

    def test_codex_fallback_review_blocks_candidate_in_survival(self) -> None:
        cid = self.write_review_artifacts()
        write_run_artifact(
            RUN_DATE,
            "codex-execution-review.json",
            {
                "schema_version": "codex_execution_review.v1",
                "run_date": RUN_DATE,
                "status": "partial_provider_unavailable",
                "reviews": [
                    {
                        "candidate_id": cid,
                        "candidate_title": "Test candidate",
                        "status": "failed_fallback_only",
                        "action": "reject_with_rescue",
                        "execution_risks": ["provider_backed_codex_execution_review_missing"],
                        "no_hardware_pilot_feasibility": "not_verified",
                        "public_dataset_or_sim_availability": "not_verified",
                        "baseline_training_cost": "not_verified",
                        "metric_automation": "not_verified",
                        "data_leakage_risk": "not_verified",
                        "minimal_repo_plan": "not_verified",
                        "real_robot_pilot_complexity": "not_verified",
                        "reproducibility_path": "not_verified",
                        "compute_time_budget": "not_verified",
                        "field_presence_only": True,
                    }
                ],
                "provider_status": {
                    "provider": "codex",
                    "provider_backed": False,
                    "mode": "codex-cli",
                    "fallback_reviews_fail_closed": True,
                },
                "artifact_hashes": {},
            },
            state="execution_reviewed",
        )

        payload = decide(run_date=RUN_DATE, allow_human_override=False)

        decision = payload["decisions"][0]
        self.assertEqual(decision["decision"], "killed")
        self.assertIn("codex_not_provider_backed_success", decision["blocks"])
        self.assertIn("codex_status_not_success", decision["blocks"])
        self.assertIn("codex_action_not_accept:reject_with_rescue", decision["blocks"])

    def test_local_only_likely_open_cannot_formal_publish(self) -> None:
        self.write_review_artifacts(novelty="likely_open", verification_scope="local_only")
        payload = decide(run_date=RUN_DATE, allow_human_override=False, target_policy="formal")
        self.assertEqual(payload["decisions"][0]["decision"], "killed")
        self.assertIn("formal_novelty_promotion_not_allowed", payload["decisions"][0]["blocks"])
        self.assertIn("formal_novelty_requires_external_or_hybrid_scope", payload["decisions"][0]["blocks"])

    def test_local_only_likely_open_can_seed_candidates_only(self) -> None:
        self.write_review_artifacts(novelty="likely_open", verification_scope="local_only")
        survival = decide(run_date=RUN_DATE, allow_human_override=False)
        write_run_artifact(RUN_DATE, "survival-decision.json", survival, state="survival_decided")
        result = publish(RUN_DATE, dry_run=False, target_policy="seed-candidates-only")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["published"], [])
        self.assertEqual(len(result["bucketed"]), 1)

    def test_deepseek_candidate_level_fallback_does_not_kill_reviewed_candidates(self) -> None:
        alpha = self.candidate("cand-alpha")
        beta = self.candidate("cand-beta")
        self.write_claim_graph_node()
        write_run_artifact(
            RUN_DATE,
            "selected-candidates.json",
            {
                "schema_version": "selected_candidates.v1",
                "run_date": RUN_DATE,
                "selected": [alpha, beta],
                "rejected": [],
                "selection_rules": {},
                "artifact_hashes": {},
            },
            state="portfolio_selected",
        )
        write_run_artifact(
            RUN_DATE,
            "deepseek-review.json",
            {
                "schema_version": "deepseek_review.v1",
                "run_date": RUN_DATE,
                "status": "partial_provider_invalid",
                "reviews": [
                    {
                        "candidate_id": "cand-alpha",
                        "candidate_title": str(alpha["title"]),
                        "status": "success",
                        "novelty_attack": "nearest work reviewed",
                        "baseline_attack": "baseline checked",
                        "mechanism_attack": "mechanism checked",
                        "evaluation_attack": "evaluation checked",
                        "scope_attack": "scope checked",
                        "a_plus_b_risk": "low",
                        "fatal_flaw": "",
                        "rescue_mutation": "",
                        "survivability_label": "survives",
                        "allowed_next_stage": "novelty_scan",
                    },
                    {
                        "candidate_id": "cand-beta",
                        "candidate_title": str(beta["title"]),
                        "status": "failed_fallback_only",
                        "novelty_attack": "Provider review unavailable.",
                        "baseline_attack": "Provider review unavailable.",
                        "mechanism_attack": "Provider review unavailable.",
                        "evaluation_attack": "Provider review unavailable.",
                        "scope_attack": "Provider review unavailable.",
                        "a_plus_b_risk": "fatal",
                        "fatal_flaw": "opencode_invalid_json",
                        "rescue_mutation": "",
                        "survivability_label": "reject_fatal",
                        "allowed_next_stage": "stop",
                    },
                ],
                "provider_status": {
                    "provider": "deepseek",
                    "provider_backed": False,
                    "mode": "opencode",
                    "candidate_level_fail_closed": True,
                },
                "artifact_hashes": {},
            },
            state="attacked_by_deepseek",
        )
        scans = []
        codex_reviews = []
        for item in [alpha, beta]:
            cid = str(item["candidate_id"])
            scans.append(
                {
                    "candidate_id": cid,
                    "candidate_title": str(item["title"]),
                    "status": "completed_local_only",
                    "novelty_classification": "likely_open",
                    "promotion_allowed": True,
                    "formal_promotion_allowed": False,
                    "verification_scope": "local_only",
                    "external_providers_used": [],
                    "formal_publish_risk": "",
                    "provider_results": {},
                    "provider_errors": [],
                    "nearest_works": [{"source": "test", "score": 0.1, "title": "Near work"}],
                    "strongest_baseline": {"source": "test", "score": 0.1, "title": "Near work"},
                    "local_scan_evidence": {},
                    "external_scan_evidence": {},
                    "duplicate_guard": {"status": "pass", "action": "allow_with_lineage", "nearest_candidates": []},
                }
            )
            codex_reviews.append(
                {
                    "candidate_id": cid,
                    "candidate_title": str(item["title"]),
                    "status": "success",
                    "action": "accept_for_user_review",
                    "no_hardware_pilot_feasibility": "feasible",
                    "public_dataset_or_sim_availability": "available",
                    "baseline_training_cost": "single GPU day",
                    "metric_automation": "scriptable",
                    "data_leakage_risk": "low",
                    "minimal_repo_plan": "notebook plus evaluator",
                    "real_robot_pilot_complexity": "low",
                    "reproducibility_path": "public sim",
                    "compute_time_budget": "one day",
                }
            )
        write_run_artifact(
            RUN_DATE,
            "novelty-scan.json",
            {
                "schema_version": "novelty_scan.v1",
                "run_date": RUN_DATE,
                "status": "completed_local_only",
                "scans": scans,
                "provider_policy": {},
                "artifact_hashes": {},
            },
            state="novelty_checked",
        )
        write_run_artifact(
            RUN_DATE,
            "codex-execution-review.json",
            {
                "schema_version": "codex_execution_review.v1",
                "run_date": RUN_DATE,
                "status": "success",
                "reviews": codex_reviews,
                "provider_status": {"provider": "codex", "provider_backed": True, "mode": "codex-cli"},
                "artifact_hashes": {},
            },
            state="execution_reviewed",
        )

        survival = decide(run_date=RUN_DATE, allow_human_override=False)

        by_id = {item["candidate_id"]: item for item in survival["decisions"]}
        self.assertEqual(by_id["cand-alpha"]["decision"], "accept_for_user_review")
        self.assertNotIn("deepseek_not_provider_backed_success", by_id["cand-alpha"]["blocks"])
        self.assertEqual(by_id["cand-beta"]["decision"], "killed")
        self.assertIn("deepseek_status_not_success", by_id["cand-beta"]["blocks"])
        self.assertIn("deepseek_label_not_survivable", by_id["cand-beta"]["blocks"])

    def test_formal_candidate_with_only_low_note_only_core_evidence_blocks(self) -> None:
        self.write_review_artifacts(
            verification_scope="local_plus_s2_or_openalex",
            deepseek_mode="opencode",
            codex_mode="codex-cli",
        )
        self.write_claim_graph_node(confidence="low", anchor_type="note_only", requires_human_check=True)
        payload = decide(run_date=RUN_DATE, allow_human_override=False, target_policy="formal")
        self.assertEqual(payload["decisions"][0]["decision"], "killed")
        self.assertIn("formal_core_evidence_not_anchored", payload["decisions"][0]["blocks"])

    def test_seed_candidates_only_records_anchorless_risk_without_formal_seed(self) -> None:
        self.write_review_artifacts(novelty="likely_open", verification_scope="local_only")
        self.write_claim_graph_node(confidence="low", anchor_type="note_only", requires_human_check=True)
        survival = decide(run_date=RUN_DATE, allow_human_override=False)
        self.assertEqual(survival["decisions"][0]["decision"], "accept_for_user_review")
        self.assertIn("anchorless_core_evidence_risk", survival["decisions"][0]["risks"])
        write_run_artifact(RUN_DATE, "survival-decision.json", survival, state="survival_decided")
        result = publish(RUN_DATE, dry_run=False, target_policy="seed-candidates-only")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["published"], [])
        self.assertEqual(list((Path(self.tmp.name) / "idea_bank" / "seed").glob("*")), [])

    def test_speculative_tension_cannot_support_formal_seed(self) -> None:
        candidate = {**self.candidate(), "supporting_tensions": ["tension-spec"]}
        self.write_review_artifacts(
            candidate=candidate,
            verification_scope="local_plus_s2_or_openalex",
            deepseek_mode="opencode",
            codex_mode="codex-cli",
        )
        write_run_artifact(
            RUN_DATE,
            "tension-map.json",
            {
                "schema_version": "tension_map.v1",
                "run_date": RUN_DATE,
                "tension_types": [],
                "tensions": [],
                "speculative_tensions": [
                    {
                        "tension_id": "tension-spec",
                        "tension_type": "speculative_tension",
                        "summary": "LLM-only tension for breakthrough lane.",
                        "supporting_nodes": [],
                        "supporting_edges": [],
                        "confidence": "low",
                        "source": "llm_only",
                        "do_not_use_as_seed_evidence": True,
                        "original_tension_type": "baseline_contradiction",
                        "allowed_lane": "breakthrough_speculative",
                    }
                ],
                "validation_errors": [],
                "artifact_hashes": {},
            },
            state="tension_mapped",
        )
        payload = decide(run_date=RUN_DATE, allow_human_override=False, target_policy="formal")
        self.assertIn("speculative_tension_not_formal_seed_evidence", payload["decisions"][0]["blocks"])

    def test_external_arxiv_only_novelty_blocks_formal_publish_with_risk_marker(self) -> None:
        self.write_review_artifacts(
            verification_scope="local_plus_arxiv_api",
            deepseek_mode="opencode",
            codex_mode="codex-cli",
        )
        survival = decide(run_date=RUN_DATE, allow_human_override=False, target_policy="formal")
        self.assertEqual(survival["decisions"][0]["decision"], "killed")
        self.assertIn("formal_novelty_arxiv_only_scope_not_broad_prior_art", survival["decisions"][0]["blocks"])
        self.assertIn("formal_novelty_requires_openalex_or_semantic_scholar", survival["decisions"][0]["blocks"])
        write_run_artifact(RUN_DATE, "survival-decision.json", survival, state="survival_decided")
        result = publish(RUN_DATE, dry_run=False, target_policy="formal")
        self.assert_legacy_formal_publish_disabled(result)

    def test_strict_publish_rejects_missing_external_broad_provider(self) -> None:
        self.write_review_artifacts(
            verification_scope="local_plus_arxiv_api",
            deepseek_mode="opencode",
            codex_mode="codex-cli",
            formal_promotion_allowed=False,
        )
        survival = decide(run_date=RUN_DATE, allow_human_override=False, target_policy="formal")
        write_run_artifact(RUN_DATE, "survival-decision.json", survival, state="survival_decided")
        manifest = read_json(run_dir(RUN_DATE) / "manifest.json")
        manifest["v2_publish_policy"] = "formal"
        manifest["formal_seed_publish_allowed"] = True
        write_json(run_dir(RUN_DATE) / "manifest.json", manifest)
        validation = validate_run(RUN_DATE, strict_publish=True)
        self.assertEqual(validation["status"], "partial_schema_blocked")
        self.assertTrue(any("formal_novelty_missing_broad_external_provider" in item for item in validation["errors"]))

    def test_deepseek_local_adapter_does_not_satisfy_gate(self) -> None:
        payload, exit_code = build_deepseek_payload([self.candidate()], run_date=RUN_DATE, provider_payload={})
        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["status"], "partial_provider_unavailable")
        self.assertFalse(payload["provider_status"]["provider_backed"])

    def test_codex_field_presence_only_does_not_accept(self) -> None:
        self.write_review_artifacts()
        result = execution_review(RUN_DATE, dry_run=False, provider_review_json="")
        self.assertEqual(result["status"], "partial_provider_unavailable")
        payload = json.loads((artifact_dir(RUN_DATE) / "codex-execution-review.json").read_text(encoding="utf-8"))
        self.assertNotEqual(payload["reviews"][0]["action"], "accept_for_user_review")
        self.assertFalse(payload["provider_status"]["provider_backed"])

    def test_novelty_no_nearest_work_defaults_unknown(self) -> None:
        guard = {"status": "pass", "action": "allow_with_lineage"}
        classification, allowed, reason = novelty_scan._classify(
            guard=guard,
            nearest_works=[],
            evidence_summary={
                "local_notes_claim_graph": {"records_scanned": 0},
                "local_arxiv_mirror": {"records_scanned": 0, "error": "missing"},
            },
            remaining_delta="test delta",
            strict_external=False,
        )
        self.assertEqual(classification, "unknown")
        self.assertFalse(allowed)
        self.assertEqual(reason, "insufficient_local_or_external_evidence")

    def test_novelty_empty_selection_is_success_noop(self) -> None:
        write_run_artifact(
            RUN_DATE,
            "selected-candidates.json",
            {"schema_version": "selected_candidates.v1", "run_date": RUN_DATE, "selected": [], "rejected": []},
            state="portfolio_selected",
        )
        old_argv = sys.argv
        try:
            sys.argv = ["novelty_baseline_scan.py", "--run-date", RUN_DATE, "--target-policy", "seed-candidates-only"]
            exit_code = novelty_scan.main()
        finally:
            sys.argv = old_argv

        payload = read_json(artifact_dir(RUN_DATE) / "novelty-scan.json")
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "success_empty_selection")
        self.assertEqual(payload["scans"], [])
        self.assertEqual(validate_payload(payload, "novelty_scan.v1"), [])

    def test_arxiv_external_provider_timeout_fails_closed(self) -> None:
        with patch.object(novelty_scan.urllib.request, "urlopen", side_effect=TimeoutError("test timeout")):
            hits, summary = novelty_scan._scan_arxiv_api(self.candidate(), max_queries=1, timeout=1, delay_sec=0)
        self.assertEqual(hits, [])
        self.assertIn("TimeoutError", summary["error"])

    def test_openalex_provider_success_sets_broad_scope(self) -> None:
        hit = {"source": "openalex", "score": 0.2, "title": "Anchored contact failure benchmark", "locator": "W1", "evidence": "near work"}
        unavailable = {"provider": "semantic_scholar", "status": "provider_unavailable", "records_scanned": 0, "cached": False, "error": "semantic_scholar_api_key_missing"}
        with patch.object(novelty_scan, "_scan_claim_graph", return_value=([], {"records_scanned": 0})), patch.object(
            novelty_scan,
            "_scan_arxiv_mirror",
            return_value=([], {"records_scanned": 0}),
        ), patch.object(
            novelty_scan,
            "_scan_arxiv_api",
            return_value=([], {"provider": "arxiv_api", "status": "provider_unavailable", "records_scanned": 0, "cached": False, "error": "arxiv_timeout"}),
        ), patch.object(
            novelty_scan,
            "_scan_openalex",
            return_value=([hit], {"provider": "openalex", "status": "success", "records_scanned": 1, "cached": False, "queries": 1}),
        ), patch.object(
            novelty_scan,
            "_scan_semantic_scholar",
            return_value=([], unavailable),
        ):
            scan = novelty_scan._build_candidate_scan(
                self.candidate(),
                target_policy="formal",
                strict_external=False,
                max_external_queries=1,
                external_timeout=1,
            )
        self.assertIn("openalex", scan["external_providers_used"])
        self.assertEqual(scan["verification_scope"], "local_plus_s2_or_openalex")
        self.assertTrue(scan["formal_promotion_allowed"])

    def test_semantic_scholar_success_with_fixture_is_broad_provider(self) -> None:
        fixture = {
            "data": [
                {
                    "paperId": "s2-1",
                    "title": "Anchored contact failure benchmark for DLO policies",
                    "abstract": "Contact rich DLO policies fail under latent friction shifts at the vision tactile controller boundary.",
                    "url": "https://www.semanticscholar.org/paper/s2-1",
                }
            ]
        }
        with patch.object(novelty_scan.urllib.request, "urlopen", self.fake_urlopen_json(fixture)):
            hits, summary = novelty_scan._scan_semantic_scholar(self.candidate(), max_queries=1, timeout=1, api_key="fake-key", delay_sec=0)
        self.assertEqual(summary["status"], "success")
        self.assertTrue(hits)

        unavailable = {"provider": "openalex", "status": "provider_unavailable", "records_scanned": 0, "cached": False, "error": "openalex_timeout"}
        with patch.object(novelty_scan, "_scan_claim_graph", return_value=([], {"records_scanned": 0})), patch.object(
            novelty_scan,
            "_scan_arxiv_mirror",
            return_value=([], {"records_scanned": 0}),
        ), patch.object(
            novelty_scan,
            "_scan_arxiv_api",
            return_value=([], {"provider": "arxiv_api", "status": "provider_unavailable", "records_scanned": 0, "cached": False, "error": "arxiv_timeout"}),
        ), patch.object(
            novelty_scan,
            "_scan_openalex",
            return_value=([], unavailable),
        ), patch.object(
            novelty_scan,
            "_scan_semantic_scholar",
            return_value=(hits, summary),
        ):
            scan = novelty_scan._build_candidate_scan(
                self.candidate(),
                target_policy="formal",
                strict_external=False,
                max_external_queries=1,
                external_timeout=1,
            )
        self.assertIn("semantic_scholar", scan["external_providers_used"])
        self.assertEqual(scan["verification_scope"], "local_plus_s2_or_openalex")
        self.assertTrue(scan["formal_promotion_allowed"])

    def test_semantic_scholar_429_fails_closed_for_formal(self) -> None:
        error = urllib.error.HTTPError(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            429,
            "Too Many Requests",
            hdrs=None,
            fp=None,
        )
        with patch.object(novelty_scan.urllib.request, "urlopen", side_effect=error):
            hits, summary = novelty_scan._scan_semantic_scholar(self.candidate(), max_queries=1, timeout=1, api_key="fake-key", delay_sec=0)
        self.assertEqual(hits, [])
        self.assertIn("HTTPError", summary["error"])

        self.write_review_artifacts(
            verification_scope="local_plus_arxiv_api",
            deepseek_mode="opencode",
            codex_mode="codex-cli",
            formal_promotion_allowed=False,
        )
        survival = decide(run_date=RUN_DATE, allow_human_override=False, target_policy="formal")
        write_run_artifact(RUN_DATE, "survival-decision.json", survival, state="survival_decided")
        result = publish(RUN_DATE, dry_run=False, target_policy="formal")
        self.assert_legacy_formal_publish_disabled(result)

    def test_openalex_partial_overlap_with_remaining_delta_can_pass_novelty_gate(self) -> None:
        local_hit = {"source": "local_notes_claim_graph", "score": 0.4, "title": "Contact failure benchmark", "locator": "claim-1", "evidence": "near overlap"}
        openalex_hit = {"source": "openalex", "score": 0.2, "title": "External baseline", "locator": "W1", "evidence": "external near work"}
        with patch.object(novelty_scan, "_scan_claim_graph", return_value=([local_hit], {"records_scanned": 1})), patch.object(
            novelty_scan,
            "_scan_arxiv_mirror",
            return_value=([], {"records_scanned": 0}),
        ), patch.object(
            novelty_scan,
            "_scan_arxiv_api",
            return_value=([], {"provider": "arxiv_api", "status": "provider_unavailable", "records_scanned": 0, "cached": False, "error": "arxiv_timeout"}),
        ), patch.object(
            novelty_scan,
            "_scan_openalex",
            return_value=([openalex_hit], {"provider": "openalex", "status": "success", "records_scanned": 1, "cached": False, "queries": 1}),
        ), patch.object(
            novelty_scan,
            "_scan_semantic_scholar",
            return_value=([], {"provider": "semantic_scholar", "status": "provider_unavailable", "records_scanned": 0, "cached": False, "error": "semantic_scholar_api_key_missing"}),
        ):
            scan = novelty_scan._build_candidate_scan(
                self.candidate(),
                target_policy="formal",
                strict_external=False,
                max_external_queries=1,
                external_timeout=1,
            )
        self.assertEqual(scan["novelty_classification"], "partial_overlap")
        self.assertTrue(scan["promotion_allowed"])
        self.assertTrue(scan["formal_promotion_allowed"])

    def test_all_external_providers_unavailable_makes_novelty_unknown(self) -> None:
        unavailable = {"status": "provider_unavailable", "records_scanned": 0, "cached": False, "error": "provider_down"}
        with patch.object(novelty_scan, "_scan_claim_graph", return_value=([], {"records_scanned": 0})), patch.object(
            novelty_scan,
            "_scan_arxiv_mirror",
            return_value=([], {"records_scanned": 0}),
        ), patch.object(
            novelty_scan,
            "_scan_arxiv_api",
            return_value=([], {"provider": "arxiv_api", **unavailable}),
        ), patch.object(
            novelty_scan,
            "_scan_openalex",
            return_value=([], {"provider": "openalex", **unavailable}),
        ), patch.object(
            novelty_scan,
            "_scan_semantic_scholar",
            return_value=([], {"provider": "semantic_scholar", **unavailable}),
        ):
            scan = novelty_scan._build_candidate_scan(
                self.candidate(),
                target_policy="formal",
                strict_external=False,
                max_external_queries=1,
                external_timeout=1,
            )
        self.assertEqual(scan["novelty_classification"], "unknown")
        self.assertFalse(scan["promotion_allowed"])
        self.assertFalse(scan["formal_promotion_allowed"])
        self.assertEqual(scan["external_providers_used"], [])

    def test_cached_openalex_response_prevents_duplicate_network_call(self) -> None:
        calls: list[object] = []
        fixture = {
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "display_name": "Anchored contact failure benchmark for DLO policies",
                    "publication_year": 2099,
                    "abstract_inverted_index": {
                        "Contact": [0],
                        "rich": [1],
                        "DLO": [2],
                        "policies": [3],
                        "fail": [4],
                        "under": [5],
                        "latent": [6],
                        "friction": [7],
                        "shifts": [8],
                    },
                }
            ]
        }
        with patch.object(novelty_scan.urllib.request, "urlopen", self.fake_urlopen_json(fixture, calls)):
            first_hits, first_summary = novelty_scan._scan_openalex(self.candidate(), max_queries=1, timeout=1, delay_sec=0)
            second_hits, second_summary = novelty_scan._scan_openalex(self.candidate(), max_queries=1, timeout=1, delay_sec=0)
        self.assertEqual(len(calls), 1)
        self.assertTrue(first_hits)
        self.assertTrue(second_hits)
        self.assertFalse(first_summary["cached"])
        self.assertTrue(second_summary["cached"])

    def test_bom_claim_graph_jsonl_first_line_is_used_by_novelty_scan(self) -> None:
        graph_path = Path(self.tmp.name) / "evidence" / "research_claim_graph.jsonl"
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": "research_claim_graph.v1",
            "node_id": "claim-1",
            "source_title": "Anchored Contact Failure Benchmark",
            "claim_type": "central_claim",
            "statement": "Contact rich DLO policies fail under latent friction shifts at the vision tactile controller interface boundary.",
            "source_note": "wiki/topics/example.md",
            "zotero_key": "ABC123",
            "anchor": {"anchor_type": "section", "section": "Results", "snippet": "Contact rich DLO policies fail under latent friction shifts."},
            "confidence": "high",
        }
        graph_path.write_text("\ufeff" + json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        nearest, summary = novelty_scan._scan_claim_graph(self.candidate())
        self.assertEqual(summary["records_scanned"], 1)
        self.assertTrue(nearest)

    def test_codex_reject_goes_to_rescue_or_parked_not_seed(self) -> None:
        self.write_review_artifacts(codex_action="reject_with_rescue")
        payload = decide(run_date=RUN_DATE, allow_human_override=False)
        self.assertNotEqual(payload["decisions"][0]["decision"], "accept_for_user_review")

    def test_mutated_candidate_requires_novelty_and_codex_rescan_for_final_id(self) -> None:
        parent = self.candidate("cand-parent")
        mutation = {**self.candidate("cand-mutated"), "parent_candidate_id": "cand-parent"}
        self.write_review_artifacts(
            candidate=parent,
            deepseek_label="survives_if_mutated",
            mutations=[mutation],
            novelty_candidate_id="cand-parent",
            codex_candidate_id="cand-mutated",
        )
        payload = decide(run_date=RUN_DATE, allow_human_override=False)
        self.assertNotEqual(payload["decisions"][0]["decision"], "accept_for_user_review")
        self.assertIn("unknown_novelty_without_human_override", payload["decisions"][0]["blocks"])

    def test_human_override_cannot_bypass_deepseek_fatal_flaw(self) -> None:
        cid = self.write_review_artifacts(novelty="unknown", fatal_flaw=True)
        override_dir = Path(self.tmp.name) / "overrides" / "human-overrides" / RUN_DATE
        override_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            override_dir / f"{cid}.json",
            {
                "schema_version": "human_override.v1",
                "candidate_id": cid,
                "run_id": RUN_DATE,
                "override_type": "accept_unknown_novelty_risk",
                "reviewer": "human",
                "created_at": "2099-01-02T00:00:00Z",
                "reason": "test override",
                "risk_acceptance": "accept unknown novelty risk only",
                "expires_after_days": 7,
                "allowed_publish_target": "seed",
                "cannot_weaken": ["deepseek_fatal_flaw", "codex_reject", "missing_survival_decision"],
            },
        )
        payload = decide(run_date=RUN_DATE, allow_human_override=True)
        self.assertIn("deepseek_fatal_flaw", payload["decisions"][0]["blocks"])
        self.assertIn("human_override_cannot_bypass_hard_block", payload["decisions"][0]["blocks"])

    def test_bom_human_override_consumed_only_when_allowed(self) -> None:
        cid = self.write_review_artifacts(novelty="unknown")
        override_dir = Path(self.tmp.name) / "overrides" / "human-overrides" / RUN_DATE
        override = {
            "schema_version": "human_override.v1",
            "candidate_id": cid,
            "run_id": RUN_DATE,
            "override_type": "accept_unknown_novelty_risk",
            "reviewer": "human",
            "created_at": "2099-01-02T00:00:00Z",
            "reason": "test override",
            "risk_acceptance": "accept unknown novelty risk only",
            "expires_after_days": 7,
            "allowed_publish_target": "seed",
            "cannot_weaken": ["deepseek_fatal_flaw", "codex_reject", "missing_survival_decision"],
        }
        self.write_bom_json(override_dir / f"{cid}.json", override)
        without_allow = decide(run_date=RUN_DATE, allow_human_override=False)
        self.assertEqual(without_allow["human_overrides_consumed"], [])
        self.assertFalse(without_allow["decisions"][0]["human_override_used"])

        with_allow = decide(run_date=RUN_DATE, allow_human_override=True)
        self.assertEqual(with_allow["human_overrides_consumed"], [cid])
        self.assertTrue(with_allow["decisions"][0]["human_override_used"])
        self.assertEqual(with_allow["decisions"][0]["decision"], "killed")
        self.assertIn("novelty_promotion_not_allowed", with_allow["decisions"][0]["blocks"])

    def test_strategy_override_cannot_weaken_hard_gates(self) -> None:
        safe, errors = sanitize_overrides({"allow_deepseek_fatal_flaw": True, "lane_quota": {"outside_analogy": 2}})
        self.assertIn("attempted_hard_gate_override:allow_deepseek_fatal_flaw", errors)
        self.assertEqual(safe["overrides"], {"lane_quota": {"outside_analogy": 2}})

    def test_missing_survival_decision_blocks_publish(self) -> None:
        self.write_review_artifacts()
        result = publish(RUN_DATE, dry_run=True)
        self.assertEqual(result["status"], "blocked_validation")
        self.assertTrue(any("survival-decision.json" in item for item in result["blocked"]))

    def test_publish_dry_run_writes_no_formal_seed(self) -> None:
        self.write_review_artifacts()
        survival = decide(run_date=RUN_DATE, allow_human_override=False)
        write_run_artifact(RUN_DATE, "survival-decision.json", survival, state="survival_decided")
        result = publish(RUN_DATE, dry_run=True)
        self.assertEqual(result["status"], "success")
        seed_root = Path(self.tmp.name) / "idea_bank" / "seed"
        self.assertEqual([path for path in seed_root.glob("*") if path.is_dir()], [])

    def test_default_v2_publish_policy_is_not_formal(self) -> None:
        self.assertEqual(resolve_v2_publish_policy(Namespace()), "seed-candidates-only")

    def test_seed_candidates_only_does_not_write_formal_seed(self) -> None:
        self.write_review_artifacts()
        survival = decide(run_date=RUN_DATE, allow_human_override=False)
        write_run_artifact(RUN_DATE, "survival-decision.json", survival, state="survival_decided")
        result = publish(RUN_DATE, dry_run=False, target_policy="seed-candidates-only")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["published"], [])
        self.assertEqual(len(result["bucketed"]), 1)
        self.assertIn("seed-candidates", result["bucketed"][0]["target"])
        self.assertEqual(list((Path(self.tmp.name) / "idea_bank" / "seed").glob("*")), [])
        manifest = read_json(run_dir(RUN_DATE) / "manifest.json")
        self.assertFalse(manifest["formal_seed_written"])
        self.assertEqual(manifest["v2_publish_policy"], "seed-candidates-only")

    def test_formal_publish_without_explicit_allow_blocks(self) -> None:
        self.write_review_artifacts()
        survival = decide(run_date=RUN_DATE, allow_human_override=False)
        write_run_artifact(RUN_DATE, "survival-decision.json", survival, state="survival_decided")
        result = publish(RUN_DATE, dry_run=False, target_policy="formal")
        self.assert_legacy_formal_publish_disabled(result)

    def test_formal_provider_json_blocks_by_default(self) -> None:
        self.write_review_artifacts(verification_scope="local_plus_s2_or_openalex")
        survival = decide(run_date=RUN_DATE, allow_human_override=False, target_policy="formal")
        write_run_artifact(RUN_DATE, "survival-decision.json", survival, state="survival_decided")
        result = publish(RUN_DATE, dry_run=False, target_policy="formal")
        self.assert_legacy_formal_publish_disabled(result)

    def test_formal_provider_json_manual_override_records_risk(self) -> None:
        self.write_review_artifacts(verification_scope="local_plus_s2_or_openalex")
        self.write_active_seed_support()
        survival = decide(run_date=RUN_DATE, allow_human_override=False, target_policy="formal")
        write_run_artifact(RUN_DATE, "survival-decision.json", survival, state="survival_decided")
        result = publish(RUN_DATE, dry_run=False, target_policy="formal")
        self.assert_legacy_formal_publish_disabled(result)

    def test_formal_publish_with_allow_still_requires_existing_gates(self) -> None:
        self.write_review_artifacts(
            codex_action="reject_with_rescue",
            verification_scope="local_plus_s2_or_openalex",
            deepseek_mode="opencode",
            codex_mode="codex-cli",
        )
        self.write_active_seed_support()
        survival = decide(run_date=RUN_DATE, allow_human_override=False, target_policy="formal")
        write_run_artifact(RUN_DATE, "survival-decision.json", survival, state="survival_decided")
        result = publish(RUN_DATE, dry_run=False, target_policy="formal")
        self.assert_legacy_formal_publish_disabled(result)

    def test_backfill_cannot_formal_publish(self) -> None:
        manifest = read_json(run_dir(RUN_DATE) / "manifest.json")
        manifest["backfill_mode"] = "ingest-only"
        write_json(run_dir(RUN_DATE) / "manifest.json", manifest)
        self.write_review_artifacts()
        survival = decide(run_date=RUN_DATE, allow_human_override=False)
        write_run_artifact(RUN_DATE, "survival-decision.json", survival, state="survival_decided")
        result = publish(RUN_DATE, dry_run=False, target_policy="formal")
        self.assert_legacy_formal_publish_disabled(result)

    def test_publish_actual_staged_rename_writes_required_artifacts(self) -> None:
        self.write_review_artifacts(
            verification_scope="local_plus_s2_or_openalex",
            deepseek_mode="opencode",
            codex_mode="codex-cli",
        )
        self.write_active_seed_support()
        survival = decide(run_date=RUN_DATE, allow_human_override=False, target_policy="formal")
        write_run_artifact(RUN_DATE, "survival-decision.json", survival, state="survival_decided")
        result = publish(RUN_DATE, dry_run=False, target_policy="formal")
        self.assert_legacy_formal_publish_disabled(result)
        manifest = json.loads((run_dir(RUN_DATE) / "manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest.get("formal_seed_written", False))

    def test_publish_existing_target_blocks_duplicate_no_overwrite(self) -> None:
        self.write_review_artifacts(
            verification_scope="local_plus_s2_or_openalex",
            deepseek_mode="opencode",
            codex_mode="codex-cli",
        )
        self.write_active_seed_support()
        survival = decide(run_date=RUN_DATE, allow_human_override=False, target_policy="formal")
        write_run_artifact(RUN_DATE, "survival-decision.json", survival, state="survival_decided")
        seed_dir = Path(self.tmp.name) / "idea_bank" / "seed" / "anchored-contact-failure-benchmark"
        seed_dir.mkdir(parents=True, exist_ok=True)
        marker = seed_dir / "idea.md"
        marker.write_text("existing", encoding="utf-8")
        result = publish(RUN_DATE, dry_run=False, target_policy="formal")
        self.assertEqual(result["status"], LEGACY_FORMAL_DISABLED_STATUS)
        self.assertEqual(result["published"], [])
        self.assertEqual(marker.read_text(encoding="utf-8"), "existing")
        duplicate_reports = list((Path(self.tmp.name) / "seed-candidates" / "duplicate-review" / RUN_DATE).glob("*.json"))
        self.assertEqual(len(duplicate_reports), 0)

    def test_concurrent_publish_lock_blocks_same_slug(self) -> None:
        self.write_review_artifacts(
            verification_scope="local_plus_s2_or_openalex",
            deepseek_mode="opencode",
            codex_mode="codex-cli",
        )
        self.write_active_seed_support()
        survival = decide(run_date=RUN_DATE, allow_human_override=False, target_policy="formal")
        write_run_artifact(RUN_DATE, "survival-decision.json", survival, state="survival_decided")
        lock = Path(self.tmp.name) / "idea_bank" / "seed" / "anchored-contact-failure-benchmark.publish.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text("held", encoding="utf-8")
        result = publish(RUN_DATE, dry_run=False, target_policy="formal")
        self.assertEqual(result["status"], LEGACY_FORMAL_DISABLED_STATUS)
        self.assertEqual(result["published"], [])
        self.assertEqual(list((Path(self.tmp.name) / "idea_bank" / "seed").glob("anchored-contact-failure-benchmark")), [])

    def test_missing_required_file_after_publish_moves_to_quarantine(self) -> None:
        self.write_review_artifacts(
            verification_scope="local_plus_s2_or_openalex",
            deepseek_mode="opencode",
            codex_mode="codex-cli",
        )
        self.write_active_seed_support()
        survival = decide(run_date=RUN_DATE, allow_human_override=False, target_policy="formal")
        write_run_artifact(RUN_DATE, "survival-decision.json", survival, state="survival_decided")
        result = publish(RUN_DATE, dry_run=False, target_policy="formal")
        self.assertEqual(result["status"], LEGACY_FORMAL_DISABLED_STATUS)
        self.assertEqual(result["published"], [])
        quarantine = Path(self.tmp.name) / "quarantine"
        self.assertFalse(any(path.is_dir() for path in quarantine.glob("anchored-contact-failure-benchmark.*")))

    def test_artifact_hash_mismatch_blocks_validation(self) -> None:
        self.write_review_artifacts()
        survival = decide(run_date=RUN_DATE, allow_human_override=False)
        write_run_artifact(RUN_DATE, "survival-decision.json", survival, state="survival_decided")
        selected_path = artifact_dir(RUN_DATE) / "selected-candidates.json"
        original_hash = file_sha256(selected_path)
        data = json.loads(selected_path.read_text(encoding="utf-8"))
        data["selected"][0]["title"] = "Tampered"
        selected_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        self.assertNotEqual(original_hash, file_sha256(selected_path))
        validation = validate_run(RUN_DATE, strict_publish=True)
        self.assertEqual(validation["status"], "partial_schema_blocked")
        self.assertTrue(any("hash_mismatch:selected-candidates.json" in item for item in validation["errors"]))

    def test_publish_preflight_ignores_stale_previous_publish_result_hashes(self) -> None:
        self.write_review_artifacts()
        survival = decide(run_date=RUN_DATE, allow_human_override=False)
        write_run_artifact(RUN_DATE, "survival-decision.json", survival, state="survival_decided")
        write_run_artifact(
            RUN_DATE,
            "publish-result.json",
            {
                "schema_version": "publish_result.v1",
                "run_date": RUN_DATE,
                "status": "blocked_validation",
                "v2_publish_policy": "seed-candidates-only",
                "formal_seed_publish_allowed": False,
                "formal_seed_written": False,
                "formal_rehearsal_written": False,
                "published": [],
                "bucketed": [],
                "blocked": ["previous failure"],
                "artifact_hashes": {"selected-candidates.json": "0" * 64},
            },
            state="publish_checked",
        )
        validation = validate_run(RUN_DATE, strict_publish=True)
        self.assertEqual(validation["status"], "partial_schema_blocked")
        self.assertTrue(any("publish-result.json:hash_mismatch:selected-candidates.json" in item for item in validation["errors"]))

        result = publish(RUN_DATE, dry_run=True, target_policy="seed-candidates-only")

        self.assertEqual(result["status"], "success")
        self.assertGreaterEqual(len(result["bucketed"]), 1)

    def test_duplicate_guard_blocks_existing_seed_title(self) -> None:
        seed_dir = Path(self.tmp.name) / "idea_bank" / "seed" / "anchored-contact-failure-benchmark"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "idea.md").write_text("# Anchored Contact Failure Benchmark\n", encoding="utf-8")
        guard = duplicate_guard(self.candidate())
        self.assertEqual(guard["status"], "possible_duplicate")
        self.assertEqual(guard["action"], "block")

    def test_raw_candidate_stage_has_no_legacy_write_seed_function(self) -> None:
        ideate_text = (SCRIPTS_DIR / "research_agenda_ideate.py").read_text(encoding="utf-8")
        update_text = (SCRIPTS_DIR / "research_agenda_update.py").read_text(encoding="utf-8")
        self.assertNotIn("def write_seed_folder", ideate_text)
        self.assertNotIn("write_seed_folder(", update_text)

    def test_core_paper_axes_are_explicit_without_industrial_constraint_axis(self) -> None:
        ideate_text = (SCRIPTS_DIR / "research_agenda_ideate.py").read_text(encoding="utf-8")
        prompt_text = (SCRIPTS_DIR / "generate_gemini_idea_prompt.py").read_text(encoding="utf-8")
        contract_text = (Path("projects/research-agenda/workflow-contracts/gemini-greenhouse-contract.md")).read_text(encoding="utf-8")
        combined = "\n".join([ideate_text, prompt_text, contract_text]).lower()
        for expected in [
            "vla/vlm/rl-token/action-interface",
            "tactile/force/contact-rich",
            "sim-to-real/robustness/failure recovery",
            "dlo/bimanual",
        ]:
            self.assertIn(expected, combined)
        self.assertNotIn("industrial", combined)

    def test_audit_detects_variable_path_seed_writer(self) -> None:
        fake = """
from research_agenda_common import IDEA_BANK_DIR
import shutil
def bad(source, recommended):
    target = IDEA_BANK_DIR / recommended / source.name
    shutil.move(str(source), str(target))
"""
        issues = _scan_text_for_seed_writer(Path(self.tmp.name) / "fake.py", fake)
        self.assertTrue(any(issue["code"] == "v2_direct_seed_writer_outside_publish" for issue in issues))

    def test_scheduled_daily_wrapper_does_not_pass_formal_publish_flags(self) -> None:
        wrapper = SCRIPTS_DIR / "run_daily_arxiv_task.ps1"
        text = wrapper.read_text(encoding="utf-8")
        self.assertNotIn("--allow-formal-seed-publish", text)
        self.assertNotIn("--allow-test-provider-for-formal", text)
        self.assertNotRegex(text, r"--v2-publish-policy\s+['\"]?formal\b")

    def test_register_daily_default_dry_run_has_no_provider_or_formal_flags(self) -> None:
        text = self.register_daily_arxiv_dry_run()
        self.assertIn("ActionArguments:", text)
        self.assertIn("run_daily_arxiv_task.ps1", text)
        self.assertNotIn("-DeepSeekProvider", text)
        self.assertNotIn("-CodexExecutionProvider", text)
        self.assertNotIn("--v2-publish-policy formal", text)
        self.assertNotIn("--allow-formal-seed-publish", text)
        self.assertNotIn("--allow-test-provider-for-formal", text)

    def test_registered_action_contains_quoted_file_path(self) -> None:
        text = self.register_daily_arxiv_dry_run()
        self.assertIn('-File "<vault-root>\\.claude\\scripts\\run_daily_arxiv_task.ps1"', text)

    def test_register_daily_explicit_provider_dry_run_has_only_safe_wrapper_flags(self) -> None:
        text = self.register_daily_arxiv_dry_run(
            "-DeepSeekProvider",
            "opencode",
            "-CodexExecutionProvider",
            "codex-cli",
        )
        self.assertIn("ActionArguments:", text)
        self.assertIn("-DeepSeekProvider opencode", text)
        self.assertIn("-CodexExecutionProvider codex-cli", text)
        self.assertNotIn("--v2-publish-policy formal", text)
        self.assertNotIn("--allow-formal-seed-publish", text)
        self.assertNotIn("--allow-test-provider-for-formal", text)
        self.assertNotIn("--commit-active-seed", text)

    def test_register_daily_openai_compatible_provider_dry_run_is_safe(self) -> None:
        text = self.register_daily_arxiv_dry_run(
            "-DeepSeekProvider",
            "openai-compatible",
            "-CodexExecutionProvider",
            "codex-cli",
        )
        self.assertIn("-DeepSeekProvider openai-compatible", text)
        self.assertIn("-CodexExecutionProvider codex-cli", text)
        self.assertNotIn("--v2-publish-policy formal", text)
        self.assertNotIn("--allow-formal-seed-publish", text)
        self.assertNotIn("--commit-active-seed", text)

    def test_registered_action_handles_non_ascii_vault_path(self) -> None:
        vault_root = Path(self.tmp.name) / "胡至伦 vault"
        script_dir = vault_root / ".claude" / "scripts"
        script_dir.mkdir(parents=True)
        script_path = script_dir / "register_daily_arxiv_task.ps1"
        shutil.copy2(SCRIPTS_DIR / "register_daily_arxiv_task.ps1", script_path)
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path),
                "-DryRun",
                "-ShowLocalPaths",
                "-Time",
                "12:00",
            ],
            cwd=vault_root,
            capture_output=True,
            encoding="utf-8",
            text=True,
            timeout=30,
        )
        text = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, text)
        self.assertIn("胡至伦", text)
        self.assertIn('-File "', text)
        self.assertIn("run_daily_arxiv_task.ps1", text)

    def test_run_daily_explicit_provider_params_translate_to_pipeline_flags(self) -> None:
        wrapper = SCRIPTS_DIR / "run_daily_arxiv_task.ps1"
        text = wrapper.read_text(encoding="utf-8")
        self.assertIn('[string]$DeepSeekProvider = "none"', text)
        self.assertIn('[string]$CodexExecutionProvider = "none"', text)
        self.assertIn('if ($DeepSeekProvider -ne "none")', text)
        self.assertIn('$PipelineArgs += @("--deepseek-provider", $DeepSeekProvider)', text)
        self.assertIn('if ($CodexExecutionProvider -ne "none")', text)
        self.assertIn('$PipelineArgs += @("--codex-execution-provider", $CodexExecutionProvider)', text)
        self.assertNotIn('"--deepseek-provider", "none"', text)
        self.assertNotIn('"--codex-execution-provider", "none"', text)

    def test_run_daily_wrapper_uses_codex_controlled_read_with_one_network_retry(self) -> None:
        wrapper = SCRIPTS_DIR / "run_daily_arxiv_task.ps1"
        text = wrapper.read_text(encoding="utf-8")
        self.assertIn('"--read-mode"', text)
        self.assertIn('"codex-controlled"', text)
        self.assertIn('"--read-retries"', text)
        self.assertIn('"--read-retries",\n    "1"', text)
        self.assertNotIn('"--allow-dangerous-claude"', text)

    def test_jsonschema_draft202012_validator_path_is_active(self) -> None:
        self.assertTrue(schema_validator_available())
        bad_payload = {
            "schema_version": "deepseek_review.v1",
            "run_date": RUN_DATE,
            "status": "not_a_valid_status",
            "reviews": [
                {
                    "candidate_id": "",
                    "status": "not_a_valid_review_status",
                    "novelty_attack": "",
                    "baseline_attack": "",
                    "mechanism_attack": "",
                    "evaluation_attack": "",
                    "scope_attack": "",
                    "a_plus_b_risk": "invalid",
                    "fatal_flaw": False,
                    "rescue_mutation": "",
                    "survivability_label": "invalid",
                    "allowed_next_stage": "invalid",
                }
            ],
            "provider_status": {"provider": "json", "provider_backed": "yes"},
        }
        errors = validate_payload(bad_payload, "deepseek_review.v1")
        self.assertTrue(any(error.startswith("jsonschema:") for error in errors), errors)
        self.assertFalse(any(error.startswith(("enum_mismatch:", "type_mismatch:")) for error in errors), errors)

    def test_docs_describe_provider_free_scheduled_fail_closed_boundary(self) -> None:
        docs = "\n".join(
            (REPO_ROOT / path).read_text(encoding="utf-8")
            for path in ["README.md", "README_EN.md", "docs/AUTOMATION.md"]
        )
        normalized = docs.lower()
        for phrase in [
            "provider-free",
            "explicit provider",
            "fail-closed",
            "partial",
            "seed-candidates-only",
            "formal seed",
        ]:
            self.assertIn(phrase, normalized)
        self.assertIn("local_plus_arxiv_api", docs)
        self.assertIn("external_scope_arxiv_only_not_full_prior_art", docs)
        self.assertIn("OpenAlex", docs)
        self.assertIn("Semantic Scholar", docs)
        self.assertIn("Scheduled formal publish remains disabled", docs)

    def test_docs_do_not_claim_provider_status_is_unforgeable(self) -> None:
        docs = "\n".join(
            (REPO_ROOT / path).read_text(encoding="utf-8")
            for path in ["README.md", "README_EN.md", "docs/AUTOMATION.md"]
        )
        normalized = docs.lower()
        self.assertIn("auditable", normalized)
        self.assertIn("not cryptographic proof", normalized)
        self.assertNotIn("provider status is unforgeable", normalized)
        self.assertNotIn("cryptographic proof that an external provider call happened.", normalized.replace("not cryptographic proof that an external provider call happened.", ""))

    def test_workflow_contracts_do_not_use_stale_formal_seed_wording(self) -> None:
        contract_root = REPO_ROOT / "projects" / "research-agenda" / "workflow-contracts"
        stale_hits: list[str] = []
        for path in contract_root.glob("*"):
            if path.suffix not in {".md", ".json"}:
                continue
            text = path.read_text(encoding="utf-8").lower()
            if "formal seed" in text or "formal_seed" in text or "idea_bank/seed" in text:
                stale_hits.append(path.name)
        self.assertEqual([], stale_hits)

    def test_audit_catches_policy_manifest_mismatch(self) -> None:
        manifest = read_json(run_dir(RUN_DATE) / "manifest.json")
        manifest["v2_publish_policy"] = "formal"
        manifest["formal_seed_publish_allowed"] = True
        manifest["formal_seed_written"] = True
        write_json(run_dir(RUN_DATE) / "manifest.json", manifest)
        write_run_artifact(
            RUN_DATE,
            "publish-result.json",
            {
                "schema_version": "publish_result.v1",
                "run_date": RUN_DATE,
                "status": "success",
                "v2_publish_policy": "seed-candidates-only",
                "formal_seed_publish_allowed": True,
                "formal_seed_written": True,
                "published": [{"candidate_id": "cand-alpha", "target": "x", "status": "seed_written"}],
                "bucketed": [],
                "blocked": [],
                "artifact_hashes": {},
            },
            state="seed_written",
        )
        _summary, issues = _audit_v2_state_machine(RUN_DATE)
        self.assertTrue(any(issue["code"] == "v2_publish_policy_mismatch" for issue in issues))

    def test_audit_catches_formal_seed_written_under_seed_candidates_only(self) -> None:
        manifest = read_json(run_dir(RUN_DATE) / "manifest.json")
        manifest["v2_publish_policy"] = "seed-candidates-only"
        manifest["formal_seed_publish_allowed"] = False
        manifest["formal_seed_written"] = True
        write_json(run_dir(RUN_DATE) / "manifest.json", manifest)
        write_run_artifact(
            RUN_DATE,
            "publish-result.json",
            {
                "schema_version": "publish_result.v1",
                "run_date": RUN_DATE,
                "status": "success",
                "v2_publish_policy": "seed-candidates-only",
                "formal_seed_publish_allowed": False,
                "formal_seed_written": True,
                "published": [{"candidate_id": "cand-alpha", "target": "x", "status": "seed_written"}],
                "bucketed": [],
                "blocked": [],
                "artifact_hashes": {},
            },
            state="seed_written",
        )
        _summary, issues = _audit_v2_state_machine(RUN_DATE)
        self.assertTrue(any(issue["code"] == "v2_formal_seed_written_policy_mismatch" for issue in issues))
        self.assertTrue(any(issue["code"] == "v2_formal_seed_written_without_allow" for issue in issues))

    def test_daily_pipeline_v2_provider_command_assembly(self) -> None:
        args = Namespace(
            deepseek_provider="opencode",
            deepseek_provider_json="",
            deepseek_model="deepseek/test-model",
            deepseek_timeout=123,
            codex_execution_provider="codex-cli",
            codex_execution_provider_json="",
            idea_timeout=45,
            allow_human_override=True,
        )
        stages = build_v2_review_stages(args, RUN_DATE)
        by_name = {name: command for name, command, _timeout in stages}
        by_timeout = {name: timeout for name, _command, timeout in stages}
        self.assertIn("--provider", by_name["deepseek_review"])
        self.assertIn("opencode", by_name["deepseek_review"])
        self.assertIn("--model", by_name["deepseek_review"])
        self.assertIn("deepseek/test-model", by_name["deepseek_review"])
        self.assertIn("--timeout", by_name["deepseek_review"])
        self.assertIn("123", by_name["deepseek_review"])
        self.assertIn("--provider-retries", by_name["deepseek_review"])
        self.assertIn("1", by_name["deepseek_review"])
        self.assertGreater(by_timeout["deepseek_review"], 123)
        self.assertGreaterEqual(by_timeout["deepseek_review"], 1800)
        self.assertIn("--provider", by_name["codex_execution_review"])
        self.assertIn("codex-cli", by_name["codex_execution_review"])
        self.assertIn("--target-policy", by_name["novelty_scan"])
        self.assertIn("seed-candidates-only", by_name["novelty_scan"])
        self.assertIn("--target-policy", by_name["survival_decision"])
        self.assertIn("--allow-human-override", by_name["survival_decision"])

    def test_build_v2_review_stages_deepseek_opencode_has_exactly_one_provider(self) -> None:
        cmd = self.review_stage_commands(deepseek_provider="opencode")["deepseek_review"]
        self.assertEqual(cmd.count("--provider"), 1)
        self.assertEqual(cmd[cmd.index("--provider") + 1], "opencode")
        self.assertEqual(cmd[cmd.index("--provider-retries") + 1], "1")

    def test_build_v2_review_stages_deepseek_openai_compatible_has_exactly_one_provider(self) -> None:
        cmd = self.review_stage_commands(deepseek_provider="openai-compatible")["deepseek_review"]
        self.assertEqual(cmd.count("--provider"), 1)
        self.assertEqual(cmd[cmd.index("--provider") + 1], "openai-compatible")
        self.assertIn("--model", cmd)

    def test_build_v2_review_stages_codex_cli_has_exactly_one_provider(self) -> None:
        cmd = self.review_stage_commands(codex_execution_provider="codex-cli")["codex_execution_review"]
        self.assertEqual(cmd.count("--provider"), 1)
        self.assertEqual(cmd[cmd.index("--provider") + 1], "codex-cli")

    def test_build_v2_review_stages_json_provider_has_exactly_one_provider(self) -> None:
        commands = self.review_stage_commands(deepseek_provider_json="deepseek-provider.json", codex_execution_provider_json="codex-provider.json")
        for name in ["deepseek_review", "codex_execution_review"]:
            cmd = commands[name]
            self.assertEqual(cmd.count("--provider"), 1)
            self.assertEqual(cmd[cmd.index("--provider") + 1], "json")
            self.assertIn("--provider-review-json", cmd)

    def test_build_v2_review_stages_provider_none_has_exactly_one_provider(self) -> None:
        commands = self.review_stage_commands()
        for name in ["deepseek_review", "codex_execution_review"]:
            cmd = commands[name]
            self.assertEqual(cmd.count("--provider"), 1)
            self.assertEqual(cmd[cmd.index("--provider") + 1], "none")

    def test_build_v2_review_stages_rejects_provider_json_conflict(self) -> None:
        with self.assertRaisesRegex(ValueError, "provider_json_conflicts_with_explicit_provider"):
            self.review_stage_commands(deepseek_provider="opencode", deepseek_provider_json="deepseek-provider.json")

    def test_build_v2_review_stages_preserves_seed_candidates_only_policy(self) -> None:
        commands = self.review_stage_commands(v2_publish_policy="formal")
        for name in ["novelty_scan", "survival_decision"]:
            cmd = commands[name]
            self.assertEqual(cmd[cmd.index("--target-policy") + 1], "seed-candidates-only")

    def test_pipeline_provider_args_single_provider_deepseek(self) -> None:
        cmd = ["python", "deepseek_scientific_review.py"]
        append_provider_args(cmd, provider="opencode", model="deepseek/test-model", timeout=123, provider_retries=1)
        self.assertEqual(cmd.count("--provider"), 1)
        self.assertIn("opencode", cmd)
        self.assertNotIn("none", cmd)
        self.assertIn("--model", cmd)
        self.assertIn("--timeout", cmd)
        self.assertEqual(cmd[cmd.index("--provider-retries") + 1], "1")

    def test_pipeline_provider_args_single_provider_deepseek_openai_compatible(self) -> None:
        cmd = ["python", "deepseek_scientific_review.py"]
        append_provider_args(cmd, provider="openai-compatible", model="abrdns/deepseek-v4-pro", timeout=123)
        self.assertEqual(cmd.count("--provider"), 1)
        self.assertIn("openai-compatible", cmd)
        self.assertNotIn("none", cmd)
        self.assertIn("--model", cmd)
        self.assertIn("--timeout", cmd)

    def test_pipeline_provider_args_single_provider_codex(self) -> None:
        cmd = ["python", "codex_seed_review.py"]
        append_provider_args(cmd, provider="codex-cli", timeout=1200)
        self.assertEqual(cmd.count("--provider"), 1)
        self.assertIn("codex-cli", cmd)
        self.assertNotIn("none", cmd)

    def test_pipeline_provider_json_and_explicit_provider_conflict(self) -> None:
        cmd = ["python", "deepseek_scientific_review.py"]
        with self.assertRaisesRegex(ValueError, "provider_json_conflicts_with_explicit_provider"):
            append_provider_args(cmd, provider="opencode", provider_json="provider.json")

    def test_pipeline_provider_args_reject_duplicate_provider(self) -> None:
        cmd = ["python", "deepseek_scientific_review.py", "--provider", "none"]
        with self.assertRaisesRegex(ValueError, "provider_arg_already_present"):
            append_provider_args(cmd, provider="opencode")

    def test_pipeline_provider_json_uses_single_json_provider(self) -> None:
        cmd = ["python", "deepseek_scientific_review.py"]
        append_provider_args(cmd, provider="none", provider_json="provider.json")
        self.assertEqual(cmd.count("--provider"), 1)
        self.assertIn("json", cmd)
        self.assertIn("--provider-review-json", cmd)

    def test_wrapper_quality_audit_failure_fails_scheduled_task(self) -> None:
        text = (SCRIPTS_DIR / "run_daily_arxiv_task.ps1").read_text(encoding="utf-8")
        self.assertIn("if ($QualityExitCode -ne 0)", text)
        self.assertIn("$ExitCode = $QualityExitCode", text)
        self.assertIn("exit $ExitCode", text)

    def test_run_daily_wrapper_quality_audit_failure_exits_nonzero(self) -> None:
        result, _root = self.run_daily_wrapper_test(pipeline_exit=0, quality_exit=7)
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)

    def test_run_daily_wrapper_pipeline_success_quality_failure_exits_quality_code(self) -> None:
        result, _root = self.run_daily_wrapper_test(pipeline_exit=0, recovery_exit=0, quality_exit=9)
        self.assertEqual(result.returncode, 9, result.stdout + result.stderr)

    def test_run_daily_wrapper_lock_released_on_quality_failure(self) -> None:
        result, root = self.run_daily_wrapper_test(pipeline_exit=0, quality_exit=7)
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        self.assertFalse((root / "projects" / "arxiv-daily" / "daily_arxiv_pipeline.lock").exists())

    def test_run_daily_wrapper_forbidden_formal_active_args_rejected(self) -> None:
        result, _root = self.run_daily_wrapper_test(extra_args=("--allow-human-override",))
        self.assertNotEqual(result.returncode, 0)

    def test_scheduled_wrapper_forbids_human_override_and_governance_mutation_args(self) -> None:
        text = (SCRIPTS_DIR / "run_daily_arxiv_task.ps1").read_text(encoding="utf-8")
        for token in [
            '"--allow-human" + "-override"',
            '"formal_rehearsal" + "_packet.py"',
            '"governance" + "_review.py"',
            '"active_seed" + "_commit.py"',
            '"strategy" + "_ledger.py"',
            '"--apply" + "-strategy"',
            '"--active-seed" + "-id"',
        ]:
            self.assertIn(token, text)

    def test_moved_to_v2_state_machine_only_suppresses_old_battle(self) -> None:
        issues = _moved_to_v2_missing_artifact_issues(RUN_DATE, mandatory_battle_status="moved_to_v2_state_machine", v2_deepseek_ok=True)
        codes = {issue["code"] for issue in issues}
        self.assertNotIn("mandatory_model_battle_not_success", codes)
        self.assertIn("v2_missing_codex_execution_review", codes)

    def test_moved_to_v2_does_not_suppress_missing_codex(self) -> None:
        issues = _moved_to_v2_missing_artifact_issues(RUN_DATE, mandatory_battle_status="moved_to_v2_state_machine", v2_deepseek_ok=True)
        self.assertTrue(any(issue["code"] == "v2_missing_codex_execution_review" for issue in issues))

    def test_moved_to_v2_does_not_suppress_missing_novelty_survival_publish(self) -> None:
        issues = _moved_to_v2_missing_artifact_issues(RUN_DATE, mandatory_battle_status="moved_to_v2_state_machine", v2_deepseek_ok=True)
        codes = {issue["code"] for issue in issues}
        self.assertIn("v2_missing_novelty_scan", codes)
        self.assertIn("v2_missing_survival_decision", codes)
        self.assertIn("v2_missing_publish_result", codes)

    def test_moved_to_v2_state_machine_missing_publish_result_still_fails(self) -> None:
        issues = _moved_to_v2_missing_artifact_issues(RUN_DATE, mandatory_battle_status="moved_to_v2_state_machine", v2_deepseek_ok=True)
        self.assertTrue(any(issue["code"] == "v2_missing_publish_result" for issue in issues))

    def test_moved_to_v2_state_machine_deepseek_not_provider_backed_still_fails(self) -> None:
        write_run_artifact(
            RUN_DATE,
            "deepseek-review.json",
            {"schema_version": "deepseek_review.v1", "run_date": RUN_DATE, "status": "partial_provider_unavailable", "provider_status": {"provider": "deepseek", "provider_backed": False, "mode": "opencode"}, "reviews": [], "artifact_hashes": {}},
            state="attacked_by_deepseek",
        )
        issue = _provider_backed_artifact_issue(run_date=RUN_DATE, artifact_name="deepseek-review.json", expected_mode="opencode", code="scheduled_deepseek_provider_not_provider_backed")
        self.assertIsNotNone(issue)
        self.assertEqual(issue["level"], "FAIL")

    def test_scheduled_provider_artifacts_not_expected_without_raw_candidates(self) -> None:
        self.assertFalse(_scheduled_provider_artifacts_expected(greenhouse_exists=False, raw_candidate_count=0))
        self.assertFalse(_scheduled_provider_artifacts_expected(greenhouse_exists=True, raw_candidate_count=0))
        self.assertTrue(_scheduled_provider_artifacts_expected(greenhouse_exists=True, raw_candidate_count=1))

    def test_audit_fails_when_scheduled_deepseek_provider_not_provider_backed(self) -> None:
        write_run_artifact(
            RUN_DATE,
            "deepseek-review.json",
            {"schema_version": "deepseek_review.v1", "run_date": RUN_DATE, "status": "partial_provider_unavailable", "provider_status": {"provider": "deepseek", "provider_backed": False, "mode": "opencode"}, "reviews": [], "artifact_hashes": {}},
            state="attacked_by_deepseek",
        )
        issue = _provider_backed_artifact_issue(run_date=RUN_DATE, artifact_name="deepseek-review.json", expected_mode="opencode", code="scheduled_deepseek_provider_not_provider_backed")
        self.assertIsNotNone(issue)
        self.assertEqual(issue["level"], "FAIL")

    def test_audit_warns_for_scheduled_deepseek_candidate_level_fail_closed(self) -> None:
        payload = {
            "schema_version": "deepseek_review.v1",
            "run_date": RUN_DATE,
            "status": "partial_provider_invalid",
            "provider_status": {
                "provider": "deepseek",
                "provider_backed": False,
                "mode": "opencode",
                "candidate_level_fail_closed": True,
                "provider_backed_success_count": 1,
                "provider_fallback_count": 1,
            },
            "reviews": [
                {**self.provider_deepseek_payload()["reviews"][0], "candidate_id": "cand-alpha", "status": "success"},
                {
                    **self.provider_deepseek_payload()["reviews"][0],
                    "candidate_id": "cand-beta",
                    "status": "failed_fallback_only",
                    "survivability_label": "reject_fatal",
                    "allowed_next_stage": "stop",
                },
            ],
            "artifact_hashes": {},
        }
        write_run_artifact(RUN_DATE, "deepseek-review.json", payload, state="attacked_by_deepseek")

        issue = _provider_backed_artifact_issue(
            run_date=RUN_DATE,
            artifact_name="deepseek-review.json",
            expected_mode="opencode",
            code="scheduled_deepseek_provider_not_provider_backed",
            allow_candidate_fail_closed=True,
        )

        self.assertIsNotNone(issue)
        self.assertEqual(issue["level"], "WARN")
        self.assertEqual(issue["code"], "scheduled_deepseek_provider_candidate_level_fail_closed")
        self.assertTrue(_v2_deepseek_review_covers_battle(payload))

    def test_audit_still_fails_when_deepseek_fail_closed_has_no_provider_success(self) -> None:
        payload = {
            "schema_version": "deepseek_review.v1",
            "run_date": RUN_DATE,
            "status": "partial_provider_unavailable",
            "provider_status": {
                "provider": "deepseek",
                "provider_backed": False,
                "mode": "opencode",
                "candidate_level_fail_closed": True,
                "provider_backed_success_count": 0,
                "provider_fallback_count": 1,
            },
            "reviews": [
                {
                    **self.provider_deepseek_payload()["reviews"][0],
                    "candidate_id": "cand-alpha",
                    "status": "failed_fallback_only",
                    "survivability_label": "reject_fatal",
                    "allowed_next_stage": "stop",
                }
            ],
            "artifact_hashes": {},
        }
        write_run_artifact(RUN_DATE, "deepseek-review.json", payload, state="attacked_by_deepseek")

        issue = _provider_backed_artifact_issue(
            run_date=RUN_DATE,
            artifact_name="deepseek-review.json",
            expected_mode="opencode",
            code="scheduled_deepseek_provider_not_provider_backed",
            allow_candidate_fail_closed=True,
        )

        self.assertIsNotNone(issue)
        self.assertEqual(issue["level"], "FAIL")
        self.assertEqual(issue["code"], "scheduled_deepseek_provider_not_provider_backed")

    def test_audit_runtime_root_uses_temp_agenda_root(self) -> None:
        repo_artifact = REPO_ROOT / "projects" / "research-agenda" / "runs" / RUN_DATE / "artifacts" / "deepseek-review.json"
        self.assertFalse(repo_artifact.exists())
        write_run_artifact(
            RUN_DATE,
            "deepseek-review.json",
            {"schema_version": "deepseek_review.v1", "run_date": RUN_DATE, "status": "success", "provider_status": {"provider": "deepseek", "provider_backed": True, "mode": "opencode", "status": "success"}, "reviews": [], "artifact_hashes": {}},
            state="attacked_by_deepseek",
        )
        issue = _provider_backed_artifact_issue(run_date=RUN_DATE, artifact_name="deepseek-review.json", expected_mode="opencode", code="scheduled_deepseek_provider_not_provider_backed")
        self.assertIsNone(issue)

    def test_audit_accepts_scheduled_deepseek_openai_compatible_provider_backed(self) -> None:
        write_run_artifact(
            RUN_DATE,
            "deepseek-review.json",
            {
                "schema_version": "deepseek_review.v1",
                "run_date": RUN_DATE,
                "status": "success",
                "provider_status": {"provider": "deepseek", "provider_backed": True, "mode": "openai-compatible", "status": "success"},
                "reviews": [],
                "artifact_hashes": {},
            },
            state="attacked_by_deepseek",
        )
        issue = _provider_backed_artifact_issue(run_date=RUN_DATE, artifact_name="deepseek-review.json", expected_mode="openai-compatible", code="scheduled_deepseek_provider_not_provider_backed")
        self.assertIsNone(issue)

    def test_workflow_contracts_still_read_from_repo_root(self) -> None:
        fake_contract_dir = Path(self.tmp.name) / "projects" / "research-agenda" / "workflow-contracts"
        fake_contract_dir.mkdir(parents=True)
        self.assertFalse(str(CONTRACT_DIR).startswith(str(fake_contract_dir.parent)))
        self.assertTrue((CONTRACT_DIR / "daily-pipeline-contract.md").exists())

    def test_audit_fails_when_scheduled_codex_provider_not_provider_backed(self) -> None:
        write_run_artifact(
            RUN_DATE,
            "codex-execution-review.json",
            {"schema_version": "codex_execution_review.v1", "run_date": RUN_DATE, "status": "partial_provider_unavailable", "provider_status": {"provider": "codex", "provider_backed": False, "mode": "codex-cli"}, "reviews": [], "artifact_hashes": {}},
            state="execution_reviewed",
        )
        issue = _provider_backed_artifact_issue(run_date=RUN_DATE, artifact_name="codex-execution-review.json", expected_mode="codex-cli", code="scheduled_codex_provider_not_provider_backed")
        self.assertIsNotNone(issue)
        self.assertEqual(issue["level"], "FAIL")

    def test_survival_decision_not_called_only_promotion_brain(self) -> None:
        self.write_review_artifacts()
        payload = decide(run_date=RUN_DATE, allow_human_override=False)
        self.assertNotIn("only promotion brain", payload["boundary"])
        self.assertIn("Legacy v2 triage decision only", payload["boundary"])

    def test_survival_active_fields_are_legacy_or_hard_false(self) -> None:
        self.write_review_artifacts()
        payload = decide(run_date=RUN_DATE, allow_human_override=False)
        decision = payload["decisions"][0]
        self.assertFalse(decision["active_seed_allowed"])
        self.assertFalse(decision["legacy_active_seed_allowed"])
        self.assertTrue(decision["requires_human_governance"])

    def test_candidate_only_governance_gaps_are_info_not_warn_in_daily_quality_audit(self) -> None:
        self.write_review_artifacts()
        survival = decide(run_date=RUN_DATE, allow_human_override=False)
        write_run_artifact(RUN_DATE, "survival-decision.json", survival, state="survival_decided")
        seed_dashboard.write_dashboard(RUN_DATE, latest=False, dry_run=False)

        _summary, issues = _audit_v2_state_machine(RUN_DATE)
        governance_codes = {
            "manual_prior_art_review_missing",
            "anchorless_core_evidence_risk",
            "active_seed_without_pilot_plan",
        }
        matched = [issue for issue in issues if issue["code"] in governance_codes]
        self.assertTrue(matched)
        self.assertEqual({issue["level"] for issue in matched}, {"INFO"})

    def test_audit_ignores_dashboard_multi_summary_row(self) -> None:
        self.write_review_artifacts()
        survival = decide(run_date=RUN_DATE, allow_human_override=False)
        write_run_artifact(RUN_DATE, "survival-decision.json", survival, state="survival_decided")
        dashboard = seed_dashboard.write_dashboard(RUN_DATE, latest=False, dry_run=False)
        aggregate = {
            **dashboard["rows"][0],
            "candidate_id": "__multi__",
            "title": "",
            "risk_markers": [],
            "blocking_reasons": [],
        }
        dashboard["rows"].insert(0, aggregate)
        write_run_artifact(RUN_DATE, "active-seed-dashboard.json", dashboard, state="dashboard_rendered")

        _summary, issues = _audit_v2_state_machine(RUN_DATE)
        self.assertFalse(any(issue["code"] == "active_seed_dashboard_extra_row" for issue in issues))

    def test_publish_formal_parser_rejects_or_disabled_before_writes(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "publish_research_run.py"), "--run-date", RUN_DATE, "--target-policy", "formal", "--dry-run"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=30,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((Path(self.tmp.name) / "runs" / RUN_DATE / "publish" / "publish-result.json").exists())

    def test_no_idea_bank_seed_mutation_except_migration_tool(self) -> None:
        publish_text = (SCRIPTS_DIR / "publish_research_run.py").read_text(encoding="utf-8")
        migrate_text = (SCRIPTS_DIR / "migrate_v03_to_v10.py").read_text(encoding="utf-8")
        self.assertNotIn("idea_bank/seed", publish_text)
        self.assertIn("idea_bank/seed", migrate_text)

    def test_publish_research_run_no_longer_writes_idea_bank_seed(self) -> None:
        publish_text = (SCRIPTS_DIR / "publish_research_run.py").read_text(encoding="utf-8")
        self.assertNotIn("_write_seed_stage", publish_text)
        self.assertNotIn("publish.lock", publish_text)
        self.assertNotIn("--allow-formal-seed-publish", publish_text)

    def test_migrate_v03_to_v10_is_only_legacy_marker_writer(self) -> None:
        publish_text = (SCRIPTS_DIR / "publish_research_run.py").read_text(encoding="utf-8")
        self.assertNotIn("migrate_legacy_status", publish_text)
        self.assertTrue((SCRIPTS_DIR / "migrate_v03_to_v10.py").exists())


if __name__ == "__main__":
    unittest.main()
