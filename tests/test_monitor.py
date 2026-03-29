# tests/test_monitor.py
"""
Tests for TokenMonitor, RunMetric, and SessionReport.

Senior pattern: we build fake LensResult objects using MagicMock
so tests run instantly without real API calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from contextlens.core.monitor import TokenMonitor, RunMetric, SessionReport


# ---------------------------------------------------------------------------
# Helpers — build fake LensResult without importing lens.py
# ---------------------------------------------------------------------------

def make_fake_result(
    run_id: str = "test-run-001",
    provider: str = "groq",
    model: str = "llama-3.1-70b-versatile",
    input_tokens: int = 200,
    output_tokens: int = 80,
    total_tokens: int = 280,
    context_tokens: int = 200,
    estimated_cost_usd: float = 0.000165,
    latency_ms: float = 620.0,
    wall_time_ms: float = 640.0,
    context_score: float | None = 78.5,
    context_grade: str | None = "B",
    tags: list[str] | None = None,
) -> MagicMock:
    result = MagicMock()

    result.response.run_id = run_id
    result.response.provider = provider
    result.response.model = model
    result.response.input_tokens = input_tokens
    result.response.output_tokens = output_tokens
    result.response.total_tokens = total_tokens
    result.response.estimated_cost_usd = estimated_cost_usd
    result.response.latency_ms = latency_ms

    result.context_entry.run_id = run_id
    result.context_entry.total_tokens = context_tokens
    result.context_entry.tags = tags or []

    result.total_wall_time_ms = wall_time_ms

    eval_mock = MagicMock()
    eval_mock.overall_score = context_score
    eval_mock.grade = context_grade
    result.evaluation = eval_mock if context_score is not None else None

    return result


# ---------------------------------------------------------------------------
# RunMetric tests
# ---------------------------------------------------------------------------

class TestRunMetric:
    def test_to_dict_rounds_cost(self):
        m = RunMetric(
            run_id="x",
            estimated_cost_usd=0.0001234567890,
            api_latency_ms=123.456789,
        )
        d = m.to_dict()
        assert d["estimated_cost_usd"] == round(0.0001234567890, 8)
        assert d["api_latency_ms"] == round(123.456789, 2)

    def test_default_timestamps_set(self):
        m = RunMetric(run_id="y")
        assert m.timestamp  # not empty


# ---------------------------------------------------------------------------
# TokenMonitor core tests
# ---------------------------------------------------------------------------

class TestTokenMonitor:
    @pytest.fixture
    def monitor(self, tmp_path):
        return TokenMonitor(
            session_id="test",
            budget_usd=0.01,
            log_dir=tmp_path,
        )

    def test_record_returns_run_metric(self, monitor):
        result = make_fake_result()
        metric = monitor.record(result)
        assert isinstance(metric, RunMetric)
        assert metric.provider == "groq"

    def test_run_count_increments(self, monitor):
        assert monitor.run_count == 0
        monitor.record(make_fake_result())
        monitor.record(make_fake_result(run_id="run-002"))
        assert monitor.run_count == 2

    def test_total_cost_accumulates(self, monitor):
        monitor.record(make_fake_result(estimated_cost_usd=0.001))
        monitor.record(make_fake_result(estimated_cost_usd=0.002, run_id="r2"))
        assert monitor.total_cost_usd == pytest.approx(0.003, rel=1e-4)

    def test_metrics_persisted_to_jsonl(self, monitor, tmp_path):
        monitor.record(make_fake_result())
        files = list(tmp_path.glob("metrics_*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert "run_id" in parsed
        assert "estimated_cost_usd" in parsed

    def test_multiple_runs_all_persisted(self, monitor, tmp_path):
        for i in range(5):
            monitor.record(make_fake_result(run_id=f"run-{i:03d}"))
        lines = list(tmp_path.glob("metrics_*.jsonl"))[0].read_text().strip().split("\n")
        assert len(lines) == 5

    def test_budget_exceeded_does_not_raise(self, monitor):
        """Budget alerts must never crash the application."""
        for i in range(20):
            monitor.record(make_fake_result(
                run_id=f"run-{i}",
                estimated_cost_usd=0.001,  # 20 runs × $0.001 = $0.02 > $0.01 budget
            ))
        # Must complete without raising
        assert monitor.total_cost_usd > monitor.budget_usd


# ---------------------------------------------------------------------------
# SessionReport tests
# ---------------------------------------------------------------------------

class TestSessionReport:
    @pytest.fixture
    def loaded_monitor(self, tmp_path):
        monitor = TokenMonitor(session_id="test", log_dir=tmp_path)
        monitor.record(make_fake_result(
            run_id="r1", provider="groq",
            total_tokens=300, estimated_cost_usd=0.000177, latency_ms=620.0,
            context_score=82.0, context_grade="B",
        ))
        monitor.record(make_fake_result(
            run_id="r2", provider="gemini",
            total_tokens=250, estimated_cost_usd=0.000019, latency_ms=1240.0,
            context_score=91.0, context_grade="A",
        ))
        monitor.record(make_fake_result(
            run_id="r3", provider="groq",
            total_tokens=180, estimated_cost_usd=0.000106, latency_ms=480.0,
            context_score=65.0, context_grade="D",
        ))
        return monitor

    def test_total_runs_correct(self, loaded_monitor):
        report = loaded_monitor.get_report()
        assert report.total_runs == 3

    def test_total_tokens_correct(self, loaded_monitor):
        report = loaded_monitor.get_report()
        assert report.total_tokens == 730

    def test_total_cost_correct(self, loaded_monitor):
        report = loaded_monitor.get_report()
        expected = 0.000177 + 0.000019 + 0.000106
        assert report.total_cost_usd == pytest.approx(expected, rel=1e-4)

    def test_latency_percentiles_exist(self, loaded_monitor):
        report = loaded_monitor.get_report()
        assert report.p50_latency_ms > 0
        assert report.p90_latency_ms >= report.p50_latency_ms
        assert report.p99_latency_ms >= report.p90_latency_ms

    def test_provider_breakdown_has_groq_and_gemini(self, loaded_monitor):
        report = loaded_monitor.get_report()
        assert "groq" in report.by_provider
        assert "gemini" in report.by_provider
        assert report.by_provider["groq"].runs == 2
        assert report.by_provider["gemini"].runs == 1

    def test_avg_context_score_correct(self, loaded_monitor):
        report = loaded_monitor.get_report()
        expected_avg = (82.0 + 91.0 + 65.0) / 3
        assert report.avg_context_score == pytest.approx(expected_avg, rel=1e-3)

    def test_runs_passed_evaluation_correct(self, loaded_monitor):
        report = loaded_monitor.get_report()
        # 82 and 91 pass (>= 70), 65 fails
        assert report.runs_passed_evaluation == 2

    def test_most_expensive_run_identified(self, loaded_monitor):
        report = loaded_monitor.get_report()
        assert report.most_expensive_run is not None
        assert report.most_expensive_run.run_id == "r1"

    def test_slowest_run_identified(self, loaded_monitor):
        report = loaded_monitor.get_report()
        assert report.slowest_run is not None
        assert report.slowest_run.run_id == "r2"

    def test_budget_status_no_budget(self, tmp_path):
        monitor = TokenMonitor(session_id="t", log_dir=tmp_path)
        monitor.record(make_fake_result())
        report = monitor.get_report()
        assert report.budget_status == "no_budget"

    def test_budget_status_warning(self, tmp_path):
        monitor = TokenMonitor(
            session_id="t", budget_usd=0.001, log_dir=tmp_path
        )
        monitor.record(make_fake_result(estimated_cost_usd=0.00085))  # 85% of budget
        report = monitor.get_report()
        assert report.budget_status == "warning"

    def test_budget_status_exceeded(self, tmp_path):
        monitor = TokenMonitor(
            session_id="t", budget_usd=0.001, log_dir=tmp_path
        )
        monitor.record(make_fake_result(estimated_cost_usd=0.0015))   # 150%
        report = monitor.get_report()
        assert report.budget_status == "exceeded"

    def test_report_serializes_to_dict(self, loaded_monitor):
        report = loaded_monitor.get_report()
        d = report.to_dict()
        assert "total_runs" in d
        assert "latency_percentiles_ms" in d
        assert "by_provider" in d
        assert "budget" in d
        assert "quality" in d

    def test_empty_session_returns_zero_report(self, tmp_path):
        monitor = TokenMonitor(session_id="empty", log_dir=tmp_path)
        report = monitor.get_report()
        assert report.total_runs == 0
        assert report.total_cost_usd == 0.0

    def test_export_creates_json_file(self, loaded_monitor, tmp_path):
        out_path = tmp_path / "test_export.json"
        loaded_monitor.export_report_json(out_path)
        assert out_path.exists()
        with open(out_path) as f:
            data = json.load(f)
        assert data["total_runs"] == 3


# ---------------------------------------------------------------------------
# Session persistence tests
# ---------------------------------------------------------------------------

class TestSessionPersistence:
    def test_metrics_survive_restart(self, tmp_path):
        """
        Critical: metrics recorded in one process must be
        readable in a new process (simulated by creating a new monitor
        instance pointing at the same log_dir).
        """
        monitor1 = TokenMonitor(session_id="persist", log_dir=tmp_path)
        monitor1.record(make_fake_result(run_id="pre-restart"))
        assert monitor1.run_count == 1

        # Simulate restart: new instance, same directory
        monitor2 = TokenMonitor(session_id="persist", log_dir=tmp_path)
        assert monitor2.run_count == 1  # loaded from disk
        assert monitor2._metrics[0].run_id == "pre-restart"