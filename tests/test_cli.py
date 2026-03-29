# tests/test_cli.py
"""
CLI tests using Typer's test runner.

We use CliRunner to invoke commands without spawning a subprocess.
This gives us coverage on argument parsing, output formatting,
and exit codes without real API calls.
"""

from pathlib import Path
import json

import pytest
from typer.testing import CliRunner

from contextlens.cli.commands import app

runner = CliRunner()


class TestCLIVersion:
    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "contextlens" in result.output.lower() or "0." in result.output


class TestCLIHelp:
    def test_root_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "compare" in result.output
        assert "analyze" in result.output
        assert "report" in result.output
        assert "check" in result.output

    def test_run_help(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--provider" in result.output
        assert "--optimize" in result.output

    def test_analyze_help(self):
        result = runner.invoke(app, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "--threshold" in result.output
        assert "--export" in result.output

    def test_report_help(self):
        result = runner.invoke(app, ["report", "--help"])
        assert result.exit_code == 0
        assert "--session" in result.output
        assert "--budget" in result.output

    def test_compare_help(self):
        result = runner.invoke(app, ["compare", "--help"])
        assert result.exit_code == 0
        assert "--providers" in result.output


class TestCLIAnalyze:
    def test_analyze_valid_log(self, tmp_path):
        """analyze should read a JSONL file and print a table."""
        log_file = tmp_path / "test_session_20241201.jsonl"
        entries = [
            {
                "run_id": f"run-{i:04d}-abcd-efgh-ijkl",
                "session_id": "test",
                "user_prompt": "How does context engineering work in production AI systems?",
                "system_prompt": "You are a senior AI engineer. Be precise and cite sources.",
                "retrieved_docs": [],
                "conversation_history": [],
                "system_tokens": 15,
                "user_tokens": 12,
                "docs_tokens": 0,
                "history_tokens": 0,
                "total_tokens": 27,
                "provider": "groq",
                "model": "llama-3.1-70b-versatile",
                "max_context_tokens": 8000,
                "tags": [],
            }
            for i in range(3)
        ]
        with open(log_file, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        result = runner.invoke(app, ["analyze", str(log_file)])
        assert result.exit_code in (0, 1)  # 0=all pass, 1=some fail
        assert "groq" in result.output

    def test_analyze_missing_file(self):
        result = runner.invoke(app, ["analyze", "nonexistent_file.jsonl"])
        assert result.exit_code != 0

    def test_analyze_empty_file(self, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        result = runner.invoke(app, ["analyze", str(empty)])
        assert result.exit_code == 2  # tool error

    def test_analyze_export(self, tmp_path):
        log_file = tmp_path / "test.jsonl"
        entry = {
            "run_id": "run-test-1234-5678-9012",
            "session_id": "test",
            "user_prompt": "What is context engineering?",
            "system_prompt": "You are an expert.",
            "retrieved_docs": [],
            "conversation_history": [],
            "system_tokens": 5, "user_tokens": 5,
            "docs_tokens": 0, "history_tokens": 0, "total_tokens": 10,
            "provider": "groq", "model": "llama-3.1-70b-versatile",
            "max_context_tokens": 8000, "tags": [],
        }
        log_file.write_text(json.dumps(entry) + "\n")
        export_path = tmp_path / "analysis_export.json"

        result = runner.invoke(app, [
            "analyze", str(log_file),
            "--export", str(export_path),
        ])
        assert export_path.exists()
        data = json.loads(export_path.read_text())
        assert isinstance(data, list)
        assert "score" in data[0]


class TestCLIReport:
    def test_report_no_data(self, tmp_path):
        result = runner.invoke(app, [
            "report",
            "--session", "nonexistent_session",
            "--log-dir", str(tmp_path),
        ])
        assert result.exit_code == 0
        assert "No runs" in result.output or "no data" in result.output.lower() or "No data" in result.output

    def test_report_with_metrics(self, tmp_path):
        """Create a fake metrics file and verify report reads it."""
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        metrics_file = tmp_path / f"metrics_test_{today}.jsonl"
        metric = {
            "run_id": "run-0001",
            "timestamp": "2024-12-01T10:00:00+00:00",
            "provider": "groq",
            "model": "llama-3.1-70b-versatile",
            "session_id": "test",
            "input_tokens": 200,
            "output_tokens": 80,
            "total_tokens": 280,
            "context_tokens": 200,
            "estimated_cost_usd": 0.000177,
            "api_latency_ms": 620.0,
            "wall_time_ms": 640.0,
            "context_score": 78.5,
            "context_grade": "B",
            "tags": [],
        }
        metrics_file.write_text(json.dumps(metric) + "\n")

        result = runner.invoke(app, [
            "report",
            "--session", "test",
            "--log-dir", str(tmp_path),
        ])
        assert result.exit_code == 0
        assert "groq" in result.output

    def test_report_export(self, tmp_path):
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        metrics_file = tmp_path / f"metrics_cli_{today}.jsonl"
        metric = {
            "run_id": "run-export-test",
            "timestamp": "2024-12-01T10:00:00+00:00",
            "provider": "groq", "model": "llama", "session_id": "cli",
            "input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
            "context_tokens": 100, "estimated_cost_usd": 0.0001,
            "api_latency_ms": 500.0, "wall_time_ms": 520.0,
            "context_score": 80.0, "context_grade": "B", "tags": [],
        }
        metrics_file.write_text(json.dumps(metric) + "\n")
        export_path = tmp_path / "report_export.json"

        result = runner.invoke(app, [
            "report",
            "--session", "cli",
            "--log-dir", str(tmp_path),
            "--export", str(export_path),
        ])
        assert export_path.exists()
        data = json.loads(export_path.read_text())
        assert "total_runs" in data