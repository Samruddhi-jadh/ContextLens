# contextlens/core/logger.py
"""
Context Logger — Phase 2

Responsibility: Capture, validate, and persist every piece of context
passed to an AI agent in a structured, queryable format.

Design decisions:
  - Pydantic models for schema validation (fails fast on bad input)
  - tiktoken for accurate token counting before the API call
  - JSONL format (one JSON object per line) for streaming-friendly logs
  - Session-based log files (one file per day) for easy querying
  - loguru for structured logging with automatic rotation
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tiktoken
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from contextlens.config import config


# ---------------------------------------------------------------------------
# Data Models (the "contract" for what a context looks like)
# ---------------------------------------------------------------------------

class RetrievedDoc(BaseModel):
    """
    One document retrieved from a vector store, RAG pipeline, or manual input.
    
    Senior note: We model docs explicitly so the evaluator can score
    each one individually — source quality, relevance, redundancy.
    """
    content: str
    source: str = "unknown"           # e.g. "wikipedia", "internal_db", "manual"
    relevance_score: float | None = None  # from retriever (0.0 - 1.0)
    token_count: int = 0              # populated by logger automatically

    @field_validator("relevance_score")
    @classmethod
    def clamp_score(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("relevance_score must be between 0.0 and 1.0")
        return v


class ContextEntry(BaseModel):
    """
    A complete, timestamped snapshot of the context for one agent run.
    
    This is the core unit of the entire ContextLens toolkit.
    Every evaluator, optimizer, and monitor reads from this structure.
    
    Senior note: We separate system/user/docs deliberately.
    Each has different optimization strategies — you don't trim a
    system prompt the same way you trim retrieved docs.
    """
    # Identity
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = "default"
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Context components (the inputs to the LLM)
    system_prompt: str = ""
    user_prompt: str
    retrieved_docs: list[RetrievedDoc] = Field(default_factory=list)
    conversation_history: list[dict[str, str]] = Field(default_factory=list)

    # Token counts (populated automatically by logger)
    system_tokens: int = 0
    user_tokens: int = 0
    docs_tokens: int = 0
    history_tokens: int = 0
    total_tokens: int = 0

    # Config at time of run
    provider: str = "unknown"
    model: str = "unknown"
    max_context_tokens: int = Field(default_factory=lambda: config.max_context_tokens)

    # Optional tags for filtering logs later
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def token_utilization(self) -> float:
        """What % of the context window is used? >0.9 is a warning sign."""
        if self.max_context_tokens == 0:
            return 0.0
        return self.total_tokens / self.max_context_tokens

    @property
    def docs_summary(self) -> dict[str, Any]:
        """Quick summary of retrieved docs for reporting."""
        return {
            "count": len(self.retrieved_docs),
            "total_tokens": self.docs_tokens,
            "sources": [d.source for d in self.retrieved_docs],
            "avg_relevance": (
                sum(d.relevance_score for d in self.retrieved_docs
                    if d.relevance_score is not None)
                / max(1, sum(1 for d in self.retrieved_docs
                             if d.relevance_score is not None))
            ) if self.retrieved_docs else 0.0,
        }

    def to_log_dict(self) -> dict[str, Any]:
        """Serialize for JSON log storage. Keeps docs as dicts."""
        data = self.model_dump()
        data["token_utilization"] = round(self.token_utilization, 4)
        data["docs_summary"] = self.docs_summary
        return data


# ---------------------------------------------------------------------------
# Token Counter
# ---------------------------------------------------------------------------

class TokenCounter:
    """
    Accurate token counting using tiktoken (OpenAI's tokenizer).
    
    Why tiktoken and not len(text.split())?
    A word like "unbelievable" = 1 word but may = 3 tokens.
    Inaccurate token counting = unexpected API errors mid-run.
    
    We use cl100k_base (GPT-4 / Claude compatible) as a close approximation
    for Groq and Gemini models. For production, use model-specific tokenizers.
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        try:
            self._enc = tiktoken.get_encoding(encoding_name)
        except Exception:
            # Fallback: rough estimate if tiktoken unavailable
            self._enc = None
            logger.warning(
                "tiktoken encoding '{}' not found. Using word-based estimate.",
                encoding_name
            )

    def count(self, text: str) -> int:
        """Count tokens in a string."""
        if not text:
            return 0
        if self._enc:
            return len(self._enc.encode(text))
        # Fallback: ~0.75 words per token (rough average)
        return max(1, int(len(text.split()) / 0.75))

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        """Count tokens in a conversation history list."""
        # Each message has ~4 overhead tokens (role, separators)
        overhead_per_message = 4
        return sum(
            self.count(m.get("content", "")) + overhead_per_message
            for m in messages
        )


# ---------------------------------------------------------------------------
# Main Logger
# ---------------------------------------------------------------------------

class ContextLogger:
    """
    Captures, counts, and persists context for every AI agent run.

    Usage:
        cl = ContextLogger()
        entry = cl.log(
            user_prompt="Explain transformers",
            system_prompt="You are an AI expert.",
            retrieved_docs=[RetrievedDoc(content="...", source="wiki")],
            provider="groq",
            model="llama-3.1-70b-versatile",
        )
        print(entry.total_tokens)   # accurate token count
        print(entry.run_id)         # unique ID for this run
    """

    def __init__(
        self,
        log_dir: Path | None = None,
        session_id: str = "default",
    ):
        self.log_dir = log_dir or config.log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self._token_counter = TokenCounter()
        self._session_log_path = self._build_log_path()

        # Configure loguru: human-readable to console, JSON to file
        logger.remove()  # remove default handler
        logger.add(
            sink=lambda msg: print(msg, end=""),
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
            level="INFO",
            colorize=True,
        )
        logger.add(
            sink=str(self.log_dir / "contextlens.log"),
            rotation="10 MB",
            retention="30 days",
            level="DEBUG",
            serialize=False,
        )

    def _build_log_path(self) -> Path:
        """
        One log file per day, named by session + date.
        e.g. logs/default_20241201.jsonl
        
        JSONL (JSON Lines) = one JSON object per line.
        This lets you stream/grep logs without loading the whole file.
        """
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        filename = f"{self.session_id}_{today}.jsonl"
        return self.log_dir / filename

    def log(
        self,
        user_prompt: str,
        system_prompt: str = "",
        retrieved_docs: list[RetrievedDoc] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        provider: str = "unknown",
        model: str = "unknown",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ContextEntry:
        """
        Log a complete context snapshot.
        
        This is the primary method called before every AI API call.
        Returns a ContextEntry that flows into the evaluator and optimizer.
        
        Args:
            user_prompt:   The user's question or instruction.
            system_prompt: The system message / persona.
            retrieved_docs: Documents from RAG / retrieval.
            conversation_history: Prior turns [{"role": "user", "content": "..."}]
            provider:      "groq" | "gemini" | "nvidia"
            model:         Model name string.
            tags:          Optional labels for filtering (e.g. ["rag", "debug"])
            metadata:      Any extra key-value pairs.

        Returns:
            ContextEntry: Fully validated, token-counted entry.
        """
        docs = retrieved_docs or []
        history = conversation_history or []

        # --- Token counting (the expensive but essential step) ---
        system_tokens = self._token_counter.count(system_prompt)
        user_tokens = self._token_counter.count(user_prompt)
        history_tokens = self._token_counter.count_messages(history)

        # Count and annotate each doc with its token count
        docs_tokens = 0
        enriched_docs = []
        for doc in docs:
            doc_tokens = self._token_counter.count(doc.content)
            enriched_doc = doc.model_copy(update={"token_count": doc_tokens})
            enriched_docs.append(enriched_doc)
            docs_tokens += doc_tokens

        total_tokens = system_tokens + user_tokens + docs_tokens + history_tokens

        # --- Build the entry ---
        entry = ContextEntry(
            session_id=self.session_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            retrieved_docs=enriched_docs,
            conversation_history=history,
            system_tokens=system_tokens,
            user_tokens=user_tokens,
            docs_tokens=docs_tokens,
            history_tokens=history_tokens,
            total_tokens=total_tokens,
            provider=provider,
            model=model,
            tags=tags or [],
            metadata=metadata or {},
        )

        # --- Warn if approaching context limit ---
        if entry.token_utilization > 0.85:
            logger.warning(
                "High context utilization: {:.1%} ({}/{} tokens). "
                "Consider running the optimizer.",
                entry.token_utilization,
                entry.total_tokens,
                entry.max_context_tokens,
            )
        elif entry.token_utilization > 1.0:
            logger.error(
                "Context OVERFLOW: {} tokens exceeds limit of {}. "
                "This run WILL fail at the API.",
                entry.total_tokens,
                entry.max_context_tokens,
            )

        # --- Persist to JSONL ---
        self._write_entry(entry)

        logger.info(
            "Logged run_id={} | provider={} | tokens={}/{} ({:.1%})",
            entry.run_id[:8],
            entry.provider,
            entry.total_tokens,
            entry.max_context_tokens,
            entry.token_utilization,
        )

        return entry

    def _write_entry(self, entry: ContextEntry) -> None:
        """
        Append entry to JSONL log file.
        
        Senior note: We append (not overwrite) so logs survive crashes.
        JSONL means each line is independently parseable.
        """
        try:
            with open(self._session_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_log_dict(), ensure_ascii=False) + "\n")
        except OSError as e:
            # Logging failure must NOT crash the main application
            logger.error("Failed to write context log: {}", e)

    # ------------------------------------------------------------------
    # Query / Read methods
    # ------------------------------------------------------------------

    def load_session(self) -> list[dict[str, Any]]:
        """
        Load all entries from today's session log.
        Returns raw dicts — parse with ContextEntry(**d) if needed.
        """
        if not self._session_log_path.exists():
            return []
        entries = []
        with open(self._session_log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning("Skipping malformed log line: {}", e)
        return entries

    def load_all_sessions(self, days: int = 7) -> list[dict[str, Any]]:
        """
        Load entries from multiple days of log files.
        Useful for trend analysis and the Streamlit dashboard (Phase 10).
        """
        all_entries = []
        for log_file in sorted(self.log_dir.glob(f"{self.session_id}_*.jsonl")):
            with open(log_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            all_entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        return all_entries[-days * 1000:]  # rough cap

    def get_token_stats(self) -> dict[str, Any]:
        """
        Aggregate token stats across the current session.
        Used by the CLI 'report' command and the monitor (Phase 5).
        """
        entries = self.load_session()
        if not entries:
            return {"runs": 0}

        total_tokens_list = [e.get("total_tokens", 0) for e in entries]
        return {
            "runs": len(entries),
            "total_tokens_used": sum(total_tokens_list),
            "avg_tokens_per_run": round(sum(total_tokens_list) / len(entries), 1),
            "max_tokens_in_run": max(total_tokens_list),
            "min_tokens_in_run": min(total_tokens_list),
            "providers_used": list({e.get("provider") for e in entries}),
        }