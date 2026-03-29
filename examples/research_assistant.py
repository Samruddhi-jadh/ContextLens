# examples/research_assistant.py
"""
Demo: AI Research Assistant using ContextLens.

This shows ContextLens working end-to-end:
  - Logging context with retrieved docs
  - Running inference on Groq
  - Comparing Groq vs Gemini on the same prompt
  - Reading back session stats

Run:
    python examples/research_assistant.py
"""

from contextlens import ContextLens
from contextlens.core.logger import RetrievedDoc
# Build a deliberately flawed context to demonstrate optimization
from contextlens.core.logger import RetrievedDoc
from contextlens.core.evaluator import ContextEvaluator
from contextlens.core.optimizer import ContextOptimizer, OptimizationConfig


def main():
    lens = ContextLens(
        session_id="research_assistant_demo",
        default_provider="groq",
    )

    # --- Single run with RAG context ---
    print("\n=== Single Run with RAG Context ===\n")

    retrieved_docs = [
        RetrievedDoc(
            content=(
                "Context engineering is the discipline of designing and optimizing "
                "the information passed to large language models. It encompasses "
                "prompt design, retrieval augmentation, and context compression."
            ),
            source="internal_wiki",
            relevance_score=0.95,
        ),
        RetrievedDoc(
            content=(
                "Poor context quality leads to hallucination, irrelevant responses, "
                "and excessive token usage, increasing both cost and latency."
            ),
            source="research_paper_2024",
            relevance_score=0.82,
        ),
    ]

    result = lens.run(
        prompt="What is context engineering and why does it matter for production AI systems?",
        system_prompt=(
            "You are a senior AI engineer. Answer concisely and technically. "
            "Use the retrieved documents to ground your response."
        ),
        retrieved_docs=retrieved_docs,
        tags=["demo", "rag"],
        metadata={"demo_version": "1.0"},
    )

    result.print_summary()

    # --- Multi-provider comparison ---
    print("\n=== Multi-Provider Comparison ===\n")

    lens.compare(
        prompt="In one paragraph, explain why token efficiency matters in production LLM systems.",
        system_prompt="You are a concise technical writer.",
        max_tokens=200,
    )

    # --- Session stats ---
    print("\n=== Session Statistics ===\n")
    stats = lens._context_logger.get_token_stats()
    for key, value in stats.items():
        print(f"  {key:28s}: {value}")
        # Add to bottom of main() in examples/research_assistant.py

    print("\n=== Token Monitor Report ===\n")
    lens.report()

    # Optionally export to JSON
    report_path = lens.export_report("./logs/demo_report.json")
    print(f"\nReport exported to: {report_path}")
        # Add to examples/research_assistant.py — after the compare() block

    print("\n=== Context Optimization Demo ===\n")


    # Near-duplicate docs + long content = perfect optimization target
    dup_content = (
        "Large language models require careful context management to produce "
        "accurate and relevant responses in production AI systems. Context "
        "engineering is the practice of designing, curating, and optimizing "
        "the information passed to these models to maximize output quality "
        "while minimizing token costs and latency."
    )
    flawed_docs = [
        RetrievedDoc(content=dup_content, source="wiki_a", relevance_score=0.90),
        RetrievedDoc(content=dup_content + " This is critical for production.", source="wiki_b", relevance_score=0.85),
        RetrievedDoc(content="The weather in Mumbai averages 27°C annually. Monsoon season runs June–September.", source="geo_db", relevance_score=0.10),
        RetrievedDoc(content=dup_content + " Optimization reduces costs significantly.", source="blog_c", relevance_score=0.80),
    ]

    # Log the flawed context entry
    from contextlens.core.logger import ContextLogger, ContextEntry
    cl = ContextLogger(session_id="optimization_demo")
    flawed_entry = cl.log(
        user_prompt="How does context engineering improve LLM accuracy in production systems?",
        system_prompt="You are a senior AI engineer. Be precise and cite sources.",
        retrieved_docs=flawed_docs,
        provider="groq",
        model="llama-3.1-70b-versatile",
    )

    # Evaluate it
    evaluator = ContextEvaluator()
    report = evaluator.evaluate(flawed_entry)
    print(f"Before optimization: score={report.overall_score:.1f}/100  grade={report.grade}")
    print(f"Issues found: {len(report.all_issues)}")
    for issue in report.all_issues:
        print(f"  • {issue[:85]}")

    # Optimize it
    optimizer = ContextOptimizer(OptimizationConfig(
        similarity_threshold=0.55,
        max_docs=3,
        max_doc_tokens=300,
    ))
    opt_result, new_report = optimizer.optimize_and_report(flawed_entry, report)

    print(f"\nAfter optimization:  score={new_report.overall_score:.1f}/100  grade={new_report.grade}")
    print(f"\n{opt_result.summary()}")


if __name__ == "__main__":
    main()