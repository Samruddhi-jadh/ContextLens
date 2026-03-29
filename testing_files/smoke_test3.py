from contextlens.core.logger import ContextEntry, RetrievedDoc
from contextlens.core.evaluator import ContextEvaluator

evaluator = ContextEvaluator()

# --- Test 1: Good context ---
doc = RetrievedDoc(
    content='Context engineering is the discipline of designing optimal '
            'prompts and retrieval pipelines to maximize LLM accuracy and '
            'minimize hallucination in production AI systems.',
    source='internal_wiki',
    relevance_score=0.94,
)
good = ContextEntry(
    user_prompt='Explain how context engineering reduces hallucinations in production LLMs, with 2 examples.',
    system_prompt='You are a senior AI engineer. Respond with precise technical detail. Cite sources. Never speculate.',
    retrieved_docs=[doc],
    total_tokens=350, system_tokens=20, user_tokens=18, docs_tokens=80,
    max_context_tokens=8000, provider='groq', model='llama-3.1-70b-versatile',
)
r1 = evaluator.evaluate(good)
print(f'Good context  → score={r1.overall_score}/100  grade={r1.grade}  issues={len(r1.all_issues)}')

# --- Test 2: Bad context ---
dup_content = 'AI is important for various things and does many useful stuff generally.'
bad = ContextEntry(
    user_prompt='tell me something about AI',
    system_prompt='',
    retrieved_docs=[
        RetrievedDoc(content=dup_content, source='a'),
        RetrievedDoc(content=dup_content, source='b'),
    ],
    total_tokens=7900, system_tokens=0, user_tokens=5, docs_tokens=7000,
    max_context_tokens=8000, provider='groq', model='llama-3.1-70b-versatile',
)
r2 = evaluator.evaluate(bad)
print(f'Bad context   → score={r2.overall_score}/100  grade={r2.grade}  issues={len(r2.all_issues)}')
print()
print('Issues found:')
for issue in r2.all_issues:
    print(f'  • {issue[:90]}')
print()
print('Suggestions:')
for sug in r2.all_suggestions:
    print(f'  → {sug[:90]}')