from contextlens.core.logger import ContextEntry, RetrievedDoc, TokenCounter
from contextlens.core.evaluator import ContextEvaluator
from contextlens.core.optimizer import ContextOptimizer, OptimizationConfig

counter = TokenCounter()
dup = (
    'Context engineering is the discipline of designing and optimizing '
    'the information passed to large language models. It involves prompt '
    'design, retrieval augmentation, and context window management to '
    'maximize accuracy and minimize hallucinations in production systems.'
)
docs = [
    RetrievedDoc(content=dup, source='wiki_a', relevance_score=0.92),
    RetrievedDoc(content=dup + ' This is critical.', source='wiki_b', relevance_score=0.88),
    RetrievedDoc(content='Annual rainfall in the Amazon exceeds 2000mm per year.', source='geo_db', relevance_score=0.05),
    RetrievedDoc(content=dup + ' Optimization reduces costs.', source='blog_c', relevance_score=0.81),
]
docs_tokens = sum(counter.count(d.content) for d in docs)
entry = ContextEntry(
    user_prompt='How does context engineering improve LLM accuracy in production?',
    system_prompt='You are a senior AI engineer. Be precise and cite sources.',
    retrieved_docs=docs,
    system_tokens=counter.count('You are a senior AI engineer. Be precise and cite sources.'),
    user_tokens=counter.count('How does context engineering improve LLM accuracy in production?'),
    docs_tokens=docs_tokens,
    total_tokens=docs_tokens + 30,
    max_context_tokens=8000,
    provider='groq', model='llama-3.1-70b-versatile',
)

evaluator = ContextEvaluator()
optimizer = ContextOptimizer(OptimizationConfig(
    similarity_threshold=0.55,
    max_docs=3,
    max_doc_tokens=200,
))

report = evaluator.evaluate(entry)
print(f'Before: score={report.overall_score:.1f}/100  grade={report.grade}  docs={len(entry.retrieved_docs)}  tokens={entry.total_tokens}')

opt_result, new_report = optimizer.optimize_and_report(entry, report)
print(f'After:  score={new_report.overall_score:.1f}/100  grade={new_report.grade}  docs={len(opt_result.optimized_entry.retrieved_docs)}  tokens={opt_result.optimized_entry.total_tokens}')
print()
print(opt_result.summary())