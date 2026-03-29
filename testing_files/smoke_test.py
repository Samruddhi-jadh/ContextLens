from contextlens.core.logger import ContextLogger, RetrievedDoc

cl = ContextLogger(session_id='smoke_test')

entry = cl.log(
    user_prompt='What is context engineering?',
    system_prompt='You are a senior AI engineer.',
    retrieved_docs=[
        RetrievedDoc(
            content='Context engineering is the art of constructing optimal input to LLMs.',
            source='internal_docs',
            relevance_score=0.92,
        ),
        RetrievedDoc(
            content='Poor context leads to hallucinations and high token costs.',
            source='research_paper',
            relevance_score=0.78,
        ),
    ],
    provider='groq',
    model='llama-3.1-70b-versatile',
    tags=['test', 'phase2'],
)

print(f'run_id       : {entry.run_id}')
print(f'total_tokens : {entry.total_tokens}')
print(f'utilization  : {entry.token_utilization:.1%}')
print(f'docs         : {entry.docs_summary}')
print('Log saved to:', 'logs/')