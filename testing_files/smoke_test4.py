from contextlens.core.monitor import TokenMonitor
from unittest.mock import MagicMock

monitor = TokenMonitor(session_id='smoke', budget_usd=0.01)

# Simulate 5 runs
providers = ['groq', 'groq', 'gemini', 'groq', 'gemini']
latencies = [620, 480, 1240, 710, 980]
costs     = [0.000177, 0.000106, 0.000019, 0.000210, 0.000022]

for i, (prov, lat, cost) in enumerate(zip(providers, latencies, costs)):
    r = MagicMock()
    r.response.run_id = f'run-{i:03d}'
    r.response.provider = prov
    r.response.model = 'llama-3.1-70b-versatile'
    r.response.input_tokens = 200
    r.response.output_tokens = 80
    r.response.total_tokens = 280
    r.response.estimated_cost_usd = cost
    r.response.latency_ms = lat
    r.context_entry.run_id = f'run-{i:03d}'
    r.context_entry.total_tokens = 200
    r.context_entry.tags = ['smoke']
    r.total_wall_time_ms = lat + 20
    ev = MagicMock()
    ev.overall_score = 70 + i * 4.0
    ev.grade = 'B'
    r.evaluation = ev
    monitor.record(r)

monitor.print_report()