from contextlens.providers.groq_provider import GroqProvider
p = GroqProvider()
ok = p.health_check()
print('Groq health check:', 'PASS' if ok else 'FAIL')