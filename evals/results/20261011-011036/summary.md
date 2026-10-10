# Evaluation summary

Settings: max_hops=1, budget_seconds=240, heavy=deepseek/deepseek-v4-pro, worker=anthropic/claude-sonnet-5, cheap=anthropic/claude-haiku-4.5, grader=anthropic/claude-haiku-4.5, reasoning_effort=high, embeddings=local

| Case | Sub-Qs | Kept | Claims | Contra. | Time | Tokens | Cost | Citations | Grounded | Coverage | Contra. recall | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| four-day-week | 6 | 11 | 9 | 1 | 335s | 50,040 | $0.104 | 100% | 89% | 80% | 100% | 92% |
| ai-coding-assistants | 5 | 19 | 28 | 6 | 641s | 122,202 | $0.153 | 100% | 100% | 100% | 100% | 100% |
| glp1-cardiovascular | 6 | 4 | 3 | 0 | 294s | 48,682 | $0.106 | 100% | 33% | 50% | 0% | 46% |
| ultra-processed-food | 6 | 11 | 16 | 1 | 718s | 104,168 | $0.131 | 100% | 100% | 75% | 50% | 81% |
| intermittent-fasting | 6 | 16 | 10 | 1 | 634s | 102,575 | $0.155 | 100% | 100% | 75% | n/a | 92% |
| remote-work-innovation | 5 | 9 | 9 | 0 | 327s | 56,803 | $0.114 | 100% | 94% | 50% | 0% | 61% |
| congestion-charges | 7 | 17 | 25 | 1 | 769s | 120,668 | $0.142 | 100% | 100% | 50% | 100% | 88% |
| grid-batteries | 6 | 25 | 28 | 0 | 500s | 108,209 | $0.175 | 100% | 100% | 100% | 0% | 75% |
| smr-economics | 5 | 10 | 23 | 1 | 495s | 86,971 | $0.145 | 100% | 100% | 50% | 100% | 88% |
| microplastics-health | 6 | 7 | 13 | 0 | 398s | 58,252 | $0.100 | 100% | 100% | 100% | 0% | 75% |
| **mean (10/10 ok)** |  |  |  |  | 511s | 85,857 | $0.133 | 100% | 92% | 73% | 50% | 80% |

Scores: citations = cited ids that exist in the bibliography; grounded = claims whose cited text supports them (model-judged, partial = 0.5); coverage = expected facets addressed; contra. recall = seeded disagreement surfaced (1.0 matches the hint, 0.5 other contradiction only, n/a when none was seeded). Cost is estimated from OpenRouter per-token prices for the models used; embeddings are not included.
