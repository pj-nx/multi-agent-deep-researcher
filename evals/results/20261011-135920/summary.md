# Evaluation summary

Settings: max_hops=1, budget_seconds=240, heavy=deepseek/deepseek-v4-pro, worker=anthropic/claude-sonnet-5, cheap=anthropic/claude-haiku-4.5, grader=openai/gpt-5-mini, reasoning_effort=high, embeddings=local

| Case | Sub-Qs | Kept | Claims | Contra. | Time | Tokens | Cost | Citations | Grounded | Coverage | Contra. recall | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| four-day-week | 5 | 67 | 46 | 5 | 873s | 259,906 | $0.292 | 100% | 90% | 100% | 50% | 85% |
| ai-coding-assistants | 5 | 71 | 58 | 18 | 970s | 310,008 | $0.343 | 100% | 95% | 100% | 100% | 99% |
| glp1-cardiovascular | 5 | 65 | 41 | 3 | 557s | 182,636 | $0.263 | 100% | 90% | 75% | 100% | 91% |
| ultra-processed-food | 4 | 40 | 37 | 3 | 556s | 144,544 | $0.211 | 100% | 92% | 75% | 100% | 92% |
| intermittent-fasting | 5 | 10 | 7 | 1 | 672s | 95,015 | $0.139 | 100% | 100% | 100% | n/a | 100% |
| remote-work-innovation | 5 | 41 | 39 | 5 | 698s | 191,593 | $0.254 | 100% | 90% | 100% | 100% | 98% |
| congestion-charges | 4 | 56 | 32 | 10 | 862s | 212,447 | $0.275 | 100% | 85% | 100% | 100% | 96% |
| grid-batteries | 6 | 55 | 57 | 4 | 674s | 214,036 | $0.320 | 100% | 98% | 100% | 100% | 100% |
| smr-economics | 5 | 66 | 50 | 9 | 811s | 265,075 | $0.341 | 100% | 93% | 100% | 100% | 98% |
| microplastics-health | 5 | 75 | 52 | 4 | 961s | 279,303 | $0.313 | 100% | 93% | 75% | 100% | 92% |
| **mean (10/10 ok)** |  |  |  |  | 763s | 215,456 | $0.275 | 100% | 93% | 92% | 94% | 95% |

Scores: citations = cited ids that exist in the bibliography; grounded = claims whose cited text supports them (model-judged, partial = 0.5); coverage = expected facets addressed; contra. recall = seeded disagreement surfaced (1.0 matches the hint, 0.5 other contradiction only, n/a when none was seeded). Cost is estimated from OpenRouter per-token prices for the models used; embeddings are not included.

Model call time by role (mean per case; summed call durations, so parallel calls overlap and the total can exceed wall time): analysis 591.53s, critic 201.21s, insight 168.56s, planner 44.85s, report 45.73s, validator 218.15s
