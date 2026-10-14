# LLM Token & Cost Log (Full Run)

| Metric | Value |
| :--- | :--- |
| **Model** | Local Grounded Engine (Free / Offline) (`local`) |
| **Total LLM Calls** | 21 |
| **Total Input Tokens** | 282 |
| **Total Output Tokens** | 898 |
| **Total Tokens** | 1,180 |
| **Input Rate** | $0.0000 per 1M tokens |
| **Output Rate** | $0.0000 per 1M tokens |
| **Total Estimated Cost** | **$0.000000 USD** |

### Efficiency Notes
- Retrieval and filtering reduce candidate volume from 728 route-weeks down to only true anomaly candidates.
- Context window per prompt is tightly bounded (system prompt + 1 candidate note + route statistics).
- Temperature is locked at `0.0` for 100% reproducible results.
