# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| `1.0.x` | Yes       |

## Multi-Tenant Isolation Guarantees
- Semantic cache partitions are strictly namespaced by `(tenant_id, blake2b(rag_context))` to prevent cross-tenant prompt or completion leakage.
- Priority load shedding rejects excess background traffic *before* deducting per-tenant RPS/TPM quotas so global bursts cannot starve isolated tenant token budgets.
- Hallucinated RAG completions (`hallucination_detected == True`) are never persisted to the semantic cache.

## Reporting a Vulnerability
Please report security issues privately to **Nagarjuna Kamireddy** (`writetonagarjuna@gmail.com`) rather than opening a public GitHub issue. Reports are acknowledged within 48 hours.
