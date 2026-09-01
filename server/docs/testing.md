# Testing & Profiling

> The three independent test suites (none validate SLAM correctness — they cover latency contracts and frontend behaviour) and the profiling helpers.

| Suite | Runner | Where | What it covers |
|-------|--------|-------|----------------|
| Python phase tests | pytest | `tests/test_phase*.py`, `tests/test_session_*.py`, `tests/test_gpu_session_broker.py` | Latency-regression and isolated-session allocation contracts, plus the durable broker session-token mint/verify/alg-dispatch (`tests/test_session_token.py`, with the conftest `jwt` fake extended for HS256). `tests/conftest.py` heavily monkeypatches `vggt_slam.*` (and `flask`/`jwt`/`socketio` for `load_app_module`) so the suite runs without GPU/model weights or network. |
| Webserver SPA | vitest + playwright | `server/webserver/tests/` | `unit/TourController.test.ts`; `e2e/onboarding.spec.ts` |
| Landing app | vitest + playwright | `landing/test/`, `landing/e2e/` | Hooks like `useOnboarding`; playwright user flows |

```bash
# Python latency suite
pytest tests/

# Webserver SPA
cd server/webserver && npm test && npm run test:e2e

# Landing app
cd landing && npm test && npm run test:e2e
```

(Scene report / Q&A tests are listed in [scene-report.md](scene-report.md).)

**Profiling helpers** (`scripts/`):
- `scripts/gpu_monitor.py` — samples `nvidia-smi` and emits `[gpu]` lines to stdout, designed to interleave with `[latency]` lines for unified log analysis. Auto-started inside the Modal container; can also be run standalone (`python scripts/gpu_monitor.py --interval 2`).
- `scripts/plot_latency.py` — turns `[latency]` log lines into plots; runs offline on saved logs in `latency_logs/`.
