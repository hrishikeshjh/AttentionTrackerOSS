# Contributing to AttentionTrackerOSS

Thank you for your interest in contributing to **AttentionTrackerOSS**! This open-source repository provides real-time, privacy-first cognitive engagement analysis, multi-person CCTV tracking, and cyber-physical hardware actuation.

We welcome contributions from developers, researchers, students, and practitioners worldwide.

---

## Code of Conduct

By participating in this project, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## How Can I Contribute?

- **Reporting Bugs**: Check existing issues first. If not found, open a [Bug Report](https://github.com/hrishikeshjh/AttentionTrackerOSS/issues/new?template=bug_report.md) with reproduction steps, environment details, and relevant logs.
- **Suggesting Enhancements**: Open a [Feature Request](https://github.com/hrishikeshjh/AttentionTrackerOSS/issues/new?template=feature_request.md) describing the problem, proposed solution, and architectural impact.
- **Improving Documentation**: Fix typos, clarify mathematical explanations, add diagrams, or expand setup guides.
- **Submitting Code**:
  - Implement edge-friendly ML optimizations or new sensor models.
  - Enhance v1.0 Next.js frontend, Chrome extension, or browser inference pipeline.
  - Optimize v2.0 FastAPI, YOLOv8/ByteTrack tracking, or WebSocket streams.
  - Add cyber-physical actuators, protocols, or firmware extensions (ESP32/Raspberry Pi).

---

## Development Workflow

### 1. Fork & Clone

```bash
git clone https://github.com/<your-username>/AttentionTrackerOSS.git
cd AttentionTrackerOSS
git checkout -b feature/my-new-feature
```

### 2. Sub-system Setup

#### Sub-system 1: v1.0 Browser App & Chrome Extension
```bash
cd v1.0
npm install
bash scripts/setup-models.sh
npm run dev # Launches web app at http://localhost:9002
```

#### Sub-system 2: v2.0 Server & Multi-person CCTV Engine
```bash
cd v2.0
python3 -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt
python run_analyzer.py --source 0 --display
```

#### Sub-system 3: ESP32 Cyber-Physical Relay App
```bash
cd esp32_relay_app
bash run.sh
```

---

## Testing & Quality Gates

Before submitting a Pull Request, ensure all tests pass:

```bash
# Test v1.0 (Vitest)
npm --prefix v1.0 test -- --run

# Lint & Typecheck
npm --prefix v1.0 run lint
npm --prefix v1.0 run typecheck

# Test v2.0
pytest v2.0/tests/
ruff check v2.0/
mypy v2.0/
```

Every PR runs GitHub Actions CI checks (`.github/workflows/ci.yml`), including:
- Unit and integration tests (320+ Vitest tests, pytest suite)
- Security audit (`npm audit`, `pip-audit`)
- Static typing (`tsc --noEmit`, `mypy`)
- Secret scanner (`gitleaks`)

---

## Pull Request Guidelines

1. **One feature/fix per PR**: Keep changes atomic and reviewable.
2. **Include Tests**: Add or update test coverage for any logic changes.
3. **Update Documentation**: If your PR touches APIs, math models, or CLI parameters, update the corresponding `README.md` and `ARCHITECTURE.md`.
4. **Follow Conventional Commits**: e.g., `feat:`, `fix:`, `docs:`, `perf:`, `test:`, `refactor:`.

---

## Maintainer & Contact

- **Lead Developer & Maintainer**: Hrishikesh Jha ([@hrishikeshjh](https://github.com/hrishikeshjh) · `hrishikeshjhapersonal@gmail.com`)
