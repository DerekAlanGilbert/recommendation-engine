# Live Recommendation Engine

A small proof of concept for recommendations that learn from live thumbs-up and thumbs-down feedback.

The first slice is a deterministic dataset of 1,000 synthetic cars, plus reproducible feature encoding and preference personas.

## Run the tests

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest -q
```
