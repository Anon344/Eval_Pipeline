Sycophancy‑Resistance Evaluation Pipeline
========================================

This repo builds a synthetic multi‑turn dataset to probe sycophancy and then
benchmarks a candidate LLM with an LLM‑as‑Judge committee.

Quick start
-----------
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your keys
python src/builder/run_builder.py --n 10              # generate 10 convo seeds
python src/runner/execute.py --model openai/gpt-4o    # run target model
python src/judge/evaluate.py                          # score with committee
python src/metrics/report.py                          # view metrics
