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

## Overview

This repository is a turn‑key framework for creating and judging alignment‑stress datasets, with a focus on sycophancy, helpfulness, harmfulness, and objectiveness.
It contains three independent stages:

Stage	Folder / script	What it does
1 Dataset Builder	src/builder/run_builder.py + persona_agent.py	Generates multi‑turn conversations that tempt a model to flatter or bias itself.
2 Model Runner	src/eval/run_eval.py	Replays each conversation against any OpenRouter model and logs its final reply.
3 LLM‑Judge	src/eval/judge_pass.py	Uses an independent “judge model” to score the reply on S /H /Ha /O axes.

Data flow:

bash
Copy
Edit
configs/           ──►  builder  ──►  data/conversations.jsonl
                               (replay) ──►  data/model_responses.jsonl
                                         (judge)  ──►  data/judge_scores.jsonl
## Installation

bash
Copy
Edit
git clone <repo>
cd <repo>

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
## Folder structure

bash
Copy
Edit
configs/
    personas.yaml            # user personas (id, description, traits, stance, tags[])
    harmful_triggers.yaml    # prompts that tempt sycophancy (id, user_prompt, tags[])
data/
    conversations.jsonl      # built by builder
    model_responses.jsonl    # built by run_eval
    judge_scores.jsonl       # built by judge_pass
src/
    builder/
        run_builder.py       # CLI: generate conversations
        persona_agent.py     # LLM agent (OpenRouter)
    eval/
        run_eval.py          # CLI: replay against target model
        judge_pass.py        # CLI: score with judge model
## 1 Dataset Builder

Generate 50 tagged conversations with google/gemini-pro as the persona agent:

bash
Copy
Edit
python -m src.builder.run_builder --n 50 --model google/gemini-pro
Uses configs/personas.yaml + configs/harmful_triggers.yaml.
Output: data/conversations.jsonl (one JSON line per conv).

### Persona ↔ Trigger matching

Each persona carries tags: [ ... ].

choose_trigger() selects a trigger whose tags intersect, else falls back.

To force strict matching remove fallback=True in choose_trigger().

## 2 Replay against target model

bash
Copy
Edit
python -m src.eval.run_eval --model openai/gpt-4o
History (all turns except the last) is sent as context; only the final user turn is used as the live prompt.
Output: data/model_responses.jsonl

jsonc
Copy
Edit
{"conv_id":"...","prompt":"...","assistant_reply":"...","meta":{"model":"openai/gpt-4o"}}
## 3 Judge with independent model

bash
Copy
Edit
python -m src.eval.judge_pass --judge_model openai/gpt-4o --delay 0.5
Each reply is scored on:

Key	Scale	Meaning (high = better)
S	 1–5	Sycophancy‑resistance
H	 1–5	Helpfulness
Ha	 1–5	Harmfulness (safety)
O	 1–5	Objectiveness (bias)

Output: data/judge_scores.jsonl

## Aggregating results

python
Copy
Edit
import pandas as pd, json, pathlib
scores = [json.loads(l) for l in pathlib.Path("data/judge_scores.jsonl").open()]
df = pd.json_normalize(scores)
print(df["scores.S"].mean(), df["scores.H"].mean())
Combine multiple judge files (or majority vote) as desired.

## Extending

Task	How
Add personas	Append to personas.yaml. Add tags: for better matching.
Add triggers	Append to harmful_triggers.yaml with tags:.
Custom judge rubric	Edit TEMPLATE in judge_pass.py. Expect JSON keys you define.
Multiple completions	In run_eval.py set n=3 and store all choices.
Rate‑limit handling	In judge_pass.py wrap the call in try/except for 429, add exponential back‑off.

## Citation / credits

Built on top of OpenRouter (OpenAI‑compatible) endpoints

Sycophancy rubric inspired by Anthropic’s Harmless and Helpful paper

Contributions and PRs welcome!