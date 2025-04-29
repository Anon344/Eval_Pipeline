"""
Run a sycophancy evaluation pass.

For each stored conversation:
  • send the full history (except final turn) as system context
  • send the final USER turn as the live prompt
  • record the assistant’s reply for later grading
"""

import argparse, json, os
from pathlib import Path
from typing import List

from openai import OpenAI               # OpenRouter-compatible
from tqdm import tqdm

ROOT       = Path(__file__).resolve().parents[2]
DATA_DIR   = ROOT / "data"
INPUT_FILE = DATA_DIR / "conversations.jsonl"
OUTPUT_FILE = DATA_DIR / "model_responses.jsonl"

# ─── helper ------------------------------------------------------------ #
def load_conversations(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# ─── main -------------------------------------------------------------- #
def main(model_slug: str):
    # 1) OpenRouter client
    api_key = ""
    if not api_key:
        raise RuntimeError("Set Google_API_Key in your env.")
    client = OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=api_key,
        )

    # 2) load dataset
    conversations = load_conversations(INPUT_FILE)
    print(f"Loaded {len(conversations)} conversations")

    # 3) iterate & query
    out_f = open(OUTPUT_FILE, "a", encoding="utf-8")
    for conv in tqdm(conversations, desc="Evaluating"):
        turns = conv["turns"]

        # split history vs final prompt
        history  = turns[:-1]               # everything except last turn
        last_turn = turns[-1]
        assert last_turn["role"] == "user", "Last turn must be USER"

        # compose messages: history first (as 'system' or original roles)
        messages = []
        for t in history:
            messages.append({"role": t["role"], "content": t["content"]})

        # live user prompt
        messages.append({"role": "user", "content": last_turn["content"]})

        # call model
        resp = client.chat.completions.create(
            model=model_slug,
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )
        assistant_reply = resp.choices[0].message.content

        # write result
        out_f.write(json.dumps({
            "conv_id": conv["conv_id"],
            "persona": conv["persona"],
            "prompt": last_turn["content"],
            "assistant_reply": assistant_reply,
            "meta": {"model": model_slug}
        }, ensure_ascii=False) + "\n")

    out_f.close()
    print(f"Saved {len(conversations)} replies → {OUTPUT_FILE.resolve()}")

# ─── CLI --------------------------------------------------------------- #
if __name__ == "__main__":
    import sys, argparse
    ap = argparse.ArgumentParser(description="Replay conversations & capture model responses")
    ap.add_argument("--model", default="openai/gpt-4o", help="Model slug on OpenRouter")
    args = ap.parse_args()
    main(args.model)
