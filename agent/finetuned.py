"""LoRA-finetuned Qwen2.5-Coder-1.5B agent.

Same Agent protocol as baseline / multi — the eval harness treats it identically.

Loads Qwen2.5-Coder-1.5B base + LoRA adapter from HF Hub the first time it
answers a question (~5 min cold start for the ~3GB base download; cached after).

Training code + lessons in the companion repo:
  https://github.com/visethchapman/text2sql-finetune
Adapter on HF Hub:
  https://huggingface.co/visethchapman/ercot-text2sql-qwen-1.5b-lora
"""
from __future__ import annotations

import time

import psycopg

from agent.base import AgentResult

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_REPO = "visethchapman/ercot-text2sql-qwen-1.5b-lora"


SYSTEM_PROMPT = """You are a Postgres SQL expert for ERCOT electricity-demand and Houston weather data.

Schema:
eia.demand(region, period, value, value_units) — region='ERCO'; period in UTC; value in MWh
noaa.daily_weather(station_id, obs_date, tmax_c, tmin_c, prcp_mm, awnd_ms) — Houston station; obs_date is local date
noaa.stations(station_id, name, state, nearest_eia_region)

Notes: All demand is UTC. Houston weather is local date. For joins,
cast period to local date: (period AT TIME ZONE 'America/Chicago')::date

Return ONLY valid Postgres SQL. No explanation, no markdown fences."""


# Loaded lazily on first call — torch + transformers are heavy imports.
_MODEL = None
_TOKENIZER = None


def _load(adapter_repo: str = ADAPTER_REPO, base_model: str = BASE_MODEL):
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _TOKENIZER

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    _TOKENIZER = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype).to(device)
    _MODEL = PeftModel.from_pretrained(base, adapter_repo).to(device)
    _MODEL.eval()
    return _MODEL, _TOKENIZER


class FinetunedAgent:
    name = "finetuned"

    def __init__(self, adapter_repo: str = ADAPTER_REPO):
        self.adapter_repo = adapter_repo

    def answer(self, question: str, conn: psycopg.Connection) -> AgentResult:
        import torch
        model, tok = _load(self.adapter_repo)
        t0 = time.time()

        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        inputs = tok.apply_chat_template(
            msgs, return_tensors="pt", add_generation_prompt=True, return_dict=True,
        ).to(model.device)
        in_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        sql = tok.decode(out[0][in_tokens:], skip_special_tokens=True).strip()
        out_tokens = out.shape[1] - in_tokens

        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
                cols = [d.name for d in cur.description] if cur.description else []
            conn.commit()
            return AgentResult(
                sql=sql, result_rows=rows, result_columns=cols,
                input_tokens=in_tokens, output_tokens=out_tokens,
                latency_sec=time.time() - t0,
            )
        except Exception as e:
            conn.rollback()
            return AgentResult(
                sql=sql, error=str(e)[:200], category="sql_error",
                input_tokens=in_tokens, output_tokens=out_tokens,
                latency_sec=time.time() - t0,
            )
