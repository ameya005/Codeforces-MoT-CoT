#!/usr/bin/env python
import os, re, json, argparse, sys, glob
from typing import Optional, Tuple, List, Dict, Set

import torch
from datasets import load_dataset, Dataset, Features, Sequence, Value
from transformers import AutoTokenizer, AutoModelForCausalLM


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

RANK        = int(os.environ.get("RANK", "0"))
WORLD_SIZE  = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL_RANK  = int(os.environ.get("LOCAL_RANK", str(RANK)))


THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

def extract_think(text: str) -> Optional[str]:
    m = list(re.finditer(re.escape(THINK_OPEN) + r"(.*?)" + re.escape(THINK_CLOSE),
                         text, flags=re.S|re.I))
    return m[-1].group(1) if m else None

def extract_last_code_block(text: str) -> Optional[Tuple[str,str]]:
    m = list(re.finditer(r"```([a-zA-Z0-9_+\-]*)\s*\n(.*?)```", text, flags=re.S))
    if not m: return None
    lang = (m[-1].group(1) or "").strip().lower()
    code = m[-1].group(2)
    return lang, code

def guess_lang(lang: str, code: str) -> str:
    if lang in {"python","py"}: return "python"
    if lang in {"cpp","c++","cc"}: return "cpp"
    if not lang:
        if re.search(r"\bdef\s+\w+\s*\(", code) or "sys.stdin" in code or "import " in code:
            return "python"
    return lang or "python"

def ensure_system_prefix(msgs: List[Dict], default_sys: str) -> List[Dict]:
    if not msgs or msgs[0].get("role") != "system":
        return [{"role":"system","content":default_sys}] + msgs
    return msgs

def sanitize_explanation(text: str, max_chars: int = 320) -> str:
    # Remove any leftover think tags & code fences
    text = re.sub(re.escape(THINK_OPEN)+r".*?"+re.escape(THINK_CLOSE), " ", text, flags=re.S|re.I)
    text = text.replace("```", " ").strip()
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "..."
    return text

def fsafe_append_jsonl(path: str, rows: List[Dict]):
    if not rows: return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def load_done_indices(jsonl_path: str) -> Set[int]:
    done: Set[int] = set()
    if not os.path.exists(jsonl_path): return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "source_idx" in obj:
                    done.add(int(obj["source_idx"]))
            except Exception:
                continue
    return done

def shard_jsonl(path: str, rank: int) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}.rank{rank}{ext}"


SUMMARY_SYS = (
    "You are a senior competitive programming mentor. "
    "Given a chain-of-thought for a correct solution, with several reasoning tracks, write a short explanation focusing on the main solution"
    "that captures only the final algorithmic logic and key edge cases. "
    "Only consider the final, correct reasoning trace. Do not include any explanation for the other traces."
    "Do NOT include code, numbered steps, or <think> tags. Plain English only."
)

@torch.inference_mode()
def summarize_batch(
    thinks: List[str],
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_new_tokens: int = 128,
    max_input_tokens: int = 2048
) -> List[str]:
    prompts, idx_map = [], []
    for j, t in enumerate(thinks):
        if t and t.strip():
            messages = [
                {"role":"system","content":SUMMARY_SYS},
                {"role":"user","content":f"Chain-of-thought:\n{t}\n\nWrite the short explanation now."},
            ]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
            idx_map.append(j)

    results = [""] * len(thinks)
    if not prompts:
        return results

    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_input_tokens)
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False, 
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
        use_cache=True,
    )

    input_lens = (enc["input_ids"] != tok.pad_token_id).sum(dim=1)
    for k, j in enumerate(idx_map):
        gen_ids = out[k, input_lens[k]:]
        raw = tok.decode(gen_ids, skip_special_tokens=False)
        results[j] = sanitize_explanation(raw)
    return results


def main():
    ap = argparse.ArgumentParser()
    # Input dataset
    ap.add_argument("--dataset_name", default="open-r1/Mixture-of-Thoughts")
    ap.add_argument("--subset", default="code")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_rows", type=int, default=-1)
    ap.add_argument("--lang_filter", default="python", choices=["python","cpp","any"])

    # Summarizer (DeepSeek-R1)
    ap.add_argument("--ds_model_id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--ds_dtype", default="bfloat16", choices=["bfloat16","float16"])
    ap.add_argument("--ds_batch_size", type=int, default=16)
    ap.add_argument("--ds_max_new_tokens", type=int, default=8192)
    ap.add_argument("--ds_max_input_tokens", type=int, default=8192)

    # Target tokenizer (for token length checks)
    ap.add_argument("--target_model_id", default="Qwen/Qwen2-7B-Instruct")
    ap.add_argument("--max_seq_len", type=int, default=8192)

    # Output / streaming
    ap.add_argument("--out_jsonl", default="data/mot_r1_messages/data_next.jsonl")
    ap.add_argument("--flush_every", type=int, default=20)
    ap.add_argument("--resume", action="store_true")

    # Finalization
    ap.add_argument("--finalize", action="store_true",
                    help="Merge shards (if any) -> HF Dataset, save_to_disk, optional push_to_hub")
    ap.add_argument("--save_to_disk", default="data/mot_r1_messages_hf")
    ap.add_argument("--push_to_hub", default=None, help="e.g. user/mot-r1-processed")
    ap.add_argument("--config_name", default="messages")
    ap.add_argument("--cache_dir", default=None, help="HF cache_dir for load/save")

    args = ap.parse_args()

    if args.finalize:
        return finalize_jsonl(args)

    src = load_dataset(args.dataset_name, name=args.subset, split=args.split, cache_dir=args.cache_dir)
    if RANK == 0:
        print(f"Loaded source rows: {len(src)} (WORLD_SIZE={WORLD_SIZE})")

    ds_dtype = torch.bfloat16 if args.ds_dtype == "bfloat16" else torch.float16

    ds_model = AutoModelForCausalLM.from_pretrained(
        args.ds_model_id,
        device_map=f"cuda:{LOCAL_RANK}",
        torch_dtype=ds_dtype,
        trust_remote_code=True,
    )

    ds_tok = AutoTokenizer.from_pretrained(args.ds_model_id, trust_remote_code=True)

    # Target tokenizer for length accounting
    tgt_tok = AutoTokenizer.from_pretrained(args.target_model_id, trust_remote_code=True)
    if tgt_tok.pad_token is None:
        tgt_tok.pad_token = tgt_tok.eos_token
    tgt_tok.padding_side = "right"

    DEFAULT_SYS = "You are a helpful competitive programming assistant."

    shard_path = shard_jsonl(args.out_jsonl, RANK)
    done_idxs: Set[int] = load_done_indices(shard_path) if args.resume else set()
    if args.resume:
        print(f"[rank{RANK}] resume: {len(done_idxs)} rows already in {shard_path}")

    kept, skipped = 0, 0
    pending: List[Dict] = []

    def flush_pending():
        nonlocal pending, kept, skipped
        if not pending:
            return
        # Batch summarize thinks
        summaries = summarize_batch(
            [p["think"] for p in pending],
            ds_tok, ds_model,
            max_new_tokens=args.ds_max_new_tokens,
            max_input_tokens=args.ds_max_input_tokens
        )
        rows = []
        for p, expl in zip(pending, summaries):
            # Compose assistant message: brief explanation (no <think>) + final code fence
            parts = []
            if expl.strip():
                parts.append(expl.strip())
            parts.append(f"```{p['lang']}\n{p['code'].rstrip()}\n```")
            assistant_msg = {"role":"assistant", "content":"\n\n".join(parts)}
            out_messages = p["sys_user"] + [assistant_msg]

            # Token length guard w.r.t. target template
            serialized = tgt_tok.apply_chat_template(
                p["sys_user"] + [assistant_msg],
                tokenize=False,
                add_generation_prompt=False, 
            )
            tot_len = len(tgt_tok(serialized).input_ids)
            if tot_len > args.max_seq_len: 
                skipped += 1
                continue

            rows.append({
                "messages": out_messages,
                "language": p["lang"],
                "num_tokens_serialized": int(tot_len),
                "source_idx": int(p["source_idx"]),
            })
            kept += 1

        if rows:
            fsafe_append_jsonl(shard_path, rows)
            print(f"[rank{RANK}] flush: wrote {len(rows)} rows (kept={kept}, skipped={skipped}) -> {shard_path}")
        pending = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        for i, row in enumerate(src):
            if i % WORLD_SIZE != RANK:
                continue
            if args.resume and (i in done_idxs):
                continue

            msgs = row.get("messages")
            if not msgs:
                skipped += 1
                continue

            # find last assistant content
            atext = None
            for m in reversed(msgs):
                if m.get("role") == "assistant":
                    atext = m.get("content", "")
                    break
            if not atext:
                skipped += 1
                continue

            # extract final code fence
            hit = extract_last_code_block(atext)
            if not hit:
                skipped += 1
                continue
            lang, code = hit
            lang = guess_lang(lang, code)
            if args.lang_filter != "any" and lang != args.lang_filter:
                continue

            # collect sys+user turns
            sys_user = [m for m in msgs if m.get("role") in ("system","user")]
            sys_user = ensure_system_prefix(sys_user, DEFAULT_SYS)

            think = extract_think(atext) or ""
            pending.append({
                "source_idx": i,
                "sys_user": sys_user,
                "lang": lang,
                "code": code,
                "think": think,
            })

            if len(pending) >= args.ds_batch_size:
                flush_pending()

        # final flush
        flush_pending()

    except KeyboardInterrupt:
        flush_pending()
        print(f"[rank{RANK}] Interrupted; flushed pending.", file=sys.stderr)

    print(f"[rank{RANK}] Done. kept={kept}, skipped={skipped}. Shard at: {shard_path}")


def finalize_jsonl(args):
    root, ext = os.path.splitext(args.out_jsonl)
    shard_glob = f"{root}.rank*{ext}"
    shard_paths = sorted(glob.glob(shard_glob))
    merged_path = args.out_jsonl

    if shard_paths:
        os.makedirs(os.path.dirname(merged_path), exist_ok=True)
        with open(merged_path, "w", encoding="utf-8") as out_f:
            for sp in shard_paths:
                with open(sp, "r", encoding="utf-8") as inp:
                    for line in inp:
                        out_f.write(line)
        print(f"[finalize] merged {len(shard_paths)} shards -> {merged_path}")
    elif os.path.exists(merged_path):
        print(f"[finalize] using existing {merged_path}")
    else:
        print(f"[finalize] nothing to merge; no shards and no {merged_path}")
        return

    features = Features({
        "messages": Sequence(feature={"role": Value("string"), "content": Value("string")}),
        "language": Value("string"),
        "num_tokens_serialized": Value("int64"),
        "source_idx": Value("int64"),
    })

    ds = load_dataset(
        "json",
        data_files=merged_path,
        split="train",
        features=features,
        cache_dir=args.cache_dir
    )
    print(ds)

    # Save to disk
    if args.save_to_disk:
        os.makedirs(args.save_to_disk, exist_ok=True)
        ds.save_to_disk(args.save_to_disk)
        print(f"[finalize] saved HF dataset to {args.save_to_disk}")

    if args.push_to_hub:
        print(f"[finalize] pushing to Hub: {args.push_to_hub} (config: {args.config_name})")
        ds.push_to_hub(args.push_to_hub, config_name=args.config_name)
        print("[finalize] push complete")

if __name__ == "__main__":
    main()
