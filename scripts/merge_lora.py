#!/usr/bin/env python
import argparse, os, json, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default=None,
                    help="Base model id/path. If omitted, use adapter's recorded base.")
    ap.add_argument("--adapter_path", required=True,
                    help="LoRA checkpoint dir (trainer output_dir)")
    ap.add_argument("--output_dir", required=True,
                    help="Where to save merged model")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16","bfloat16"])
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    adapter_path = Path(args.adapter_path)

    base_id = args.base_model
    if base_id is None:
        raise ValueError("Cannot determine base model. Pass --base_model or ensure adapter has base_model_name_or_path.")


    tok = AutoTokenizer.from_pretrained(adapter_path.as_posix(), trust_remote_code=True)

    print(f"Loading base: {base_id}")
    base = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=dtype, device_map="cpu", trust_remote_code=True
    )

    base_vocab = base.get_input_embeddings().weight.size(0)
    tok_vocab  = len(tok)
    if base_vocab != tok_vocab:
        print(f"Resizing token embeddings: base_vocab={base_vocab} -> tok_vocab={tok_vocab}")
        base.resize_token_embeddings(tok_vocab)
        # From reddit post on getting model weights to match
        if hasattr(base, "tie_weights"):
            try: base.tie_weights()
            except Exception: pass

    print("Attaching adapter…")
    peft_model = PeftModel.from_pretrained(base, adapter_path.as_posix())
    print("Merging and unloading…")
    merged = peft_model.merge_and_unload()

    # 6) Save merged model + tokenizer
    os.makedirs(args.output_dir, exist_ok=True)
    print("Saving merged model…")
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    print("Saving tokenizer…")
    tok.save_pretrained(args.output_dir)
    print("Done. Merged model at:", args.output_dir)

if __name__ == "__main__":
    main()