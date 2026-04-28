import argparse
import json
import os

import torch

from model import GPT, GPTConfig


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()


class Tokenizer:
    def __init__(self, stoi: dict, bos_token: str, eos_token: str, pad_token: str):
        self.stoi = stoi
        self.itos = {v: k for k, v in stoi.items()}
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

    def encode_tokens(self, tokens: list[str]) -> list[int]:
        ids = []
        for tok in tokens:
            if tok not in self.stoi:
                raise ValueError(f"Unknown token: {tok}")
            ids.append(self.stoi[tok])
        return ids

    def decode_ids(self, ids: list[int]) -> list[str]:
        toks = []
        for idx in ids:
            if idx not in self.itos:
                raise ValueError(f"Unknown token id: {idx}")
            toks.append(self.itos[idx])
        return toks


def load_model_and_tokenizer(checkpoint_dir: str):
    tok_path = os.path.join(checkpoint_dir, "tokenizer.json")
    model_path = os.path.join(checkpoint_dir, "model.pt")
    gpt_cfg_path = os.path.join(checkpoint_dir, "gpt_config.json")
    run_cfg_path = os.path.join(checkpoint_dir, "config.json")

    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"Missing tokenizer file: {tok_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model file: {model_path}")

    with open(tok_path, "r") as f:
        tok_data = json.load(f)

    tokenizer = Tokenizer(
        stoi=tok_data["stoi"],
        bos_token=tok_data["bos_token"],
        eos_token=tok_data["eos_token"],
        pad_token=tok_data["pad_token"],
    )

    if os.path.exists(gpt_cfg_path):
        with open(gpt_cfg_path, "r") as f:
            cfg_data = json.load(f)
    else:
        if not os.path.exists(run_cfg_path):
            raise FileNotFoundError(
                f"Missing both gpt_config.json and config.json in {checkpoint_dir}"
            )
        with open(run_cfg_path, "r") as f:
            run_cfg = json.load(f)
        cfg_data = {
            "block_size": 6,
            "vocab_size": len(tokenizer.stoi),
            "n_layer": run_cfg["n_layer"],
            "n_head": run_cfg["n_head"],
            "n_embd": run_cfg["n_embd"],
            "dropout": 0.0,
            "bias": run_cfg.get("bias", False),
        }

    model_cfg = GPTConfig(**cfg_data)
    model = GPT(model_cfg)

    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    return model, tokenizer


@torch.no_grad()
def greedy_generate(model: GPT, start_ids: list[int], max_new_tokens: int) -> list[int]:
    idx = torch.tensor(start_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    for _ in range(max_new_tokens):
        logits = model(idx)
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
    return idx.squeeze(0).tolist()


@torch.no_grad()
def predict_answer(model: GPT, tokenizer: Tokenizer, a: int, b: int, op: str) -> int:
    prompt = [tokenizer.bos_token, str(a), op, str(b), "="]
    prompt_ids = tokenizer.encode_tokens(prompt)

    out_ids = greedy_generate(model, prompt_ids, max_new_tokens=1)
    answer_id = out_ids[-1]
    answer_tok = tokenizer.decode_ids([answer_id])[0]
    return int(answer_tok)


def run_sanity_generation(model: GPT, tokenizer: Tokenizer):
    start = [tokenizer.stoi[tokenizer.bos_token]]
    out = greedy_generate(model, start, max_new_tokens=5)
    tokens = tokenizer.decode_ids(out)
    print("generated_ids:", out)
    print("generated_tokens:", tokens)


def run_custom_prompt_generation(model: GPT, tokenizer: Tokenizer, prompt_tokens: list[str], max_new_tokens: int):
    prompt_ids = tokenizer.encode_tokens(prompt_tokens)
    out = greedy_generate(model, prompt_ids, max_new_tokens=max_new_tokens)
    tokens = tokenizer.decode_ids(out)
    print("prompt_tokens:", prompt_tokens)
    print("generated_ids:", out)
    print("generated_tokens:", tokens)


def run_equation_prediction(model: GPT, tokenizer: Tokenizer, a: int, b: int, op: str):
    ans = predict_answer(model, tokenizer, a, b, op)
    print(f"prediction: {a} {op} {b} = {ans}")


def parse_args():
    parser = argparse.ArgumentParser(description="Inference utility for HW1 checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--task", type=str, default="sanity", choices=["sanity", "equation", "custom"])
    parser.add_argument("--a", type=int, default=3)
    parser.add_argument("--b", type=int, default=5)
    parser.add_argument("--op", type=str, default="+", choices=["+", "-", "/"])
    parser.add_argument(
        "--prompt_tokens",
        type=str,
        default="",
        help="Comma-separated tokens for custom task, e.g. '<BOS>,I,love,machine'",
    )
    parser.add_argument("--max_new_tokens", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"device={DEVICE}")
    model, tokenizer = load_model_and_tokenizer(args.checkpoint_dir)

    if args.task == "sanity":
        run_sanity_generation(model, tokenizer)
    elif args.task == "equation":
        run_equation_prediction(model, tokenizer, args.a, args.b, args.op)
    else:
        if not args.prompt_tokens:
            raise ValueError("--prompt_tokens is required for --task custom")
        prompt_tokens = [tok.strip() for tok in args.prompt_tokens.split(",") if tok.strip()]
        run_custom_prompt_generation(
            model,
            tokenizer,
            prompt_tokens=prompt_tokens,
            max_new_tokens=args.max_new_tokens,
        )


if __name__ == "__main__":
    main()
