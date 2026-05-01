import argparse
import csv
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


class Tokenizer:
    def __init__(self, vocab_tokens: list[str]):
        self.pad_token = "<PAD>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        vocab = [self.pad_token, self.bos_token, self.eos_token] + vocab_tokens
        dedup = []
        seen = set()
        for tok in vocab:
            if tok not in seen:
                dedup.append(tok)
                seen.add(tok)

        self.stoi = {tok: i for i, tok in enumerate(dedup)}
        self.itos = {i: tok for tok, i in self.stoi.items()}

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


def equation_tokens(a: int, b: int, op: str, c: int, tokenizer: Tokenizer) -> list[str]:
    return [tokenizer.bos_token, str(a), op, str(b), "=", str(c), tokenizer.eos_token]


def modular_result(a: int, b: int, op: str, p: int) -> int:
    if op == "+":
        return (a + b) % p
    if op == "-":
        return (a - b) % p
    if op == "/":
        inv_b = pow(b, p - 2, p)
        return (a * inv_b) % p
    
    raise ValueError(f"Unsupported op: {op}")


def save_jsonl(path: str, rows: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def build_sanity_dataset(mask_prefix_tokens: int = 0):
    tokenizer = Tokenizer(["I", "love", "machine", "learning"])
    seq = [tokenizer.bos_token, "I", "love", "machine", "learning", tokenizer.eos_token]
    ids = tokenizer.encode_tokens(seq)

    x = ids[:-1]
    y = ids[1:]
    loss_mask = [1.0] * len(y)
    for i in range(min(mask_prefix_tokens, len(loss_mask))):
        loss_mask[i] = 0.0

    sample = {
        "x": x,
        "y": y,
        "loss_mask": loss_mask,
        "target_pos": None,
    }

    return tokenizer, [sample], [sample], [sample]


def build_modular_dataset(p: int, op: str, train_frac: float, val_frac: float, seed: int, save_split_dir: str | None = None):
    b_values = range(1, p) if op == "/" else range(0, p)

    pairs = [(a, b) for a in range(0, p) for b in b_values]
    rng = random.Random(seed)
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    n_test = n - n_train - n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError("Invalid split sizes. Adjust train_frac/val_frac.")

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train : n_train + n_val]
    test_pairs = pairs[n_train + n_val :]

    tokenizer = Tokenizer([str(i) for i in range(p)] + ["+", "-", "/", "="])

    def to_samples(subset_pairs):
        out = []
        records = []
        for a, b in subset_pairs:
            c = modular_result(a, b, op, p)
            seq_tokens = equation_tokens(a, b, op, c, tokenizer)
            ids = tokenizer.encode_tokens(seq_tokens)

            # y = [a, op, b, =, c, EOS], so answer token c is at index 4.
            x = ids[:-1]
            y = ids[1:]
            target_pos = 4
            loss_mask = [0.0] * len(y)
            loss_mask[target_pos] = 1.0

            out.append(
                {
                    "x": x,
                    "y": y,
                    "loss_mask": loss_mask,
                    "target_pos": target_pos,
                }
            )

            records.append({"a": a, "b": b, "c": c, "op": op, "p": p, "tokens": seq_tokens, "text": " ".join(seq_tokens)})
        
        return out, records

    train_data, train_records = to_samples(train_pairs)
    val_data, val_records = to_samples(val_pairs)
    test_data, test_records = to_samples(test_pairs)

    if save_split_dir is not None:
        os.makedirs(save_split_dir, exist_ok=True)
        save_jsonl(os.path.join(save_split_dir, "train.jsonl"), train_records)
        save_jsonl(os.path.join(save_split_dir, "val.jsonl"), val_records)
        save_jsonl(os.path.join(save_split_dir, "test.jsonl"), test_records)

        split_summary = {
            "p": p,
            "op": op,
            "train_frac": train_frac,
            "val_frac": val_frac,
            "seed": seed,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
            "total_size": len(train_data) + len(val_data) + len(test_data),
            "division_excludes_b_zero": op == "/",
        }
        with open(os.path.join(save_split_dir, "summary.json"), "w") as f:
            json.dump(split_summary, f, indent=2)

    return tokenizer, train_data, val_data, test_data


def collate_batch(samples: list[dict], pad_id: int):
    max_len = max(len(s["x"]) for s in samples)

    xs = []
    ys = []
    masks = []
    target_positions = []

    for s in samples:
        x = s["x"]
        y = s["y"]
        m = s["loss_mask"]
        pad_n = max_len - len(x)

        xs.append(x + [pad_id] * pad_n)
        ys.append(y + [pad_id] * pad_n)
        masks.append(m + [0.0] * pad_n)
        target_positions.append(s["target_pos"])

    x_t = torch.tensor(xs, dtype=torch.long, device=DEVICE)
    y_t = torch.tensor(ys, dtype=torch.long, device=DEVICE)
    m_t = torch.tensor(masks, dtype=torch.float32, device=DEVICE)
    return x_t, y_t, m_t, target_positions


def compute_masked_loss(logits, targets, loss_mask):
    b, t, v = logits.shape
    flat_loss = F.cross_entropy(
        logits.view(b * t, v),
        targets.view(b * t),
        reduction="none",
    ).view(b, t)
    masked = flat_loss * loss_mask
    denom = loss_mask.sum().clamp(min=1.0)
    return masked.sum() / denom


def get_lr(step: int, config: dict) -> float:
    base_lr = config["lr"]
    sched = config["lr_schedule"]
    if sched == "none":
        return base_lr
    if sched == "cosine":
        warmup_steps = max(0, config["warmup_steps"])
        min_lr = config["min_lr"]
        total_steps = max(1, config["steps"])

        if step <= warmup_steps and warmup_steps > 0:
            return base_lr * (step / warmup_steps)

        if total_steps <= warmup_steps:
            return base_lr

        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        progress = max(0.0, min(1.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + cosine * (base_lr - min_lr)

    raise ValueError(f"Unknown lr_schedule: {sched}")


@torch.no_grad()
def evaluate(model: GPT, data: list[dict], pad_id: int, batch_size: int):
    model.eval()
    total_loss = 0.0
    total_tokens = 0.0
    total_acc = 0
    total_count = 0

    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        x, y, loss_mask, target_positions = collate_batch(batch, pad_id)
        logits = model(x)

        loss = compute_masked_loss(logits, y, loss_mask)
        tokens_in_batch = loss_mask.sum().item()
        total_loss += loss.item() * tokens_in_batch
        total_tokens += tokens_in_batch

        pred = torch.argmax(logits, dim=-1)
        for row, tgt_pos in enumerate(target_positions):
            if tgt_pos is None:
                continue
            if int(pred[row, tgt_pos].item()) == int(y[row, tgt_pos].item()):
                total_acc += 1
            total_count += 1

    avg_loss = total_loss / max(total_tokens, 1.0)
    acc = total_acc / total_count if total_count > 0 else float("nan")
    return avg_loss, acc


@torch.no_grad()
def greedy_generate(model: GPT, start_ids: list[int], max_new_tokens: int):
    model.eval()
    idx = torch.tensor(start_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    for _ in range(max_new_tokens):
        logits = model(idx)
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)

    return idx.squeeze(0).tolist()


def write_history_files(out_dir: str, history: list[dict]):
    json_path = os.path.join(out_dir, "history.json")
    csv_path = os.path.join(out_dir, "history.csv")

    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)

    if history:
        keys = list(history[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(history)


def save_checkpoint(
    out_dir: str,
    model: GPT,
    optimizer,
    tokenizer: Tokenizer,
    config: dict,
    metrics: dict,
    gpt_cfg: GPTConfig,
    history: list[dict],
):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    with open(os.path.join(out_dir, "tokenizer.json"), "w") as f:
        json.dump(
            {
                "stoi": tokenizer.stoi,
                "bos_token": tokenizer.bos_token,
                "eos_token": tokenizer.eos_token,
                "pad_token": tokenizer.pad_token,
            },
            f,
            indent=2,
        )

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(out_dir, "gpt_config.json"), "w") as f:
        json.dump(
            {
                "block_size": gpt_cfg.block_size,
                "vocab_size": gpt_cfg.vocab_size,
                "n_layer": gpt_cfg.n_layer,
                "n_head": gpt_cfg.n_head,
                "n_embd": gpt_cfg.n_embd,
                "dropout": gpt_cfg.dropout,
                "bias": gpt_cfg.bias,
            },
            f,
            indent=2,
        )

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    write_history_files(out_dir, history)
    torch.save(optimizer.state_dict(), os.path.join(out_dir, "optimizer.pt"))


def train(config: dict):
    set_seed(config["seed"])
    os.makedirs(config["out_dir"], exist_ok=True)

    if config["mode"] == "sanity":
        tokenizer, train_data, val_data, test_data = build_sanity_dataset(
            mask_prefix_tokens=config["mask_prefix_tokens"]
        )
    elif config["mode"] == "modular":
        split_dir = os.path.join(config["out_dir"], "splits") if config["save_splits"] else None
        tokenizer, train_data, val_data, test_data = build_modular_dataset(
            p=config["p"],
            op=config["op"],
            train_frac=config["train_frac"],
            val_frac=config["val_frac"],
            seed=config["seed"],
            save_split_dir=split_dir,
        )
    else:
        raise ValueError(f"Unknown mode: {config['mode']}")

    block_size = max(len(s["x"]) for s in train_data)
    gpt_cfg = GPTConfig(
        block_size=block_size,
        vocab_size=len(tokenizer.stoi),
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        dropout=config["dropout"],
        bias=config["bias"],
    )

    model = GPT(gpt_cfg).to(DEVICE)
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
            betas=(config["beta1"], config["beta2"]),
            weight_decay=config["weight_decay"],
        )
    elif config["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            betas=(config["beta1"], config["beta2"]),
            weight_decay=config["weight_decay"],
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")

    pad_id = tokenizer.stoi[tokenizer.pad_token]
    steps = config["steps"]
    batch_size = config["batch_size"]

    best_val_loss = math.inf
    best_metrics = {}
    history = []
    start_step = 0
    elapsed_offset_sec = 0.0

    if config["resume"]:
        model_last_path = os.path.join(config["out_dir"], "model_last.pt")
        history_path = os.path.join(config["out_dir"], "history.json")
        optimizer_last_path = os.path.join(config["out_dir"], "optimizer_last.pt")
        metrics_path = os.path.join(config["out_dir"], "metrics.json")

        if os.path.exists(model_last_path):
            model.load_state_dict(torch.load(model_last_path, map_location=DEVICE))
            print(f"[resume] loaded model from {model_last_path}")
        else:
            print(f"[resume] model_last.pt not found in {config['out_dir']}; starting fresh")

        if os.path.exists(optimizer_last_path):
            try:
                optimizer.load_state_dict(torch.load(optimizer_last_path, map_location=DEVICE))
                print(f"[resume] loaded optimizer from {optimizer_last_path}")
            except Exception as e:
                print(f"[resume] failed to load optimizer state ({e}); continuing with fresh optimizer")

        if os.path.exists(history_path):
            try:
                with open(history_path, "r") as f:
                    history = json.load(f)
                if history:
                    start_step = int(history[-1].get("step", 0))
                    elapsed_offset_sec = float(history[-1].get("elapsed_sec", 0.0))
                    print(f"[resume] history found with last step={start_step}")
            except Exception as e:
                print(f"[resume] failed to load history ({e}); continuing from step 0")

        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r") as f:
                    best_metrics = json.load(f)
                best_val_loss = float(best_metrics.get("val_loss", math.inf))
            except Exception:
                pass

        if start_step >= steps:
            print(
                f"[resume] run already reached step={start_step} (target steps={steps}); "
                "nothing to do."
            )
            print("best metrics:", json.dumps(best_metrics, indent=2))
            print(f"checkpoint dir: {config['out_dir']}")
            return

    print(
        f"device={DEVICE} train={len(train_data)} val={len(val_data)} test={len(test_data)} "
        f"batch_size={batch_size}"
    )

    start_time = time.time()

    for step in range(start_step + 1, steps + 1):
        model.train()

        current_lr = get_lr(step, config)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        batch = random.sample(train_data, k=min(batch_size, len(train_data)))
        x, y, loss_mask, _ = collate_batch(batch, pad_id)

        logits = model(x)
        loss = compute_masked_loss(logits, y, loss_mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        optimizer.step()

        should_eval = step == 1 or step % config["eval_interval"] == 0 or step == steps
        if should_eval:
            train_loss, train_acc = evaluate(model, train_data, pad_id, batch_size)
            val_loss, val_acc = evaluate(model, val_data, pad_id, batch_size)
            test_loss, test_acc = evaluate(model, test_data, pad_id, batch_size)

            row = {
                "step": step,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "elapsed_sec": elapsed_offset_sec + (time.time() - start_time),
            }
            history.append(row)

            # Autosave current progress so interrupted runs can resume.
            write_history_files(config["out_dir"], history)
            torch.save(model.state_dict(), os.path.join(config["out_dir"], "model_last.pt"))
            torch.save(optimizer.state_dict(), os.path.join(config["out_dir"], "optimizer_last.pt"))

            if step == 1 or step % config["log_interval"] == 0 or step == steps:
                print(
                    f"step={step} train_loss={train_loss:.6f} train_acc={train_acc:.4f} "
                    f"val_loss={val_loss:.6f} val_acc={val_acc:.4f} "
                    f"test_loss={test_loss:.6f} test_acc={test_acc:.4f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = dict(row)
                save_checkpoint(
                    config["out_dir"],
                    model,
                    optimizer,
                    tokenizer,
                    config,
                    best_metrics,
                    gpt_cfg,
                    history,
                )

    if config["mode"] == "sanity":
        generated_ids = greedy_generate(
            model,
            [tokenizer.stoi[tokenizer.bos_token]],
            max_new_tokens=5,
        )
        print("generated:", tokenizer.decode_ids(generated_ids))

    # Always save final history even if no improvement happened.
    write_history_files(config["out_dir"], history)

    # Save final model as an auxiliary artifact.
    torch.save(model.state_dict(), os.path.join(config["out_dir"], "model_last.pt"))
    torch.save(optimizer.state_dict(), os.path.join(config["out_dir"], "optimizer_last.pt"))

    print("best metrics:", json.dumps(best_metrics, indent=2))
    print(f"checkpoint dir: {config['out_dir']}")


def parse_args():
    parser = argparse.ArgumentParser(description="CSE 493 trainer")

    parser.add_argument("--mode", type=str, default="sanity", choices=["sanity", "modular"])
    parser.add_argument("--out_dir", type=str, default="out/run")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--lr_schedule", type=str, default="none", choices=["none", "cosine"])
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=100)

    parser.add_argument("--n_layer", type=int, default=1)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--n_embd", type=int, default=32)
    parser.add_argument("--bias", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--mask_prefix_tokens", type=int, default=0)

    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--op", type=str, default="+", choices=["+", "-", "/"])
    parser.add_argument("--train_frac", type=float, default=0.3)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--save_splits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)

    return parser.parse_args()


def main():
    args = parse_args()
    config = vars(args)
    train(config)


if __name__ == "__main__":
    main()
