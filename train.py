import argparse
import json
import math
import os
import random
from dataclasses import asdict

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
        # b in [1, p-1] for prime p; inverse exists
        inv_b = pow(b, p - 2, p)
        return (a * inv_b) % p
    raise ValueError(f"Unsupported op: {op}")


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


def build_modular_dataset(
    p: int,
    op: str,
    train_frac: float,
    val_frac: float,
    seed: int,
):
    # For division we skip b=0 to avoid undefined inverse.
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
        for a, b in subset_pairs:
            c = modular_result(a, b, op, p)
            seq = equation_tokens(a, b, op, c, tokenizer)
            ids = tokenizer.encode_tokens(seq)

            # Sequence layout after shift:
            # y tokens are [a, op, b, =, c, EOS]
            # We train loss only on c by default (position 4 in y).
            x = ids[:-1]
            y = ids[1:]
            loss_mask = [0.0] * len(y)
            target_pos = 4
            loss_mask[target_pos] = 1.0

            out.append(
                {
                    "x": x,
                    "y": y,
                    "loss_mask": loss_mask,
                    "target_pos": target_pos,
                }
            )
        return out

    train_data = to_samples(train_pairs)
    val_data = to_samples(val_pairs)
    test_data = to_samples(test_pairs)
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

        # Accuracy on the answer token position.
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


def save_checkpoint(out_dir: str, model: GPT, tokenizer: Tokenizer, config: dict, metrics: dict):
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
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def train(config: dict):
    set_seed(config["seed"])
    # Assignment-focused defaults (kept internal to simplify CLI).
    batch_size = 64
    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.95
    weight_decay = 0.0
    log_interval = 100
    dropout = 0.0

    if config["mode"] == "sanity":
        tokenizer, train_data, val_data, test_data = build_sanity_dataset(
            mask_prefix_tokens=config["mask_prefix_tokens"]
        )
    elif config["mode"] == "modular":
        tokenizer, train_data, val_data, test_data = build_modular_dataset(
            p=config["p"],
            op=config["op"],
            train_frac=0.3,
            val_frac=0.1,
            seed=config["seed"],
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
        dropout=dropout,
        bias=config["bias"],
    )

    model = GPT(gpt_cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )

    pad_id = tokenizer.stoi[tokenizer.pad_token]
    steps = config["steps"]

    best_val_loss = math.inf
    best_metrics = {}

    print(f"device={DEVICE} train={len(train_data)} val={len(val_data)} test={len(test_data)}")

    for step in range(1, steps + 1):
        model.train()

        batch = random.sample(train_data, k=min(batch_size, len(train_data)))
        x, y, loss_mask, _ = collate_batch(batch, pad_id)

        logits = model(x)
        loss = compute_masked_loss(logits, y, loss_mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % log_interval == 0 or step == steps:
            train_loss, train_acc = evaluate(model, train_data, pad_id, batch_size)
            val_loss, val_acc = evaluate(model, val_data, pad_id, batch_size)
            print(
                f"step={step} train_loss={train_loss:.6f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.6f} val_acc={val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                test_loss, test_acc = evaluate(model, test_data, pad_id, batch_size)
                best_metrics = {
                    "step": step,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                }
                save_checkpoint(config["out_dir"], model, tokenizer, config, best_metrics)

    if config["mode"] == "sanity":
        generated_ids = greedy_generate(
            model,
            [tokenizer.stoi[tokenizer.bos_token]],
            max_new_tokens=5,
        )
        print("generated:", tokenizer.decode_ids(generated_ids))

    print("best metrics:", json.dumps(best_metrics, indent=2))
    print(f"checkpoint dir: {config['out_dir']}")


def parse_args():
    parser = argparse.ArgumentParser(description="CSE 493 trainer")

    parser.add_argument("--mode", type=str, default="sanity", choices=["sanity", "modular"])
    parser.add_argument("--out_dir", type=str, default="out/run")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1200)

    parser.add_argument("--n_layer", type=int, default=1)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--n_embd", type=int, default=32)
    parser.add_argument("--bias", action="store_true")

    # Sanity mode only
    parser.add_argument("--mask_prefix_tokens", type=int, default=0)

    # Modular mode only
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--op", type=str, default="+", choices=["+", "-", "/"])

    return parser.parse_args()


def main():
    args = parse_args()
    config = vars(args)
    train(config)


if __name__ == "__main__":
    main()
