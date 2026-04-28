import argparse
import json
import random
from pathlib import Path


OPS = ["+", "-", "/"]
OP_NAME = {"+": "plus", "-": "minus", "/": "div"}


def mod_result(a, b, op, p):
    a, b = a % p, b % p
    if op == "+":
        return (a + b) % p
    if op == "-":
        return (a - b) % p
    return (a * pow(b, p - 2, p)) % p


def make_examples(p, op):
    out = []
    for a in range(p + 1):
        for b in range(p + 1):
            if op == "/" and b % p == 0:
                continue
            out.append(f"{a} {op} {b} = {mod_result(a, b, op, p)}")
    return out


def split_data(examples, train, val, seed):
    data = list(examples)
    random.Random(seed).shuffle(data)
    n = len(data)
    n_train = int(n * train)
    n_val = int(n * val)
    return data[:n_train], data[n_train : n_train + n_val], data[n_train + n_val :]


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_for_prime(p, out_dir, train, val, seed):
    prime_dir = out_dir / f"p{p}"
    prime_dir.mkdir(parents=True, exist_ok=True)

    combined = {"train": [], "val": [], "test": []}
    per_op = {}

    for i, op in enumerate(OPS):
        examples = make_examples(p, op)
        tr, va, te = split_data(examples, train, val, seed + p * 100 + i)

        name = OP_NAME[op]
        write_lines(prime_dir / f"{name}_train.txt", tr)
        write_lines(prime_dir / f"{name}_val.txt", va)
        write_lines(prime_dir / f"{name}_test.txt", te)

        combined["train"].extend(tr)
        combined["val"].extend(va)
        combined["test"].extend(te)

        per_op[name] = {
            "total": len(examples),
            "train": len(tr),
            "val": len(va),
            "test": len(te),
        }

    random.Random(seed + p * 1000 + 1).shuffle(combined["train"])
    random.Random(seed + p * 1000 + 2).shuffle(combined["val"])
    random.Random(seed + p * 1000 + 3).shuffle(combined["test"])

    write_lines(prime_dir / "all_train.txt", combined["train"])
    write_lines(prime_dir / "all_val.txt", combined["val"])
    write_lines(prime_dir / "all_test.txt", combined["test"])

    meta = {
        "p": p,
        "range": "0 <= a,b <= p",
        "ops": OPS,
        "split_fractions": {"train": train, "val": val, "test": 1.0 - train - val},
        "notes": "Division excludes b where b mod p = 0 (b in {0, p}).",
        "per_operator_counts": per_op,
        "all_counts": {
            "train": len(combined["train"]),
            "val": len(combined["val"]),
            "test": len(combined["test"]),
            "total": len(combined["train"]) + len(combined["val"]) + len(combined["test"]),
        },
    }

    (prime_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/mod_arithmetic"))
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [
        build_for_prime(p, args.output_dir, args.train_frac, args.val_frac, args.seed)
        for p in [97, 113]
    ]
    summary = args.output_dir / "summary.json"
    summary.write_text(json.dumps({"datasets": datasets}, indent=2), encoding="utf-8")
    print(f"Generated datasets at: {args.output_dir}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
