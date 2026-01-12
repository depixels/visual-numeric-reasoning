"""Analyze Stage1 hierarchical contrastive pairs."""

import argparse
import json
from collections import Counter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Stage1 hierarchical pairs")
    parser.add_argument("--jsonl", required=True)
    return parser.parse_args()


def _load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    args = _parse_args()

    hand_config_counts = Counter()
    seconds_counts = Counter()
    delta_seconds_hist = Counter()

    for row in _load_jsonl(args.jsonl):
        meta = row.get("meta", {})
        hand_config_counts[str(meta.get("hand_config_a", "unknown"))] += 1
        hand_config_counts[str(meta.get("hand_config_b", "unknown"))] += 1

        label = row.get("label", {})
        if label.get("time_a_seconds_total") is not None:
            seconds_counts[int(label.get("time_a_seconds_total")) % 60] += 1
        if label.get("time_b_seconds_total") is not None:
            seconds_counts[int(label.get("time_b_seconds_total")) % 60] += 1

        if meta.get("bucket") == "neg_hard_hms" and label.get("delta_seconds") is not None:
            delta_seconds_hist[int(label.get("delta_seconds"))] += 1

    print("hand_config_counts:", dict(hand_config_counts))
    print("seconds_coverage:", dict(sorted(seconds_counts.items())))
    print("delta_seconds_hist:", dict(sorted(delta_seconds_hist.items())))


if __name__ == "__main__":
    main()
