"""Score ``run.py``'s ``pred.json`` against the synthetic input's ``truth.json``.

This is the same metric the competition uses, so it doubles as an independent check on the
in-process evaluation harness: both should report the same numbers for the same patches.
"""
import argparse
import json
import os
import sys


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=".local_input")
    parser.add_argument("--pred", default=".local_output/pred.json")
    args = parser.parse_args()

    truth = json.load(open(os.path.join(args.input, "truth.json"), encoding="utf-8"))
    truth = {int(k): v for k, v in truth.items()}
    pred = json.load(open(args.pred, encoding="utf-8"))
    print(f"pred.json records: {len(pred)}  truth entries: {len(truth)}")

    retrieval = {r["query_id"]: r for r in pred if r.get("type") == "retrieval"}
    grounding = {r["query_id"]: r for r in pred if r.get("type") == "grounding"}

    r1 = r5 = n_ret = 0
    for query_id, entry in truth.items():
        if entry["type"] != "retrieval":
            continue
        n_ret += 1
        record = retrieval.get(query_id)
        if record is None:
            print(f"  query {query_id}: MISSING retrieval record")
            continue
        ranking = list(record.get("ranking") or [])
        if "ranking" not in record:
            print(f"  query {query_id}: record has no ranking field")
        target = entry["image_id"]
        r1 += bool(ranking and ranking[0] == target)
        r5 += target in ranking[:5]

    a50 = a75 = n_ground = 0
    ious = []
    for query_id, entry in truth.items():
        if entry["type"] != "grounding":
            continue
        n_ground += 1
        record = grounding.get(query_id)
        if record is None:
            print(f"  query {query_id}: MISSING grounding record")
            continue
        if record["image_id"] != entry["image_id"]:
            print(f"  query {query_id}: wrong image_id {record['image_id']} != {entry['image_id']}")
        value = iou(record["bbox"], entry["bbox"])
        ious.append(value)
        a50 += value >= 0.5
        a75 += value >= 0.75

    print()
    print(f"retrieval queries scored : {n_ret}")
    print(f"grounding queries scored : {n_ground}")
    if n_ret:
        print(f"R@1     = {r1/n_ret:.4f}")
        print(f"R@5     = {r5/n_ret:.4f}")
    if n_ground:
        print(f"Acc@0.5 = {a50/n_ground:.4f}")
        print(f"Acc@0.75= {a75/n_ground:.4f}")
        print(f"mean IoU= {sum(ious)/len(ious):.4f}  exact: {sum(1 for v in ious if v == 1.0)}/{n_ground}")

    # Combined score uses whatever counts exist, so a synthetic run with few queries still
    # yields a comparable number.
    total = 0.0
    parts = []
    if n_ret:
        parts.append((0.1, r1 / n_ret))
        parts.append((0.2, r5 / n_ret))
    if n_ground:
        parts.append((0.6, a50 / n_ground))
        parts.append((0.1, a75 / n_ground))
    weight = sum(w for w, _ in parts)
    if weight:
        total = sum(w * v for w, v in parts) / weight
    print(f"weighted score (normalised) = {total:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
