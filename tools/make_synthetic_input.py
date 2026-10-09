"""Build a synthetic ``/input_path`` from a labelled split, to exercise ``run.py`` end to end.

The real evaluation directory is not available, so this mirrors the documented layout
(``info.json`` + ``images/`` + ``patches/`` + ``queries.json``) using the training pack.  Both
query types are emitted so that the retrieval and grounding code paths are both covered, and
the script reports the ground truth it used so the outputs can be checked against it.

Usage::

    python tools/make_synthetic_input.py --split val --out .local_input --retrieval 4 --grounding 8
    python run.py --input .local_input --output .local_output
"""
import argparse
import json
import os
import shutil
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="qiyuan_patch_train_v1")
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", default=".local_input")
    parser.add_argument("--retrieval", type=int, default=4)
    parser.add_argument("--grounding", type=int, default=8)
    parser.add_argument("--copy-images", action="store_true",
                        help="copy image files instead of symlinking (works without symlink rights)")
    args = parser.parse_args()

    root = os.path.join(args.data, args.split)
    gt = json.load(open(os.path.join(root, "patch_gt.json"), encoding="utf-8"))
    image_dir = os.path.join(root, "images")
    patch_dir = os.path.join(root, "patches")

    out = args.out
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "images"))
    os.makedirs(os.path.join(out, "patches"))

    # every library image is exposed, with a fresh contiguous image_id
    names = sorted(os.listdir(image_dir))
    info = []
    for image_id, name in enumerate(names):
        source = os.path.abspath(os.path.join(image_dir, name))
        target = os.path.join(out, "images", name)
        if args.copy_images:
            shutil.copyfile(source, target)
        else:
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
        info.append({"image_id": image_id, "file_name": name})
    with open(os.path.join(out, "info.json"), "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=1)

    id_by_name = {name: i for i, name in enumerate(names)}
    keys = sorted(gt)
    retrieval_keys = keys[: args.retrieval]
    grounding_keys = keys[args.retrieval : args.retrieval + args.grounding]

    queries = []
    truth = {}
    next_id = 1
    for key in retrieval_keys:
        target = os.path.join(out, "patches", key)
        shutil.copyfile(os.path.join(patch_dir, key), target)
        queries.append({"query_id": next_id, "type": "retrieval", "image_patch": f"patches/{key}"})
        truth[next_id] = {"type": "retrieval", "image_id": id_by_name[gt[key]["image"]]}
        next_id += 1
    for key in grounding_keys:
        target = os.path.join(out, "patches", key)
        shutil.copyfile(os.path.join(patch_dir, key), target)
        image_id = id_by_name[gt[key]["image"]]
        queries.append({
            "query_id": next_id,
            "type": "grounding",
            "image_patch": f"patches/{key}",
            "image_id": image_id,
        })
        truth[next_id] = {
            "type": "grounding",
            "image_id": image_id,
            "bbox": gt[key]["bbox"],
        }
        next_id += 1

    with open(os.path.join(out, "queries.json"), "w", encoding="utf-8") as handle:
        json.dump(queries, handle, indent=1)
    with open(os.path.join(out, "truth.json"), "w", encoding="utf-8") as handle:
        json.dump(truth, handle, indent=1)

    print(f"wrote {out}: {len(info)} images, {len(queries)} queries "
          f"({len(retrieval_keys)} retrieval, {len(grounding_keys)} grounding)")
    print(f"ground truth for checking: {os.path.join(out, 'truth.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
