import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


def load_split_pairs(split_file: Path) -> List[Tuple[str, str]]:
    with split_file.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    pairs: List[Tuple[str, str]] = []
    for item in raw:
        if not isinstance(item, list) or len(item) != 2:
            continue
        a, b = str(item[0]), str(item[1])
        pairs.append((a, b))
    return pairs


def collect_png_frames(video_dir: Path) -> List[str]:
    if not video_dir.exists():
        return []
    frames = [p.as_posix() for p in sorted(video_dir.glob("*.png"))]
    return frames


def build_real_entries(
    ffpp_root: Path,
    split_pairs: List[Tuple[str, str]],
    split_name: str,
    compression: str,
) -> Dict[str, Dict[str, object]]:
    real_entries: Dict[str, Dict[str, object]] = {}
    real_frames_root = ffpp_root / "original_sequences" / "youtube" / compression / "frames"
    seen_ids: Set[str] = set()

    for a, b in split_pairs:
        seen_ids.add(a)
        seen_ids.add(b)

    for vid in sorted(seen_ids):
        frame_dir = real_frames_root / vid
        frames = collect_png_frames(frame_dir)
        if not frames:
            continue
        real_entries[vid] = {
            "label": "FF-real",
            "frames": frames,
        }
    return real_entries


def build_fake_entries_for_method(
    ffpp_root: Path,
    split_pairs: List[Tuple[str, str]],
    split_name: str,
    compression: str,
    method_dir_name: str,
    label_name: str,
) -> Dict[str, Dict[str, object]]:
    fake_entries: Dict[str, Dict[str, object]] = {}
    fake_frames_root = ffpp_root / "manipulated_sequences" / method_dir_name / compression / "frames"
    if not fake_frames_root.exists():
        return fake_entries

    for a, b in split_pairs:
        candidates = [f"{a}_{b}", f"{b}_{a}"]
        for video_name in candidates:
            frame_dir = fake_frames_root / video_name
            frames = collect_png_frames(frame_dir)
            if not frames:
                continue
            fake_entries[video_name] = {
                "label": label_name,
                "frames": frames,
            }
    return fake_entries


def build_dataset_json(ffpp_root: Path, compression: str) -> Dict[str, object]:
    split_map = {
        "train": load_split_pairs(ffpp_root / "train.json"),
        "val": load_split_pairs(ffpp_root / "val.json"),
        "test": load_split_pairs(ffpp_root / "test.json"),
    }

    method_map = {
        "FF-DF": "Deepfakes",
        "FF-F2F": "Face2Face",
        "FF-FS": "FaceSwap",
        "FF-NT": "NeuralTextures",
    }

    dataset: Dict[str, object] = {
        "FaceForensics++": {
            "FF-real": {
                "train": {compression: {}},
                "test": {compression: {}},
            },
            "FF-DF": {
                "train": {compression: {}},
                "test": {compression: {}},
            },
            "FF-F2F": {
                "train": {compression: {}},
                "test": {compression: {}},
            },
            "FF-FS": {
                "train": {compression: {}},
                "test": {compression: {}},
            },
            "FF-NT": {
                "train": {compression: {}},
                "test": {compression: {}},
            },
        }
    }

    # ForensicsAdapter only reads train/test. Merge val into train.
    train_pairs = split_map["train"] + split_map["val"]
    test_pairs = split_map["test"]

    real_train = build_real_entries(ffpp_root, train_pairs, "train", compression)
    real_test = build_real_entries(ffpp_root, test_pairs, "test", compression)
    dataset["FaceForensics++"]["FF-real"]["train"][compression] = real_train
    dataset["FaceForensics++"]["FF-real"]["test"][compression] = real_test

    for label_name, method_dir_name in method_map.items():
        fake_train = build_fake_entries_for_method(
            ffpp_root, train_pairs, "train", compression, method_dir_name, label_name
        )
        fake_test = build_fake_entries_for_method(
            ffpp_root, test_pairs, "test", compression, method_dir_name, label_name
        )
        dataset["FaceForensics++"][label_name]["train"][compression] = fake_train
        dataset["FaceForensics++"][label_name]["test"][compression] = fake_test

    return dataset


def print_summary(dataset_json: Dict[str, object], compression: str) -> None:
    root = dataset_json["FaceForensics++"]
    print("Summary (video count):")
    for label in ["FF-real", "FF-DF", "FF-F2F", "FF-FS", "FF-NT"]:
        train_count = len(root[label]["train"][compression])
        test_count = len(root[label]["test"][compression])
        print(f"  {label}: train={train_count}, test={test_count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FaceForensics++.json for ForensicsAdapter from DeepfakeBench FF++ layout."
    )
    parser.add_argument(
        "--ffpp_root",
        type=str,
        default="/public/zxj/DeepfakeBench/datasets/rgb/FaceForensics++",
        help="Path to DeepfakeBench FaceForensics++ root",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="c23",
        choices=["c23", "c40"],
        help="Compression level to index",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/public/zxj/ForensicsAdapter/data/FaceForensics++.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    ffpp_root = Path(args.ffpp_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_json = build_dataset_json(ffpp_root, args.compression)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_json, f, ensure_ascii=False)

    print(f"Saved: {output_path.as_posix()}")
    print_summary(dataset_json, args.compression)


if __name__ == "__main__":
    main()
