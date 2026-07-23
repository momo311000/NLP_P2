import csv
import json
import random
from contextlib import ExitStack
from pathlib import Path

from tasks1_2 import clean_sentence, valid_pair


ROOT = Path(__file__).parent
CONFIG_FILE = ROOT / "task5_config.json"


def resolve_path(path_text):
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def count_clean_pairs(en_path, sv_path):
    total = 0
    clean = 0
    with en_path.open("r", encoding="utf-8", errors="replace") as en_file:
        with sv_path.open("r", encoding="utf-8", errors="replace") as sv_file:
            for en_line, sv_line in zip(en_file, sv_file):
                total += 1
                if valid_pair(en_line, sv_line):
                    clean += 1
    return total, clean


def prepare_data(config):
    en_path = resolve_path(config["english_swedish_english_file"])
    sv_path = resolve_path(config["english_swedish_swedish_file"])
    output_dir = resolve_path(config["task5_processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    total, clean_count = count_clean_pairs(en_path, sv_path)
    sample_count = round(clean_count * config["sample_ratio"])

    rng = random.Random(config["random_seed"])
    selected = rng.sample(range(clean_count), sample_count)
    rng.shuffle(selected)

    train_count = int(sample_count * config["train_ratio"])
    valid_count = int(sample_count * config["valid_ratio"])
    assignment = {}
    for position, clean_index in enumerate(selected):
        if position < train_count:
            assignment[clean_index] = "train"
        elif position < train_count + valid_count:
            assignment[clean_index] = "valid"
        else:
            assignment[clean_index] = "test"

    counts = {"train": 0, "valid": 0, "test": 0}
    with ExitStack() as stack:
        writers = {
            (split, lang): stack.enter_context(
                (output_dir / f"{split}.{lang}").open("w", encoding="utf-8")
            )
            for split in counts
            for lang in ("en", "sv")
        }
        with en_path.open("r", encoding="utf-8", errors="replace") as en_file:
            with sv_path.open("r", encoding="utf-8", errors="replace") as sv_file:
                clean_index = 0
                for en_line, sv_line in zip(en_file, sv_file):
                    if not valid_pair(en_line, sv_line):
                        continue
                    split = assignment.get(clean_index)
                    if split is not None:
                        writers[(split, "en")].write(clean_sentence(en_line) + "\n")
                        writers[(split, "sv")].write(clean_sentence(sv_line) + "\n")
                        counts[split] += 1
                    clean_index += 1

    summary = {
        "total_pairs": total,
        "clean_pairs": clean_count,
        "sampled_pairs": sample_count,
        "train_pairs": counts["train"],
        "valid_pairs": counts["valid"],
        "test_pairs": counts["test"],
    }
    with (output_dir / "preprocessing_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    return summary


def main():
    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        config = json.load(f)
    summary = prepare_data(config)
    print("Task 5 English-Swedish data prepared")
    for name, value in summary.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
