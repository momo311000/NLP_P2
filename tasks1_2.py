import csv
import json
import os
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import StrMethodFormatter
import numpy as np


CONFIG_FILE = "tasks1_2_config.json"

PLOT_FORMAT = "png"
DPI = 160
MAX_LENGTH_IN_PLOTS = 100
TOP_N_WORDS = 20


def load_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_path(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def save_csv(rows, path, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def clean_sentence(text):
    return text.strip().lower()


def valid_pair(en_line, fr_line):
    en_line = en_line.strip()
    fr_line = fr_line.strip()
    if not en_line or not fr_line:
        return False
    if en_line.startswith("<") or fr_line.startswith("<"):
        return False
    return True


def length_stats(lengths):
    values = np.array(lengths)
    return {
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": round(float(values.mean()), 2),
        "median": round(float(np.median(values)), 2),
        "p90": round(float(np.percentile(values, 90)), 2),
        "p95": round(float(np.percentile(values, 95)), 2),
    }


def analyse_data(config):
    en_path = get_path(config["english_file"])
    fr_path = get_path(config["french_file"])

    total = 0
    empty_removed = 0
    xml_removed = 0

    en_lengths = []
    fr_lengths = []
    en_words = Counter()
    fr_words = Counter()
    examples = []

    with en_path.open("r", encoding="utf-8", errors="replace") as en_file:
        with fr_path.open("r", encoding="utf-8", errors="replace") as fr_file:
            for en_line, fr_line in zip(en_file, fr_file):
                total += 1

                if not en_line.strip() or not fr_line.strip():
                    empty_removed += 1
                    continue

                if en_line.strip().startswith("<") or fr_line.strip().startswith("<"):
                    xml_removed += 1
                    continue

                en_clean = clean_sentence(en_line)
                fr_clean = clean_sentence(fr_line)

                en_tokens = en_clean.split()
                fr_tokens = fr_clean.split()

                en_lengths.append(len(en_tokens))
                fr_lengths.append(len(fr_tokens))
                en_words.update(en_tokens)
                fr_words.update(fr_tokens)

                if len(examples) < 5:
                    examples.append((en_clean, fr_clean))

    return {
        "total": total,
        "kept": len(en_lengths),
        "empty_removed": empty_removed,
        "xml_removed": xml_removed,
        "en_lengths": en_lengths,
        "fr_lengths": fr_lengths,
        "en_words": en_words,
        "fr_words": fr_words,
        "examples": examples,
    }


def save_cleaned_files(config):
    en_path = get_path(config["english_file"])
    fr_path = get_path(config["french_file"])
    data_dir = get_path(config["processed_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    en_out = data_dir / "cleaned_corpus.en"
    fr_out = data_dir / "cleaned_corpus.fr"
    count = 0

    with en_path.open("r", encoding="utf-8", errors="replace") as en_file:
        with fr_path.open("r", encoding="utf-8", errors="replace") as fr_file:
            with en_out.open("w", encoding="utf-8") as en_writer:
                with fr_out.open("w", encoding="utf-8") as fr_writer:
                    for en_line, fr_line in zip(en_file, fr_file):
                        if valid_pair(en_line, fr_line):
                            en_writer.write(clean_sentence(en_line) + "\n")
                            fr_writer.write(clean_sentence(fr_line) + "\n")
                            count += 1

    return count


def save_sample(config, clean_pairs):
    en_path = get_path(config["english_file"])
    fr_path = get_path(config["french_file"])
    data_dir = get_path(config["processed_dir"])

    random.seed(config["random_seed"])
    sample_size = round(clean_pairs * config["sample_ratio"])
    selected = set(random.sample(range(clean_pairs), sample_size))

    en_out = data_dir / "sample_10pct.en"
    fr_out = data_dir / "sample_10pct.fr"
    tsv_out = data_dir / "sample_10pct.tsv"

    pairs = []
    clean_index = 0

    with en_path.open("r", encoding="utf-8", errors="replace") as en_file:
        with fr_path.open("r", encoding="utf-8", errors="replace") as fr_file:
            with en_out.open("w", encoding="utf-8") as en_writer:
                with fr_out.open("w", encoding="utf-8") as fr_writer:
                    with tsv_out.open("w", encoding="utf-8", newline="") as tsv_file:
                        writer = csv.writer(tsv_file, delimiter="\t")
                        writer.writerow(["english", "french"])

                        for en_line, fr_line in zip(en_file, fr_file):
                            if not valid_pair(en_line, fr_line):
                                continue

                            if clean_index in selected:
                                en_clean = clean_sentence(en_line)
                                fr_clean = clean_sentence(fr_line)
                                en_writer.write(en_clean + "\n")
                                fr_writer.write(fr_clean + "\n")
                                writer.writerow([en_clean, fr_clean])
                                pairs.append((en_clean, fr_clean))

                            clean_index += 1

    return pairs


def save_splits(config, pairs):
    data_dir = get_path(config["processed_dir"])

    random.seed(config["random_seed"])
    random.shuffle(pairs)

    train_end = int(len(pairs) * config["train_ratio"])
    valid_end = train_end + int(len(pairs) * config["valid_ratio"])

    split_data = {
        "train": pairs[:train_end],
        "valid": pairs[train_end:valid_end],
        "test": pairs[valid_end:],
    }

    for name, split_pairs in split_data.items():
        with (data_dir / f"{name}.en").open("w", encoding="utf-8") as en_file:
            with (data_dir / f"{name}.fr").open("w", encoding="utf-8") as fr_file:
                with (data_dir / f"{name}.tsv").open("w", encoding="utf-8", newline="") as tsv_file:
                    writer = csv.writer(tsv_file, delimiter="\t")
                    writer.writerow(["english", "french"])
                    for en_sent, fr_sent in split_pairs:
                        en_file.write(en_sent + "\n")
                        fr_file.write(fr_sent + "\n")
                        writer.writerow([en_sent, fr_sent])

    return {name: len(split_pairs) for name, split_pairs in split_data.items()}


def plot_lengths(stats, output_dir):
    plt.figure(figsize=(10, 6))
    plt.hist(stats["en_lengths"], bins=60, range=(0, MAX_LENGTH_IN_PLOTS), alpha=0.65, color="#1f77b4")
    plt.hist(stats["fr_lengths"], bins=60, range=(0, MAX_LENGTH_IN_PLOTS), alpha=0.55, color="#ff7f0e")
    plt.title("Sentence length distribution")
    plt.xlabel("Sentence length in words")
    plt.ylabel("Number of sentence pairs")
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    plt.legend(
        handles=[
            Patch(facecolor="#1f77b4", alpha=0.65, label="English"),
            Patch(facecolor="#ff7f0e", alpha=0.55, label="French"),
            Patch(facecolor="#7f6f61", alpha=0.80, label="Overlap"),
        ]
    )
    plt.tight_layout()
    plt.savefig(output_dir / f"sentence_length_distribution.{PLOT_FORMAT}", dpi=DPI)
    plt.close()

    random.seed(42)
    pairs = list(zip(stats["en_lengths"], stats["fr_lengths"]))
    pairs = random.sample(pairs, min(5000, len(pairs)))

    plt.figure(figsize=(7, 7))
    plt.scatter([x for x, y in pairs], [y for x, y in pairs], s=8, alpha=0.25)
    plt.xlim(0, 120)
    plt.ylim(0, 120)
    plt.title("English vs French sentence length")
    plt.xlabel("English length in words")
    plt.ylabel("French length in words")
    plt.tight_layout()
    plt.savefig(output_dir / f"english_french_length_scatter.{PLOT_FORMAT}", dpi=DPI)
    plt.close()


def plot_words(counter, title, filename, output_dir):
    top_words = counter.most_common(TOP_N_WORDS)
    words = [word for word, count in top_words]
    counts = [count for word, count in top_words]

    plt.figure(figsize=(11, 6))
    bars = plt.bar(words, counts)
    plt.title(title)
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    plt.bar_label(bars, labels=[f"{count:,}" for count in counts], padding=3, fontsize=8, rotation=90)
    plt.tight_layout()
    plt.savefig(output_dir / f"{filename}.{PLOT_FORMAT}", dpi=DPI)
    plt.close()


def save_results(config, stats, cleaned_count, split_sizes):
    output_dir = get_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    en_stats = length_stats(stats["en_lengths"])
    fr_stats = length_stats(stats["fr_lengths"])
    diff = np.array(stats["fr_lengths"]) - np.array(stats["en_lengths"])

    rows = [
        ["total sentence pairs", stats["total"]],
        ["sentence pairs after cleaning", stats["kept"]],
        ["empty pairs removed", stats["empty_removed"]],
        ["xml/tag pairs removed", stats["xml_removed"]],
        ["english mean length", en_stats["mean"]],
        ["english median length", en_stats["median"]],
        ["english max length", en_stats["max"]],
        ["french mean length", fr_stats["mean"]],
        ["french median length", fr_stats["median"]],
        ["french max length", fr_stats["max"]],
        ["average french minus english length", round(float(diff.mean()), 2)],
        ["french longer pairs", int((diff > 0).sum())],
        ["english longer pairs", int((diff < 0).sum())],
        ["same length pairs", int((diff == 0).sum())],
        ["cleaned corpus pairs written", cleaned_count],
        ["train pairs", split_sizes["train"]],
        ["valid pairs", split_sizes["valid"]],
        ["test pairs", split_sizes["test"]],
    ]
    save_csv(rows, output_dir / "task1_2_statistics.csv", ["metric", "value"])

    save_csv(stats["en_words"].most_common(30), output_dir / "top_english_words.csv", ["word", "count"])
    save_csv(stats["fr_words"].most_common(30), output_dir / "top_french_words.csv", ["word", "count"])

    plot_lengths(stats, output_dir)
    plot_words(stats["en_words"], "Top 20 English words", "top_english_words", output_dir)
    plot_words(stats["fr_words"], "Top 20 French words", "top_french_words", output_dir)

    with (output_dir / "task1_2_notes.txt").open("w", encoding="utf-8") as f:
        f.write("Task 1 and 2 notes\n")
        f.write("==================\n\n")
        f.write(f"Original aligned sentence pairs: {stats['total']:,}\n")
        f.write(f"Pairs after cleaning: {stats['kept']:,}\n")
        f.write(f"Empty pairs removed: {stats['empty_removed']:,}\n")
        f.write(f"XML/tag pairs removed: {stats['xml_removed']:,}\n\n")
        f.write("Preprocessing used: strip spaces, lowercase text, remove empty pairs, remove tag lines.\n")
        f.write("I kept punctuation and stopwords because they can be important for translation.\n\n")
        f.write(f"English mean length: {en_stats['mean']} words\n")
        f.write(f"French mean length: {fr_stats['mean']} words\n")
        f.write(f"French sentences are about {round(float(diff.mean()), 2)} words longer on average.\n\n")
        f.write("Example cleaned sentence pairs:\n")
        for en_sent, fr_sent in stats["examples"]:
            f.write(f"EN: {en_sent}\n")
            f.write(f"FR: {fr_sent}\n\n")


def main():
    config = load_config()
    output_dir = get_path(config["output_dir"])
    data_dir = get_path(config["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Running tasks 1 and 2...")
    stats = analyse_data(config)

    print("Saving cleaned corpus...")
    cleaned_count = save_cleaned_files(config)

    print("Saving 10 percent sample...")
    sample_pairs = save_sample(config, stats["kept"])

    print("Saving train/valid/test splits...")
    split_sizes = save_splits(config, sample_pairs)

    print("Saving graphs and csv files...")
    save_results(config, stats, cleaned_count, split_sizes)

    print(f"Done. Outputs saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
