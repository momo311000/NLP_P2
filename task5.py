import copy
import csv
import json
import os
from pathlib import Path

import torch
import torch.nn as nn

from task4 import (
    AttentionDecoder,
    AttentionEncoder,
    Seq2SeqAttention,
    evaluate_loss_attention,
    run_predictions_with_attention,
    train_one_epoch_attention,
)
from tasks3 import (
    EOS,
    PAD,
    SOS,
    Vocab,
    filter_pairs,
    get_device,
    get_teacher_forcing,
    make_loaders,
    read_split,
    save_csv,
    set_seed,
    simple_bleu,
    token_accuracy,
)


ROOT = Path(__file__).parent
CONFIG_FILE = ROOT / "task5_config.json"


def resolve_path(path_text):
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def translate_text(model, src_vocab, trg_vocab, text, max_len, device):
    if hasattr(model, "eval"):
        model.eval()
    src_ids = src_vocab.encode(text, "word", add_sos=False, add_eos=True)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_len = torch.tensor([len(src_ids)])
    with torch.no_grad():
        output_ids = model.translate(
            src,
            src_len,
            trg_vocab.stoi[SOS],
            trg_vocab.stoi[EOS],
            max_len + 10,
        )
    return " ".join(trg_vocab.decode(output_ids))


def build_pivot_rows(fr_en_bundle, en_sv_bundle, limit, max_len, device):
    rows = []
    for french, english_reference in fr_en_bundle["test_pairs"][:limit]:
        english_prediction = translate_text(
            fr_en_bundle["model"],
            fr_en_bundle["src_vocab"],
            fr_en_bundle["trg_vocab"],
            french,
            max_len,
            device,
        )
        swedish_oracle = translate_text(
            en_sv_bundle["model"],
            en_sv_bundle["src_vocab"],
            en_sv_bundle["trg_vocab"],
            english_reference,
            max_len,
            device,
        )
        swedish_pivot = translate_text(
            en_sv_bundle["model"],
            en_sv_bundle["src_vocab"],
            en_sv_bundle["trg_vocab"],
            english_prediction,
            max_len,
            device,
        )
        rows.append({
            "french_source": french,
            "english_reference": english_reference,
            "english_prediction": english_prediction,
            "swedish_from_reference_english": swedish_oracle,
            "swedish_from_predicted_english": swedish_pivot,
        })
    return rows


def train_component(name, data_dir, source_lang, target_lang, config, device):
    max_len = config["max_len_words"]
    train_pairs = filter_pairs(
        read_split(data_dir, "train", source_lang, target_lang, config["max_train_pairs"]),
        "word",
        max_len,
    )
    valid_pairs = filter_pairs(
        read_split(data_dir, "valid", source_lang, target_lang, config["max_valid_pairs"]),
        "word",
        max_len,
    )
    test_pairs = filter_pairs(
        read_split(data_dir, "test", source_lang, target_lang, config["max_test_pairs"]),
        "word",
        max_len,
    )
    if not train_pairs or not valid_pairs or not test_pairs:
        raise ValueError(f"{name} has an empty train, validation, or test split")

    src_vocab = Vocab(
        [src for src, _ in train_pairs], "word", config["min_freq"], config["max_vocab_size"]
    )
    trg_vocab = Vocab(
        [trg for _, trg in train_pairs], "word", config["min_freq"], config["max_vocab_size"]
    )
    train_loader, valid_loader, test_loader, trg_pad_idx = make_loaders(
        train_pairs, valid_pairs, test_pairs, src_vocab, trg_vocab, "word", config
    )

    encoder = AttentionEncoder(
        len(src_vocab),
        config["embedding_dim"],
        config["hidden_dim"],
        config["dropout"],
        src_vocab.stoi[PAD],
        config["bidirectional_encoder"],
    )
    decoder = AttentionDecoder(
        len(trg_vocab),
        config["embedding_dim"],
        config["hidden_dim"],
        config["dropout"],
        trg_vocab.stoi[PAD],
        encoder.output_dim,
    )
    model = Seq2SeqAttention(encoder, decoder, trg_pad_idx, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=trg_pad_idx,
        label_smoothing=config.get("label_smoothing", 0),
    )

    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    bad_epochs = 0

    print(f"Starting Task 5 component: {name}")
    print(f"Train pairs used: {len(train_pairs)}")
    print(f"Valid pairs used: {len(valid_pairs)}")
    print(f"Test pairs used: {len(test_pairs)}")

    for epoch in range(1, config["epochs"] + 1):
        teacher_forcing = get_teacher_forcing(config, epoch)
        train_loss = train_one_epoch_attention(
            model, train_loader, optimizer, criterion, teacher_forcing, device
        )
        valid_loss = evaluate_loss_attention(model, valid_loader, criterion, device)
        scheduler.step(valid_loss)
        history.append((epoch, round(train_loss, 4), round(valid_loss, 4)))

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(
            f"{name} epoch {epoch}: train={train_loss:.4f}, "
            f"valid={valid_loss:.4f}, tf={teacher_forcing:.2f}"
        )
        if bad_epochs >= config["early_stop_patience"]:
            print(f"Stopping {name} early. Best epoch: {best_epoch}")
            break

    model.load_state_dict(best_state)
    references, predictions, prediction_rows, _ = run_predictions_with_attention(
        model, test_loader, src_vocab, trg_vocab, "word", config, device
    )

    component_dir = resolve_path(config["output_dir"]) / name
    component_dir.mkdir(parents=True, exist_ok=True)
    save_csv(history, component_dir / "loss_history.csv", ["epoch", "train_loss", "valid_loss"])
    save_csv(
        [[row["source"], row["reference"], row["prediction"]] for row in prediction_rows[:100]],
        component_dir / "sample_predictions.csv",
        ["source", "reference", "prediction"],
    )
    torch.save(
        {
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "source_vocab": src_vocab.itos,
            "target_vocab": trg_vocab.itos,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "model_config": {
                key: config[key]
                for key in (
                    "embedding_dim",
                    "hidden_dim",
                    "dropout",
                    "bidirectional_encoder",
                    "max_len_words",
                )
            },
        },
        component_dir / "model.pt",
    )

    result = {
        "experiment": name,
        "source": source_lang,
        "target": target_lang,
        "train_pairs": len(train_pairs),
        "valid_pairs": len(valid_pairs),
        "test_pairs": len(test_pairs),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_valid_loss": round(best_loss, 4),
        "bleu": simple_bleu(references, predictions),
        "token_accuracy": token_accuracy(references, predictions),
    }
    return {
        "model": model,
        "src_vocab": src_vocab,
        "trg_vocab": trg_vocab,
        "test_pairs": test_pairs,
        "result": result,
    }


def main():
    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config["output_dir"] = os.environ.get("TASK5_OUTPUT_DIR", config["output_dir"])
    set_seed(config["random_seed"])
    device = get_device(config)
    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")

    fr_en = train_component(
        "fr_en_attention",
        resolve_path(config["fr_en_processed_dir"]),
        "fr",
        "en",
        config,
        device,
    )
    en_sv = train_component(
        "en_sv_attention",
        resolve_path(config["task5_processed_dir"]),
        "en",
        "sv",
        config,
        device,
    )

    result_header = list(fr_en["result"])
    save_csv(
        [[bundle["result"][key] for key in result_header] for bundle in (fr_en, en_sv)],
        output_dir / "component_results.csv",
        result_header,
    )

    pivot_rows = build_pivot_rows(
        fr_en,
        en_sv,
        config["pivot_samples"],
        config["max_len_words"],
        device,
    )
    pivot_header = list(pivot_rows[0]) if pivot_rows else [
        "french_source",
        "english_reference",
        "english_prediction",
        "swedish_from_reference_english",
        "swedish_from_predicted_english",
    ]
    save_csv(
        [[row[key] for key in pivot_header] for row in pivot_rows],
        output_dir / "pivot_predictions.csv",
        pivot_header,
    )

    with (output_dir / "task5_notes.txt").open("w", encoding="utf-8") as f:
        f.write("Task 5 uses cascaded pivot translation: French -> English -> Swedish.\n")
        f.write("Component BLEU scores use each component's own parallel test set.\n")
        f.write("No direct French-Swedish BLEU is reported because no aligned reference is available.\n")
    print(f"Done. Task 5 outputs saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
