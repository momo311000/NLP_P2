import copy
import csv
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import tasks3
from tasks3 import (
    EOS,
    PAD,
    SOS,
    Encoder,
    Vocab,
    apply_embedding_choice,
    filter_pairs,
    get_device,
    get_path,
    get_teacher_forcing,
    make_loaders,
    one_sentence_accuracy,
    read_split,
    save_analysis_files,
    save_csv,
    save_loss_plot,
    save_prediction_files,
    sentence_features,
    set_seed,
    simple_bleu,
    tokenize,
    token_accuracy,
)


ROOT = Path(__file__).parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TASK4_OUTPUT_DIR = "outputs/task4"
TASK4_EXPERIMENT = {
    "name": "word_en_fr_attention",
    "source_lang": "en",
    "target_lang": "fr",
    "level": "word",
    "embedding_type": "random",
    "attention": True,
}


class AttentionEncoder(Encoder):
    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout, pad_idx, bidirectional=True):
        super().__init__(vocab_size, emb_dim, hidden_dim, dropout, pad_idx, bidirectional)
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim

    def forward(self, src, src_lens):
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(
            embedded,
            src_lens.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, hidden = self.rnn(packed)
        encoder_outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=src.size(1),
        )

        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            hidden = torch.tanh(self.bridge(hidden)).unsqueeze(0)

        return encoder_outputs, hidden


class Attention(nn.Module):
    def __init__(self, encoder_output_dim, hidden_dim):
        super().__init__()
        self.encoder_projection = nn.Linear(encoder_output_dim, hidden_dim, bias=False)

    def forward(self, hidden, encoder_outputs, src_lens):
        projected_outputs = self.encoder_projection(encoder_outputs)
        query = hidden[-1].unsqueeze(2)
        scores = torch.bmm(projected_outputs, query).squeeze(2)

        max_len = encoder_outputs.size(1)
        positions = torch.arange(max_len, device=encoder_outputs.device).unsqueeze(0)
        mask = positions >= src_lens.to(encoder_outputs.device).unsqueeze(1)
        scores = scores.masked_fill(mask, -1e9)
        return torch.softmax(scores, dim=1)


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout, pad_idx, encoder_output_dim=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.hidden_dim = hidden_dim
        self.encoder_output_dim = encoder_output_dim or hidden_dim * 2
        self.attention = Attention(self.encoder_output_dim, hidden_dim)
        self.rnn = nn.GRU(emb_dim + self.encoder_output_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim + self.encoder_output_dim + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token, hidden, encoder_outputs, src_lens):
        token = token.unsqueeze(1)
        embedded = self.dropout(self.embedding(token))
        attn_weights = self.attention(hidden, encoder_outputs, src_lens)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        prediction_input = torch.cat(
            (output.squeeze(1), context.squeeze(1), embedded.squeeze(1)),
            dim=1,
        )
        prediction = self.fc(prediction_input)
        return prediction, hidden, attn_weights


class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder, trg_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def forward(self, src, src_lens, trg, teacher_forcing):
        batch_size = src.size(0)
        src_len = src.size(1)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size, device=self.device)
        attentions = torch.zeros(batch_size, trg_len, src_len, device=self.device)

        encoder_outputs, hidden = self.encoder(src, src_lens)
        input_token = trg[:, 0]

        for t in range(1, trg_len):
            prediction, hidden, attn_weights = self.decoder(input_token, hidden, encoder_outputs, src_lens)
            outputs[:, t] = prediction
            attentions[:, t] = attn_weights
            best_guess = prediction.argmax(1)
            use_teacher = random.random() < teacher_forcing
            input_token = trg[:, t] if use_teacher else best_guess

        return outputs, attentions

    def translate(self, src, src_len, sos_idx, eos_idx, max_len, return_attention=False):
        self.eval()
        with torch.no_grad():
            encoder_outputs, hidden = self.encoder(src, src_len)
            input_token = torch.tensor([sos_idx], device=self.device)
            output_ids = []
            attention_rows = []

            for _ in range(max_len):
                prediction, hidden, attn_weights = self.decoder(input_token, hidden, encoder_outputs, src_len)
                next_id = int(prediction.argmax(1).item())
                if next_id == eos_idx:
                    break
                output_ids.append(next_id)
                attention_rows.append(attn_weights.squeeze(0).detach().cpu())
                input_token = torch.tensor([next_id], device=self.device)

        if return_attention:
            if attention_rows:
                return output_ids, torch.stack(attention_rows)
            return output_ids, torch.zeros(0, src.size(1))
        return output_ids


def train_one_epoch_attention(model, loader, optimizer, criterion, teacher_forcing, device):
    model.train()
    total_loss = 0

    for src, src_lens, trg, trg_lens, raw_src, raw_trg in loader:
        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()
        output, _ = model(src, src_lens, trg, teacher_forcing)

        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def evaluate_loss_attention(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, src_lens, trg, trg_lens, raw_src, raw_trg in loader:
            src = src.to(device)
            trg = trg.to(device)

            output, _ = model(src, src_lens, trg, teacher_forcing=0)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg)
            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def run_predictions_with_attention(model, loader, src_vocab, trg_vocab, level, config, device):
    predictions = []
    references = []
    rows = []
    attention_example = None

    sos_idx = trg_vocab.stoi[SOS]
    eos_idx = trg_vocab.stoi[EOS]
    max_len = config["max_len_chars"] if level == "char" else config["max_len_words"]

    for src, src_lens, trg, trg_lens, raw_src, raw_trg in loader:
        src = src.to(device)
        output_ids, attention_matrix = model.translate(
            src,
            src_lens,
            sos_idx,
            eos_idx,
            max_len + 10,
            return_attention=True,
        )
        pred_tokens = trg_vocab.decode(output_ids)
        ref_tokens = tokenize(raw_trg[0], level)

        predictions.append(pred_tokens)
        references.append(ref_tokens)

        pred_text = "".join(pred_tokens) if level == "char" else " ".join(pred_tokens)
        features = sentence_features(raw_src[0], level)
        sent_bleu = simple_bleu([ref_tokens], [pred_tokens])
        sent_acc = one_sentence_accuracy(ref_tokens, pred_tokens)

        rows.append({
            "source": raw_src[0],
            "reference": raw_trg[0],
            "prediction": pred_text,
            "source_len": features["source_len"],
            "length_group": features["length_group"],
            "has_number": features["has_number"],
            "has_punctuation": features["has_punctuation"],
            "sentence_bleu": sent_bleu,
            "sentence_accuracy": sent_acc,
        })

        if attention_example is None and len(pred_tokens) > 0:
            source_tokens = tokenize(raw_src[0], level)
            attention_example = {
                "source": raw_src[0],
                "reference": raw_trg[0],
                "prediction": pred_text,
                "source_tokens": source_tokens,
                "target_tokens": pred_tokens,
                "attention": attention_matrix[:, :len(source_tokens)].numpy(),
            }

    return references, predictions, rows, attention_example


def save_attention_files(attention_example, output_dir):
    if attention_example is None:
        return

    source_tokens = attention_example["source_tokens"]
    target_tokens = attention_example["target_tokens"]
    attention = attention_example["attention"]

    rows = []
    for target_token, weights in zip(target_tokens, attention):
        rows.append([target_token] + [round(float(weight), 6) for weight in weights])
    save_csv(rows, output_dir / "attention_weights.csv", ["target_token"] + source_tokens)

    plt.figure(figsize=(max(7, len(source_tokens) * 0.45), max(4, len(target_tokens) * 0.35)))
    plt.imshow(attention, aspect="auto", cmap="viridis")
    plt.colorbar(label="attention")
    plt.xticks(range(len(source_tokens)), source_tokens, rotation=45, ha="right")
    plt.yticks(range(len(target_tokens)), target_tokens)
    plt.xlabel("Source tokens")
    plt.ylabel("Generated target tokens")
    plt.title("Attention weights for one sample")
    plt.tight_layout()
    plt.savefig(output_dir / "attention_heatmap.png", dpi=180)
    plt.close()

    save_csv(
        [[attention_example["source"], attention_example["reference"], attention_example["prediction"]]],
        output_dir / "attention_sample.csv",
        ["source", "reference", "prediction"],
    )


def read_task3_baseline(task3_results_path, baseline_name="word_en_fr_random"):
    if not task3_results_path.exists():
        return None

    with task3_results_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("experiment") == baseline_name:
                return row
    return None


def save_attention_comparison(result, output_dir):
    baseline = read_task3_baseline(ROOT / "outputs" / "task3" / "task3_results.csv")
    rows = []

    if baseline is not None:
        rows.append([
            baseline["experiment"],
            "without_attention",
            baseline.get("final_valid_loss", ""),
            baseline.get("bleu", ""),
            baseline.get("token_accuracy", ""),
        ])

    rows.append([
        result["experiment"],
        "with_attention",
        result["final_valid_loss"],
        result["bleu"],
        result["token_accuracy"],
    ])

    save_csv(
        rows,
        output_dir / "attention_comparison.csv",
        ["experiment", "model_type", "valid_loss", "bleu", "token_accuracy"],
    )


def run_attention_experiment(exp, config, device):
    data_dir = get_path(config["processed_dir"])
    output_dir = get_path(config["output_dir"]) / exp["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_config = config.copy()
    for key in [
        "batch_size",
        "epochs",
        "learning_rate",
        "embedding_dim",
        "hidden_dim",
        "dropout",
        "teacher_forcing",
        "teacher_forcing_start",
        "teacher_forcing_end",
        "label_smoothing",
        "early_stop_patience",
        "bidirectional_encoder",
        "min_freq",
        "max_vocab_size",
    ]:
        if key in exp:
            exp_config[key] = exp[key]

    level = exp["level"]
    max_len = exp_config["max_len_chars"] if level == "char" else exp_config["max_len_words"]

    train_pairs = read_split(data_dir, "train", exp["source_lang"], exp["target_lang"], exp_config["max_train_pairs"])
    valid_pairs = read_split(data_dir, "valid", exp["source_lang"], exp["target_lang"], exp_config["max_valid_pairs"])
    test_pairs = read_split(data_dir, "test", exp["source_lang"], exp["target_lang"], exp_config["max_test_pairs"])

    train_pairs = filter_pairs(train_pairs, level, max_len)
    valid_pairs = filter_pairs(valid_pairs, level, max_len)
    test_pairs = filter_pairs(test_pairs, level, max_len)

    src_vocab = Vocab([src for src, trg in train_pairs], level, exp_config["min_freq"], exp_config["max_vocab_size"])
    trg_vocab = Vocab([trg for src, trg in train_pairs], level, exp_config["min_freq"], exp_config["max_vocab_size"])

    train_loader, valid_loader, test_loader, trg_pad_idx = make_loaders(
        train_pairs,
        valid_pairs,
        test_pairs,
        src_vocab,
        trg_vocab,
        level,
        exp_config,
    )

    encoder = AttentionEncoder(
        len(src_vocab),
        exp_config["embedding_dim"],
        exp_config["hidden_dim"],
        exp_config["dropout"],
        src_vocab.stoi[PAD],
        exp_config.get("bidirectional_encoder", True),
    )
    decoder = AttentionDecoder(
        len(trg_vocab),
        exp_config["embedding_dim"],
        exp_config["hidden_dim"],
        exp_config["dropout"],
        trg_vocab.stoi[PAD],
        encoder.output_dim,
    )
    model = Seq2SeqAttention(encoder, decoder, trg_pad_idx, device).to(device)

    embedding_info = apply_embedding_choice(exp, exp_config, model, src_vocab, trg_vocab, train_pairs, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=exp_config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    criterion = nn.CrossEntropyLoss(
        ignore_index=trg_pad_idx,
        label_smoothing=exp_config.get("label_smoothing", 0),
    )

    history = {"epoch": [], "train_loss": [], "valid_loss": []}
    best_valid_loss = float("inf")
    best_epoch = 0
    best_state = None
    bad_epochs = 0

    print(f"Starting Task 4 experiment: {exp['name']}")
    print(f"Train pairs used: {len(train_pairs)}")
    print(f"Valid pairs used: {len(valid_pairs)}")
    print(f"Test pairs used: {len(test_pairs)}")
    print(f"Source vocab: {len(src_vocab)}, Target vocab: {len(trg_vocab)}")

    for epoch in range(1, exp_config["epochs"] + 1):
        teacher_forcing = get_teacher_forcing(exp_config, epoch)
        train_loss = train_one_epoch_attention(model, train_loader, optimizer, criterion, teacher_forcing, device)
        valid_loss = evaluate_loss_attention(model, valid_loader, criterion, device)
        scheduler.step(valid_loss)

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["valid_loss"].append(round(valid_loss, 4))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(
            f"{exp['name']} epoch {epoch}: train={train_loss:.4f}, "
            f"valid={valid_loss:.4f}, tf={teacher_forcing:.2f}"
        )

        if bad_epochs >= exp_config.get("early_stop_patience", 99):
            print(f"Stopping {exp['name']} early. Best validation was at epoch {best_epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    references, predictions, prediction_rows, attention_example = run_predictions_with_attention(
        model,
        test_loader,
        src_vocab,
        trg_vocab,
        level,
        exp_config,
        device,
    )

    bleu = simple_bleu(references, predictions)
    acc = token_accuracy(references, predictions)

    save_csv(
        zip(history["epoch"], history["train_loss"], history["valid_loss"]),
        output_dir / "loss_history.csv",
        ["epoch", "train_loss", "valid_loss"],
    )
    save_prediction_files(prediction_rows, output_dir)
    save_analysis_files(prediction_rows, output_dir)
    save_loss_plot(history, output_dir / "loss_curve.png")
    save_attention_files(attention_example, output_dir)

    return {
        "experiment": exp["name"],
        "source": exp["source_lang"],
        "target": exp["target_lang"],
        "level": level,
        "embedding_type": exp["embedding_type"],
        "embedding_dim": exp_config["embedding_dim"],
        "hidden_dim": exp_config["hidden_dim"],
        "bidirectional_encoder": exp_config.get("bidirectional_encoder", True),
        "dropout": exp_config["dropout"],
        "learning_rate": exp_config["learning_rate"],
        "epochs": exp_config["epochs"],
        "best_epoch": best_epoch,
        "teacher_forcing_start": exp_config.get("teacher_forcing_start", exp_config["teacher_forcing"]),
        "teacher_forcing_end": exp_config.get("teacher_forcing_end", exp_config["teacher_forcing"]),
        "train_pairs": len(train_pairs),
        "valid_pairs": len(valid_pairs),
        "test_pairs": len(test_pairs),
        "source_vocab": len(src_vocab),
        "target_vocab": len(trg_vocab),
        "src_pretrained_found": embedding_info["src_pretrained_found"],
        "trg_pretrained_found": embedding_info["trg_pretrained_found"],
        "word2vec_used": embedding_info["word2vec_used"],
        "final_train_loss": history["train_loss"][-1],
        "final_valid_loss": round(best_valid_loss, 4),
        "bleu": bleu,
        "token_accuracy": acc,
    }


def save_task4_results(result, output_dir):
    header = [
        "experiment",
        "source",
        "target",
        "level",
        "embedding_type",
        "embedding_dim",
        "hidden_dim",
        "bidirectional_encoder",
        "dropout",
        "learning_rate",
        "epochs",
        "best_epoch",
        "teacher_forcing_start",
        "teacher_forcing_end",
        "train_pairs",
        "valid_pairs",
        "test_pairs",
        "source_vocab",
        "target_vocab",
        "src_pretrained_found",
        "trg_pretrained_found",
        "word2vec_used",
        "final_train_loss",
        "final_valid_loss",
        "bleu",
        "token_accuracy",
    ]
    save_csv([[result[col] for col in header]], output_dir / "task4_results.csv", header)


def main():
    config = tasks3.load_config()
    config["output_dir"] = TASK4_OUTPUT_DIR
    set_seed(config["random_seed"])
    device = get_device(config)

    output_dir = get_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    result = run_attention_experiment(TASK4_EXPERIMENT, config, device)
    save_task4_results(result, output_dir)
    save_attention_comparison(result, output_dir)

    with (output_dir / "task4_notes.txt").open("w", encoding="utf-8") as f:
        f.write("Task 4 notes\n")
        f.write("============\n\n")
        f.write("This task extends the Task 3 GRU encoder-decoder with attention.\n")
        f.write("The encoder returns all source hidden states instead of only the final state.\n")
        f.write("The decoder computes attention weights over source tokens at each generation step.\n")
        f.write("The attention weights are saved as a CSV matrix and a heatmap for one sample.\n")
        f.write("The comparison file uses the Task 3 word_en_fr_random baseline when available.\n")

    print(f"Done. Task 4 outputs saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
