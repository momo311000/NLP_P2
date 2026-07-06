import csv
import json
import math
import os
import random
import string
import copy
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


CONFIG_FILE = "task3_config.json"

PAD = "<pad>"
UNK = "<unk>"
SOS = "<sos>"
EOS = "<eos>"

SPECIAL_TOKENS = [PAD, UNK, SOS, EOS]


def load_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_path(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config):
    if config["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config["device"])


def save_csv(rows, path, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def setting(exp, config, name):
    if name in exp:
        return exp[name]
    return config[name]


def read_split(data_dir, split_name, src_lang, trg_lang, max_pairs):
    src_path = data_dir / f"{split_name}.{src_lang}"
    trg_path = data_dir / f"{split_name}.{trg_lang}"
    pairs = []

    with src_path.open("r", encoding="utf-8") as src_file:
        with trg_path.open("r", encoding="utf-8") as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                pairs.append((src_line.strip(), trg_line.strip()))
                if max_pairs and len(pairs) >= max_pairs:
                    break

    return pairs


def tokenize(text, level):
    if level == "char":
        return list(text)
    return text.split()


def filter_pairs(pairs, level, max_len):
    kept = []
    for src, trg in pairs:
        if len(tokenize(src, level)) <= max_len and len(tokenize(trg, level)) <= max_len:
            kept.append((src, trg))
    return kept


class Vocab:
    def __init__(self, texts, level, min_freq, max_size):
        counts = Counter()
        for text in texts:
            counts.update(tokenize(text, level))

        self.itos = SPECIAL_TOKENS[:]
        for token, count in counts.most_common():
            if count < min_freq:
                continue
            if token in self.itos:
                continue
            if len(self.itos) >= max_size:
                break
            self.itos.append(token)

        self.stoi = {token: i for i, token in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, text, level, add_sos=False, add_eos=True):
        ids = []
        if add_sos:
            ids.append(self.stoi[SOS])

        for token in tokenize(text, level):
            ids.append(self.stoi.get(token, self.stoi[UNK]))

        if add_eos:
            ids.append(self.stoi[EOS])
        return ids

    def decode(self, ids):
        tokens = []
        for idx in ids:
            token = self.itos[int(idx)]
            if token == EOS:
                break
            if token not in SPECIAL_TOKENS:
                tokens.append(token)
        return tokens


class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, trg_vocab, level):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.level = level

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        src, trg = self.pairs[index]
        src_ids = self.src_vocab.encode(src, self.level, add_sos=False, add_eos=True)
        trg_ids = self.trg_vocab.encode(trg, self.level, add_sos=True, add_eos=True)
        return torch.tensor(src_ids), torch.tensor(trg_ids), src, trg


def make_collate(pad_idx):
    def collate(batch):
        batch = sorted(batch, key=lambda item: len(item[0]), reverse=True)
        src_batch = [item[0] for item in batch]
        trg_batch = [item[1] for item in batch]
        raw_src = [item[2] for item in batch]
        raw_trg = [item[3] for item in batch]
        src_lens = torch.tensor([len(x) for x in src_batch])
        trg_lens = torch.tensor([len(x) for x in trg_batch])

        src_pad = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
        trg_pad = pad_sequence(trg_batch, batch_first=True, padding_value=pad_idx)
        return src_pad, src_lens, trg_pad, trg_lens, raw_src, raw_trg

    return collate


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout, pad_idx, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(
            emb_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            self.bridge = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.bridge = None

    def forward(self, src, src_lens):
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(
            embedded,
            src_lens.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        output, hidden = self.rnn(packed)
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            hidden = torch.tanh(self.bridge(hidden)).unsqueeze(0)
        return hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token, hidden):
        token = token.unsqueeze(1)
        embedded = self.dropout(self.embedding(token))
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, trg_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def forward(self, src, src_lens, trg, teacher_forcing):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        hidden = self.encoder(src, src_lens)
        input_token = trg[:, 0]

        for t in range(1, trg_len):
            prediction, hidden = self.decoder(input_token, hidden)
            outputs[:, t] = prediction
            best_guess = prediction.argmax(1)
            use_teacher = random.random() < teacher_forcing
            input_token = trg[:, t] if use_teacher else best_guess

        return outputs

    def translate(self, src, src_len, sos_idx, eos_idx, max_len):
        self.eval()
        with torch.no_grad():
            hidden = self.encoder(src, src_len)
            input_token = torch.tensor([sos_idx], device=self.device)
            output_ids = []

            for _ in range(max_len):
                prediction, hidden = self.decoder(input_token, hidden)
                next_id = int(prediction.argmax(1).item())
                if next_id == eos_idx:
                    break
                output_ids.append(next_id)
                input_token = torch.tensor([next_id], device=self.device)

        return output_ids


class CbowModel(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.output = nn.Linear(emb_dim, vocab_size)

    def forward(self, context):
        emb = self.embedding(context)
        avg_emb = emb.mean(dim=1)
        return self.output(avg_emb)


def make_cbow_examples(texts, vocab, level, window, limit):
    examples = []
    for text in texts[:limit]:
        ids = [vocab.stoi.get(tok, vocab.stoi[UNK]) for tok in tokenize(text, level)]
        if len(ids) < 3:
            continue

        for i in range(len(ids)):
            context = []
            for j in range(i - window, i + window + 1):
                if j != i and 0 <= j < len(ids):
                    context.append(ids[j])

            if context:
                while len(context) < window * 2:
                    context.append(vocab.stoi[PAD])
                examples.append((context[:window * 2], ids[i]))

    return examples


def train_cbow_embeddings(texts, vocab, level, config, device):
    if level != "word":
        return None

    examples = make_cbow_examples(
        texts,
        vocab,
        level,
        config["word2vec_window"],
        config["word2vec_sentences"],
    )
    if not examples:
        return None

    model = CbowModel(len(vocab), config["embedding_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    batch_size = config["word2vec_batch_size"]

    for epoch in range(config["word2vec_epochs"]):
        random.shuffle(examples)
        total_loss = 0

        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            context = torch.tensor([item[0] for item in batch], device=device)
            target = torch.tensor([item[1] for item in batch], device=device)

            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"word2vec epoch {epoch + 1}: loss={total_loss / max(1, len(examples)):.4f}")

    return model.embedding.weight.detach().cpu()


def load_pretrained_embedding(vocab, embedding_file, emb_dim):
    path = get_path(embedding_file) if embedding_file else None
    if path is None or not path.exists():
        return None, 0

    weights = torch.randn(len(vocab), emb_dim) * 0.1
    weights[vocab.stoi[PAD]].zero_()
    found = 0

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) != emb_dim + 1:
                continue

            word = parts[0]
            if word not in vocab.stoi:
                continue

            try:
                vector = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float)
            except ValueError:
                continue

            weights[vocab.stoi[word]] = vector
            found += 1

    return weights, found


def apply_embedding_choice(exp, config, model, src_vocab, trg_vocab, train_pairs, device):
    emb_type = exp["embedding_type"]
    info = {"src_pretrained_found": 0, "trg_pretrained_found": 0, "word2vec_used": False}

    if emb_type == "frozen_random":
        model.encoder.embedding.weight.requires_grad = False
        model.decoder.embedding.weight.requires_grad = False

    if emb_type == "pretrained":
        src_file = config["embedding_files"].get(exp["source_lang"], "")
        trg_file = config["embedding_files"].get(exp["target_lang"], "")
        emb_dim = setting(exp, config, "embedding_dim")

        src_weights, src_found = load_pretrained_embedding(src_vocab, src_file, emb_dim)
        trg_weights, trg_found = load_pretrained_embedding(trg_vocab, trg_file, emb_dim)

        if src_weights is not None:
            model.encoder.embedding.weight.data.copy_(src_weights)
            info["src_pretrained_found"] = src_found
        if trg_weights is not None:
            model.decoder.embedding.weight.data.copy_(trg_weights)
            info["trg_pretrained_found"] = trg_found

        if src_weights is None or trg_weights is None:
            print("Pretrained embedding file missing, so this experiment keeps random embeddings.")

    if emb_type == "word2vec":
        print("Training simple Word2Vec/CBOW embeddings for this experiment...")
        src_texts = [src for src, trg in train_pairs]
        trg_texts = [trg for src, trg in train_pairs]

        src_weights = train_cbow_embeddings(src_texts, src_vocab, exp["level"], config, device)
        trg_weights = train_cbow_embeddings(trg_texts, trg_vocab, exp["level"], config, device)

        if src_weights is not None:
            model.encoder.embedding.weight.data.copy_(src_weights.to(model.encoder.embedding.weight.device))
        if trg_weights is not None:
            model.decoder.embedding.weight.data.copy_(trg_weights.to(model.decoder.embedding.weight.device))
        info["word2vec_used"] = src_weights is not None and trg_weights is not None

    return info


def get_teacher_forcing(config, epoch):
    start = config.get("teacher_forcing_start", config["teacher_forcing"])
    end = config.get("teacher_forcing_end", config["teacher_forcing"])
    epochs = max(config["epochs"] - 1, 1)
    progress = min((epoch - 1) / epochs, 1)
    return start + (end - start) * progress


def train_one_epoch(model, loader, optimizer, criterion, teacher_forcing, device):
    model.train()
    total_loss = 0

    for src, src_lens, trg, trg_lens, raw_src, raw_trg in loader:
        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()
        output = model(src, src_lens, trg, teacher_forcing)

        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, src_lens, trg, trg_lens, raw_src, raw_trg in loader:
            src = src.to(device)
            trg = trg.to(device)

            output = model(src, src_lens, trg, teacher_forcing=0)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg)
            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def ngram_counts(tokens, n):
    counts = Counter()
    for i in range(len(tokens) - n + 1):
        counts[tuple(tokens[i:i + n])] += 1
    return counts


def simple_bleu(references, predictions, max_n=4):
    matches = [0] * max_n
    totals = [0] * max_n
    ref_len = 0
    pred_len = 0

    for ref, pred in zip(references, predictions):
        ref_len += len(ref)
        pred_len += len(pred)

        for n in range(1, max_n + 1):
            ref_counts = ngram_counts(ref, n)
            pred_counts = ngram_counts(pred, n)
            totals[n - 1] += max(sum(pred_counts.values()), 1)

            for gram, count in pred_counts.items():
                matches[n - 1] += min(count, ref_counts.get(gram, 0))

    precisions = []
    for match, total in zip(matches, totals):
        precisions.append((match + 1) / (total + 1))

    if pred_len == 0:
        return 0.0

    brevity = 1.0
    if pred_len < ref_len:
        brevity = math.exp(1 - ref_len / pred_len)

    score = brevity * math.exp(sum(math.log(p) for p in precisions) / max_n)
    return round(score * 100, 2)


def token_accuracy(references, predictions):
    correct = 0
    total = 0

    for ref, pred in zip(references, predictions):
        max_len = max(len(ref), len(pred))
        total += max_len
        for i in range(max_len):
            ref_token = ref[i] if i < len(ref) else None
            pred_token = pred[i] if i < len(pred) else None
            if ref_token == pred_token:
                correct += 1

    if total == 0:
        return 0.0
    return round(correct / total * 100, 2)


def one_sentence_accuracy(ref, pred):
    max_len = max(len(ref), len(pred))
    if max_len == 0:
        return 0.0
    correct = 0
    for i in range(max_len):
        ref_token = ref[i] if i < len(ref) else None
        pred_token = pred[i] if i < len(pred) else None
        if ref_token == pred_token:
            correct += 1
    return round(correct / max_len * 100, 2)


def length_group(length):
    if length <= 10:
        return "short"
    if length <= 20:
        return "medium"
    return "long"


def sentence_features(text, level):
    tokens = tokenize(text, level)
    return {
        "source_len": len(tokens),
        "length_group": length_group(len(tokens)),
        "has_number": any(char.isdigit() for char in text),
        "has_punctuation": any(char in string.punctuation for char in text),
    }


def summarise_group(rows, group_column):
    groups = {}
    for row in rows:
        group = row[group_column]
        groups.setdefault(group, []).append(row)

    summary = []
    for group, items in sorted(groups.items()):
        summary.append([
            group,
            len(items),
            round(sum(item["sentence_bleu"] for item in items) / len(items), 2),
            round(sum(item["sentence_accuracy"] for item in items) / len(items), 2),
            round(sum(item["source_len"] for item in items) / len(items), 2),
        ])
    return summary


def make_loaders(train_pairs, valid_pairs, test_pairs, src_vocab, trg_vocab, level, config):
    pad_idx = src_vocab.stoi[PAD]
    trg_pad_idx = trg_vocab.stoi[PAD]

    train_data = TranslationDataset(train_pairs, src_vocab, trg_vocab, level)
    valid_data = TranslationDataset(valid_pairs, src_vocab, trg_vocab, level)
    test_data = TranslationDataset(test_pairs, src_vocab, trg_vocab, level)

    train_loader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=make_collate(pad_idx),
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=make_collate(pad_idx),
    )
    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        collate_fn=make_collate(pad_idx),
    )

    return train_loader, valid_loader, test_loader, trg_pad_idx


def run_predictions(model, loader, src_vocab, trg_vocab, level, config, device):
    predictions = []
    references = []
    rows = []

    sos_idx = trg_vocab.stoi[SOS]
    eos_idx = trg_vocab.stoi[EOS]
    max_len = config["max_len_chars"] if level == "char" else config["max_len_words"]

    for src, src_lens, trg, trg_lens, raw_src, raw_trg in loader:
        src = src.to(device)
        output_ids = model.translate(src, src_lens, sos_idx, eos_idx, max_len + 10)
        pred_tokens = trg_vocab.decode(output_ids)
        ref_tokens = tokenize(raw_trg[0], level)

        predictions.append(pred_tokens)
        references.append(ref_tokens)

        if level == "char":
            pred_text = "".join(pred_tokens)
        else:
            pred_text = " ".join(pred_tokens)

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

    return references, predictions, rows


def save_loss_plot(history, path):
    plt.figure(figsize=(7, 4.5))
    plt.plot(history["epoch"], history["train_loss"], marker="o", label="train")
    plt.plot(history["epoch"], history["valid_loss"], marker="o", label="valid")
    plt.title("Training and validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_prediction_files(prediction_rows, output_dir):
    header = [
        "source",
        "reference",
        "prediction",
        "source_len",
        "length_group",
        "has_number",
        "has_punctuation",
        "sentence_bleu",
        "sentence_accuracy",
    ]
    rows = [[row[col] for col in header] for row in prediction_rows]

    sample_rows = rows[:100]
    save_csv(sample_rows, output_dir / "sample_predictions.csv", header)


def save_analysis_files(prediction_rows, output_dir):
    length_rows = summarise_group(prediction_rows, "length_group")
    save_csv(
        length_rows,
        output_dir / "length_analysis.csv",
        ["length_group", "sentences", "avg_bleu", "avg_token_accuracy", "avg_source_len"],
    )

    feature_rows = []
    for feature in ["has_number", "has_punctuation"]:
        for row in summarise_group(prediction_rows, feature):
            feature_rows.append([feature] + row)

    save_csv(
        feature_rows,
        output_dir / "sentence_feature_analysis.csv",
        ["feature", "value", "sentences", "avg_bleu", "avg_token_accuracy", "avg_source_len"],
    )


def save_overall_plots(results, output_dir):
    names = [row["experiment"] for row in results]
    bleu_scores = [row["bleu"] for row in results]
    acc_scores = [row["token_accuracy"] for row in results]

    plt.figure(figsize=(max(8, len(names) * 1.5), 5))
    plt.bar(names, bleu_scores)
    plt.title("BLEU comparison")
    plt.xlabel("Experiment")
    plt.ylabel("BLEU")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "bleu_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(max(8, len(names) * 1.5), 5))
    plt.bar(names, acc_scores)
    plt.title("Token accuracy comparison")
    plt.xlabel("Experiment")
    plt.ylabel("Token accuracy")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "token_accuracy_comparison.png", dpi=160)
    plt.close()


def save_tuning_summary(results, output_dir):
    if not results:
        return

    best_valid = min(results, key=lambda row: row["final_valid_loss"])
    best_bleu = max(results, key=lambda row: row["bleu"])

    rows = [
        ["best valid loss model", best_valid["experiment"]],
        ["best valid loss", best_valid["final_valid_loss"]],
        ["hidden_dim used", best_valid["hidden_dim"]],
        ["learning_rate used", best_valid["learning_rate"]],
        ["dropout used", best_valid["dropout"]],
        ["best BLEU model", best_bleu["experiment"]],
        ["best BLEU score", best_bleu["bleu"]],
    ]
    save_csv(rows, output_dir / "hyperparameter_tuning_summary.csv", ["item", "value"])


def save_required_tables(results, output_dir):
    embedding_rows = []
    direction_rows = []
    char_rows = []

    for row in results:
        base = [
            row["experiment"],
            row["source"],
            row["target"],
            row["level"],
            row["embedding_type"],
            row["final_valid_loss"],
            row["bleu"],
            row["token_accuracy"],
        ]

        if row["source"] == "en" and row["target"] == "fr" and row["level"] == "word":
            embedding_rows.append(base)

        if row["level"] == "word" and row["embedding_type"] == "random":
            direction_rows.append(base)

        if row["source"] == "en" and row["target"] == "fr":
            char_rows.append(base)

    header = [
        "experiment",
        "source",
        "target",
        "level",
        "embedding_type",
        "valid_loss",
        "bleu",
        "token_accuracy",
    ]
    save_csv(embedding_rows, output_dir / "embedding_comparison.csv", header)
    save_csv(direction_rows, output_dir / "direction_comparison.csv", header)
    save_csv(char_rows, output_dir / "character_vs_word_comparison.csv", header)


def run_experiment(exp, config, device):
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

    src_vocab = Vocab(
        [src for src, trg in train_pairs],
        level,
        exp_config["min_freq"],
        exp_config["max_vocab_size"],
    )
    trg_vocab = Vocab(
        [trg for src, trg in train_pairs],
        level,
        exp_config["min_freq"],
        exp_config["max_vocab_size"],
    )

    train_loader, valid_loader, test_loader, trg_pad_idx = make_loaders(
        train_pairs,
        valid_pairs,
        test_pairs,
        src_vocab,
        trg_vocab,
        level,
        exp_config,
    )

    encoder = Encoder(
        len(src_vocab),
        exp_config["embedding_dim"],
        exp_config["hidden_dim"],
        exp_config["dropout"],
        src_vocab.stoi[PAD],
        exp_config.get("bidirectional_encoder", False),
    )
    decoder = Decoder(
        len(trg_vocab),
        exp_config["embedding_dim"],
        exp_config["hidden_dim"],
        exp_config["dropout"],
        trg_vocab.stoi[PAD],
    )
    model = Seq2Seq(encoder, decoder, trg_pad_idx, device).to(device)

    embedding_info = apply_embedding_choice(exp, exp_config, model, src_vocab, trg_vocab, train_pairs, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=exp_config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=trg_pad_idx,
        label_smoothing=exp_config.get("label_smoothing", 0),
    )

    history = {"epoch": [], "train_loss": [], "valid_loss": []}
    best_valid_loss = float("inf")
    best_epoch = 0
    best_state = None
    bad_epochs = 0

    print(f"Starting experiment: {exp['name']}")
    print(f"Train pairs used: {len(train_pairs)}")
    print(f"Valid pairs used: {len(valid_pairs)}")
    print(f"Test pairs used: {len(test_pairs)}")
    print(f"Source vocab: {len(src_vocab)}, Target vocab: {len(trg_vocab)}")

    for epoch in range(1, exp_config["epochs"] + 1):
        teacher_forcing = get_teacher_forcing(exp_config, epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, teacher_forcing, device)
        valid_loss = evaluate_loss(model, valid_loader, criterion, device)
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

    references, predictions, prediction_rows = run_predictions(
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

    return {
        "experiment": exp["name"],
        "source": exp["source_lang"],
        "target": exp["target_lang"],
        "level": level,
        "embedding_type": exp["embedding_type"],
        "embedding_dim": exp_config["embedding_dim"],
        "hidden_dim": exp_config["hidden_dim"],
        "bidirectional_encoder": exp_config.get("bidirectional_encoder", False),
        "dropout": exp_config["dropout"],
        "learning_rate": exp_config["learning_rate"],
        "epochs": exp_config["epochs"],
        "best_epoch": best_epoch,
        "teacher_forcing": exp_config["teacher_forcing"],
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


def main():
    config = load_config()
    set_seed(config["random_seed"])
    device = get_device(config)

    output_dir = get_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")

    all_results = []
    for exp in config["experiments"]:
        if exp.get("run", True) is False:
            print(f"Skipping experiment: {exp['name']}")
            continue
        result = run_experiment(exp, config, device)
        all_results.append(result)

    rows = []
    for result in all_results:
        rows.append([
            result["experiment"],
            result["source"],
            result["target"],
            result["level"],
            result["embedding_type"],
            result["embedding_dim"],
            result["hidden_dim"],
            result["bidirectional_encoder"],
            result["dropout"],
            result["learning_rate"],
            result["epochs"],
            result["best_epoch"],
            result["teacher_forcing"],
            result["teacher_forcing_start"],
            result["teacher_forcing_end"],
            result["train_pairs"],
            result["valid_pairs"],
            result["test_pairs"],
            result["source_vocab"],
            result["target_vocab"],
            result["src_pretrained_found"],
            result["trg_pretrained_found"],
            result["word2vec_used"],
            result["final_train_loss"],
            result["final_valid_loss"],
            result["bleu"],
            result["token_accuracy"],
        ])

    save_csv(
        rows,
        output_dir / "task3_results.csv",
        [
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
            "teacher_forcing",
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
        ],
    )
    save_overall_plots(all_results, output_dir)
    save_tuning_summary(all_results, output_dir)
    save_required_tables(all_results, output_dir)

    with (output_dir / "task3_notes.txt").open("w", encoding="utf-8") as f:
        f.write("Task 3 notes\n")
        f.write("============\n\n")
        f.write("I used a GRU encoder-decoder model for the translation task.\n")
        f.write("The encoder is bidirectional, but the decoder is still a normal GRU decoder without attention.\n")
        f.write("The two scores reported are BLEU and token accuracy.\n")
        f.write("I trained English-French, French-English, and character-level models.\n")
        f.write("For tuning, I changed the hidden size and learning rate in the config file.\n")
        f.write("For embeddings, I compared random and Word2Vec/CBOW embeddings.\n")
        f.write("The script keeps the model from the best validation epoch before testing.\n")
        f.write("The length and sentence feature csv files are used to discuss which sentences worked better.\n")
        f.write("Pretrained embeddings can be tried by adding file paths in task3_config.json.\n")

    print(f"Done. Task 3 outputs saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
