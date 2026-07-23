import tempfile
import unittest
from pathlib import Path

import torch

import prepare_task5_data
import task5


class Task5PreprocessingTests(unittest.TestCase):
    def test_prepare_data_keeps_pairs_aligned_and_removes_invalid_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            en_path = root / "tiny.en"
            sv_path = root / "tiny.sv"
            en_path.write_text("HELLO WORLD\n<tag>\n\nGOOD NIGHT\nTHANK YOU\n", encoding="utf-8")
            sv_path.write_text("HEJ VARLD\nsvensk text\ntom\nGOD NATT\nTACK\n", encoding="utf-8")

            output_dir = root / "processed"
            summary = prepare_task5_data.prepare_data({
                "english_swedish_english_file": str(en_path),
                "english_swedish_swedish_file": str(sv_path),
                "task5_processed_dir": str(output_dir),
                "random_seed": 42,
                "sample_ratio": 1.0,
                "train_ratio": 0.5,
                "valid_ratio": 0.25,
                "test_ratio": 0.25,
            })

            self.assertEqual(summary["clean_pairs"], 3)
            self.assertEqual(summary["sampled_pairs"], 3)

            all_pairs = []
            for split in ("train", "valid", "test"):
                en_lines = (output_dir / f"{split}.en").read_text(encoding="utf-8").splitlines()
                sv_lines = (output_dir / f"{split}.sv").read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(en_lines), len(sv_lines))
                all_pairs.extend(zip(en_lines, sv_lines))

            self.assertCountEqual(all_pairs, [
                ("hello world", "hej varld"),
                ("good night", "god natt"),
                ("thank you", "tack"),
            ])


class FakeVocab:
    def __init__(self, tokens):
        self.itos = ["<pad>", "<sos>", "<eos>", "<unk>"] + tokens
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    def encode(self, text, level, add_sos=False, add_eos=True):
        ids = [self.stoi.get(token, self.stoi["<unk>"]) for token in text.split()]
        if add_eos:
            ids.append(self.stoi["<eos>"])
        return ids

    def decode(self, ids):
        return [self.itos[index] for index in ids if self.itos[index] not in {"<pad>", "<sos>", "<eos>", "<unk>"}]


class FakeModel:
    def __init__(self, outputs):
        self.outputs = outputs

    def translate(self, src, src_len, sos_idx, eos_idx, max_len, return_attention=False):
        return self.outputs[int(src[0, 0])]


class Task5PivotTests(unittest.TestCase):
    def test_build_pivot_rows_translates_reference_and_predicted_english(self):
        fr_vocab = FakeVocab(["bonjour"])
        en_target_vocab = FakeVocab(["hello", "hi"])
        en_source_vocab = FakeVocab(["hello", "hi"])
        sv_vocab = FakeVocab(["hej", "tjena"])

        fr_model = FakeModel({fr_vocab.stoi["bonjour"]: [en_target_vocab.stoi["hi"], 2]})
        sv_model = FakeModel({
            en_source_vocab.stoi["hello"]: [sv_vocab.stoi["hej"], 2],
            en_source_vocab.stoi["hi"]: [sv_vocab.stoi["tjena"], 2],
        })

        rows = task5.build_pivot_rows(
            {
                "model": fr_model,
                "src_vocab": fr_vocab,
                "trg_vocab": en_target_vocab,
                "test_pairs": [("bonjour", "hello")],
            },
            {
                "model": sv_model,
                "src_vocab": en_source_vocab,
                "trg_vocab": sv_vocab,
            },
            limit=1,
            max_len=5,
            device=torch.device("cpu"),
        )

        self.assertEqual(rows, [{
            "french_source": "bonjour",
            "english_reference": "hello",
            "english_prediction": "hi",
            "swedish_from_reference_english": "hej",
            "swedish_from_predicted_english": "tjena",
        }])

    def test_train_component_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            pairs = [
                ("hello world", "hej varld"),
                ("good night", "god natt"),
                ("thank you", "tack sa mycket"),
                ("yes please", "ja tack"),
            ]
            for split in ("train", "valid", "test"):
                (data_dir / f"{split}.en").write_text(
                    "\n".join(source for source, _ in pairs) + "\n", encoding="utf-8"
                )
                (data_dir / f"{split}.sv").write_text(
                    "\n".join(target for _, target in pairs) + "\n", encoding="utf-8"
                )

            config = {
                "output_dir": str(root / "output"),
                "max_train_pairs": 4,
                "max_valid_pairs": 4,
                "max_test_pairs": 4,
                "max_len_words": 10,
                "min_freq": 1,
                "max_vocab_size": 50,
                "batch_size": 2,
                "epochs": 1,
                "learning_rate": 0.001,
                "embedding_dim": 4,
                "hidden_dim": 6,
                "bidirectional_encoder": True,
                "dropout": 0.0,
                "teacher_forcing": 0.5,
                "teacher_forcing_start": 0.5,
                "teacher_forcing_end": 0.5,
                "label_smoothing": 0.0,
                "early_stop_patience": 1,
            }

            bundle = task5.train_component(
                "tiny_en_sv",
                data_dir,
                "en",
                "sv",
                config,
                torch.device("cpu"),
            )

            self.assertEqual(bundle["result"]["epochs_run"], 1)
            self.assertTrue((root / "output" / "tiny_en_sv" / "model.pt").exists())
            self.assertTrue((root / "output" / "tiny_en_sv" / "loss_history.csv").exists())


if __name__ == "__main__":
    unittest.main()
