"""PyTorch dataset wrapping fixture images + generated (or human) text."""

from __future__ import annotations

from datetime import UTC

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from microvlm.data.coco import FixtureRecord
from microvlm.data.schemas import GeneratedRecord


class SimpleTokenizer:
    """Whitespace tokenizer with a fixed vocab size (no external tokenizer).

    Special ids: pad=0, bos=1, eos=2, unk=3. Remaining slots are filled from
    training text. ``vocab_size`` is taken from the model YAML so parameter
    counts stay in the NanoVLM ballpark (GPT-2's 50k vocab would dominate).
    """

    PAD, BOS, EOS, UNK = 0, 1, 2, 3

    def __init__(self, vocab_size: int = 8192) -> None:
        self.vocab_size = vocab_size
        self.stoi: dict[str, int] = {}

    def fit(self, texts: list[str]) -> None:
        """Build a vocab from whitespace tokens, truncated to ``vocab_size``."""

        counts: dict[str, int] = {}
        for text in texts:
            for tok in text.lower().split():
                counts[tok] = counts.get(tok, 0) + 1
        ranked = sorted(counts, key=lambda t: (-counts[t], t))
        self.stoi = {}
        next_id = 4
        for tok in ranked:
            if next_id >= self.vocab_size:
                break
            self.stoi[tok] = next_id
            next_id += 1

    def encode(self, text: str, max_len: int, add_special: bool = True) -> list[int]:
        """Encode text to ids, padded/truncated to ``max_len``."""

        ids = [self.BOS] if add_special else []
        for tok in text.lower().split():
            ids.append(self.stoi.get(tok, self.UNK))
        if add_special:
            ids.append(self.EOS)
        ids = ids[:max_len]
        ids += [self.PAD] * (max_len - len(ids))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Detokenize, skipping specials."""

        itos = {i: t for t, i in self.stoi.items()}
        words = []
        for i in ids:
            if i in {self.PAD, self.BOS, self.EOS}:
                continue
            words.append(itos.get(i, "<unk>"))
        return " ".join(words)


def default_image_transform(image_size: int = 224) -> transforms.Compose:
    """Resize/crop/normalize to a 224×224 RGB tensor."""

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


class CaptionDataset(Dataset):
    """Pairs fixture images with target text for teacher-forced training."""

    def __init__(
        self,
        fixture_index: dict[str, FixtureRecord],
        records: list[GeneratedRecord],
        tokenizer: SimpleTokenizer,
        image_size: int = 224,
        max_seq_len: int = 64,
    ) -> None:
        self.fixture_index = fixture_index
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.transform = default_image_transform(image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        rec = self.records[idx]
        fixture = self.fixture_index[rec.image_id]
        image = Image.open(fixture.image_path).convert("RGB")
        pixel = self.transform(image)
        ids = self.tokenizer.encode(rec.generated_text, max_len=self.max_seq_len)
        token_ids = torch.tensor(ids[:-1], dtype=torch.long)
        targets = torch.tensor(ids[1:], dtype=torch.long)
        targets[token_ids == SimpleTokenizer.PAD] = -100
        return {
            "image": pixel,
            "token_ids": token_ids,
            "targets": targets,
            "image_id": rec.image_id,
            "text": rec.generated_text,
        }


def records_from_fixture_captions(
    fixtures: list[FixtureRecord],
    prompt_id: str = "human",
    teacher_name: str = "human",
) -> list[GeneratedRecord]:
    """Use the first human caption as target text (no teacher required)."""

    from datetime import datetime

    out: list[GeneratedRecord] = []
    for fx in fixtures:
        text = fx.captions[0] if fx.captions else ""
        out.append(
            GeneratedRecord(
                image_id=fx.image_id,
                captions=fx.captions,
                prompt_id=prompt_id,
                teacher_name=teacher_name,
                generated_text=text,
                word_count=len(text.split()),
                timestamp=datetime.now(UTC),
            )
        )
    return out
