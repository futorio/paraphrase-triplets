"""Модели векторизации предложений от Сбербанка"""

from typing import Optional, List, Dict, Any

from transformers import AutoTokenizer, AutoModel
import torch

from para_tri_dataset.config import Config
from para_tri_dataset.paraphrase_dataset.base import Phrase
from para_tri_dataset.phrase_vector_model.base import PhraseVectorModel, PhraseNumpyVector


DEFAULT_DEVICE = torch.device("cpu")


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask) -> torch.FloatTensor:

    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    vectors = sum_embeddings / sum_mask

    inv_norms = torch.linalg.inv(torch.diag(torch.linalg.norm(vectors, ord=2, dim=1)))
    return inv_norms @ vectors


class SbertLargeMTNLU(PhraseVectorModel):
    """
    https://huggingface.co/sberbank-ai/sbert_large_nlu_ru

    Пачка текстов приводится к максимальной длине, отсечением токенов и паддингом
    """

    HF_URL = "sberbank-ai/sbert_large_mt_nlu_ru"

    def __init__(self, model, tokenizer, seq_len: int = 24):
        self.model = model
        self.tokenizer = tokenizer

        self.seq_len = seq_len

    @staticmethod
    def get_name() -> str:
        return "sbert_large_mt_nlu_ru"

    @classmethod
    def from_config(cls, cfg: Config) -> "SbertLargeMTNLU":
        model_path = cfg.get("path")
        seq_len = cfg.get("seq_len")

        if model_path is None:
            path = cls.HF_URL
        else:
            path = model_path

        return cls.load(path, seq_len)

    @classmethod
    def load(cls, model_path: Optional[str], seq_len: int = 24) -> "SbertLargeMTNLU":
        if model_path is None:
            path = cls.HF_URL
        else:
            path = model_path

        model = AutoModel.from_pretrained(path)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(path)
        return cls(model, tokenizer, seq_len)

    def to_device(self, device: torch.device):
        self.model.to(device)

    def get_vector_size(self) -> int:
        return self.model.config.hidden_size

    def create_phrases_vectors(
        self, phrases: List[Phrase], device: torch.device = DEFAULT_DEVICE
    ) -> List[PhraseNumpyVector]:

        texts = [p.text for p in phrases]
        tokenized = self.tokenizer(
            texts, max_length=self.seq_len, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = self.model(**tokenized)

        matrix = mean_pooling(output, tokenized["attention_mask"]).cpu().numpy()

        phrases_vectors = []
        for phrase, vector in zip(phrases, matrix):
            phrases_vectors.append(PhraseNumpyVector(phrase.id, vector))

        return phrases_vectors
