"""Модели векторизации предложений от Сбербанка"""

from typing import Optional, List, Dict, Any

from transformers import AutoTokenizer, AutoModel
import torch

from para_tri_dataset.config import Config
from para_tri_dataset.paraphrase_dataset.base import Phrase
from para_tri_dataset.phrase_vector_model.base import PhraseVectorModel, PhraseNumpyVector


DEFAULT_DEVICE = torch.device("cpu")
EPSILON = 1e-7


def l2_normalize(vectors: torch.FloatTensor) -> torch.FloatTensor:
    """
    Создание векторов с единичной l2 нормой.

    Из матрицы построчно вычисляется вектор l2 нормы
    Из этого вектора строится диагональная матрица
    Диагональная матрица инвертируется в значения обратные l2 нормам векторов

    При матричном умножении диагональной матрицы на матрицу векторов i-й вектор умножается на i-й диагональный
    элемент обратной l2 нормы данного вектора.

    Чтобы диагональная матрица l2 норм была невырожденная, к вектору l2 норм прибавляется небольшое значение EPSILON
    """
    l2_norms = torch.linalg.vector_norm(vectors, ord=2, dim=1)
    l2_norms += EPSILON

    l2_matrix = torch.diag(l2_norms)
    inv_l2 = torch.linalg.inv(l2_matrix)
    return inv_l2 @ vectors


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask, normalize_l2: bool = False) -> torch.FloatTensor:
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    vectors = sum_embeddings / sum_mask

    if normalize_l2:
        return l2_normalize(vectors)
    else:
        return vectors


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
        model_path = cfg.get("path", None)
        seq_len = cfg.get("seq_len")

        return cls.load(model_path, seq_len)

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

        matrix = mean_pooling(output, tokenized["attention_mask"], normalize_l2=True).cpu().numpy()

        phrases_vectors = []
        for i in range(len(phrases)):
            phrase, vector = phrases[i], matrix[i]
            phrases_vectors.append(PhraseNumpyVector(phrase.id, vector))

        return phrases_vectors
