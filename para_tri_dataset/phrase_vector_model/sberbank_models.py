"""Модели векторизации предложений от Сбербанка"""

from typing import Optional, List, Dict, Any

from transformers import AutoTokenizer, AutoModel
import torch

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
    return sum_embeddings / sum_mask


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
    def from_config(cls, cfg: Dict[str, Any]) -> "SbertLargeMTNLU":
        # TODO: просто скопировал это из ParaPhraserPlusFileDataset.from_config нужно вынести этот код

        try:
            name = cfg["name"]
        except KeyError:
            raise ValueError('config has not "name" attribute')

        dataset_name = cls.get_name()
        if dataset_name != name:
            msg = f'config dataset name "{name}" does not compare with dataset class name "{dataset_name}"'
            raise ValueError(msg)

        model_path = cfg.get("path")

        try:
            seq_len = cfg["seq_len"]
        except KeyError:
            raise ValueError(f"config has not attribute seq_len")

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
