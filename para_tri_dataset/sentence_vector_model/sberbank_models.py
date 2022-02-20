"""Модели векторизации предложений от Сбербанка"""

from typing import Optional, List

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask) -> torch.FloatTensor:

    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class SbertLargeMTNLU:
    """
    https://huggingface.co/sberbank-ai/sbert_large_nlu_ru

    Пачка текстов приводится к максимальной длине, отсечением токенов и паддингом
    """

    HF_URL = 'sberbank-ai/sbert_large_mt_nlu_ru'

    def __init__(self, model, tokenizer, max_seq_len: int = 24):
        self.name = 'sbert_large_mt_nlu_ru'

        self.model = model
        self.tokenizer = tokenizer

        self.max_seq_len = max_seq_len

    @classmethod
    def load(cls, model_path: Optional[str], seq_len: int = 24) -> 'SbertLargeMTNLU':
        if model_path is None:
            path = cls.HF_URL
        else:
            path = model_path

        model = AutoModel.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)

        return cls(model, tokenizer, seq_len)

    def get_vector_size(self) -> int:
        return self.model.config.hidden_size

    def create_sentences_vectors(self, sentences: List[str], max_seq_len: int = 24,
                                 device: torch.device = torch.device('cpu')) -> np.array:

        tokenized = self.tokenizer(sentences, max_length=max_seq_len, padding=True, truncation=True,
                                   return_tensors='pt').to(device)

        with torch.no_grad():
            output = self.model(**tokenized)

        return mean_pooling(output, tokenized['attention_mask']).cpu().numpy()
