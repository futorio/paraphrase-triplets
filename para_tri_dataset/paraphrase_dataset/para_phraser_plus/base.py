from dataclasses import dataclass

from para_tri_dataset.paraphrase_dataset.base import Phrase


@dataclass
class ParaPhraserPlusPhrase(Phrase):
    id: int
    text: str
