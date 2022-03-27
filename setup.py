from typing import List

from setuptools import setup, find_packages


def get_requirements() -> List[str]:
    with open("requirements.txt", "r") as f:
        return list(f)


setup(
    name="paraphrase-triplets",
    version="0.0.2",
    url="https://github.com/futorio/paraphrase-triplets.git",
    author="Futorio Franklin",
    description="Create triplet dataset from paraphrase with ANN negative sampling",
    packages=find_packages(),
    install_requires=get_requirements(),
    entry_points={
        "console_scripts": ["vectorize-phrases=para_tri_dataset.vectorize_phrases:main"]
    }
)
