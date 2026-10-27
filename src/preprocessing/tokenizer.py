"""
Tokenização de textos clínicos.
Pode ser expandido para tokenizadores específicos (WordPiece, SentencePiece, etc.).
"""

from typing import List, Optional
import spacy
from spacy.tokens import Doc


class ClinicalTokenizer:
    """
    Wrapper de tokenização usando spaCy.
    Facilita troca futura por tokenizadores de transformers.
    """

    def __init__(self, model_name: str = "en_core_sci_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            # Fallback para modelo genérico caso o científico não esteja instalado
            print(f"[Aviso] Modelo '{model_name}' não encontrado. Usando 'en_core_web_sm'.")
            self.nlp = spacy.load("en_core_web_sm")

    def tokenize(self, text: str) -> List[str]:
        """Retorna lista de tokens."""
        doc = self.nlp(text)
        return [token.text for token in doc]

    def tokenize_with_pos(self, text: str) -> List[dict]:
        """Retorna tokens com POS tags."""
        doc = self.nlp(text)
        return [
            {
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
                "is_stop": token.is_stop,
            }
            for token in doc
        ]

    def get_doc(self, text: str) -> Doc:
        """Retorna o objeto Doc completo do spaCy."""
        return self.nlp(text)
