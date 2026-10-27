"""
Pré-processamento de textos clínicos.
Compatível com técnicas de Data Preprocessing e NLP.
"""

import re
from typing import List, Optional


class ClinicalTextCleaner:
    """
    Limpeza e normalização básica de textos clínicos.
    
    Mantém abreviações médicas, unidades (mg, mmHg, etc.) e
    estruturas importantes para NER posterior.
    """

    def __init__(self, lowercase: bool = True, normalize_spaces: bool = True):
        self.lowercase = lowercase
        self.normalize_spaces = normalize_spaces
        
        self.patterns = {
            "multiple_spaces": re.compile(r"\s+"),
            "leading_trailing": re.compile(r"^\s+|\s+$"),
            # Remoção opcional de caracteres muito especiais (cuidado com unidades)
            "special_chars": re.compile(r"[^\w\s\.\,\;\:\-\/\(\)\%\+\=]"),
        }

    def clean(self, text: str) -> str:
        """
        Limpa um único texto clínico.
        
        Args:
            text: Texto bruto
            
        Returns:
            Texto limpo
        """
        if not text or not isinstance(text, str):
            return ""

        cleaned = text.strip()

        if self.lowercase:
            cleaned = cleaned.lower()

        if self.normalize_spaces:
            cleaned = self.patterns["multiple_spaces"].sub(" ", cleaned)

        return cleaned.strip()

    def batch_clean(self, texts: List[str]) -> List[str]:
        """Aplica limpeza em lote."""
        return [self.clean(t) for t in texts]

    def clean_preserving_case(self, text: str) -> str:
        """Versão que preserva capitalização (útil para alguns modelos)."""
        if not text:
            return ""
        cleaned = self.patterns["multiple_spaces"].sub(" ", text.strip())
        return cleaned
