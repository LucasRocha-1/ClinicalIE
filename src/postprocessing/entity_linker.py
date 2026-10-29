"""
Entity linking para um subconjunto local de conceitos clínicos UMLS.
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional


class EntityLinker:
    """Liga entidades a conceitos conhecidos sem exigir conexão ou credenciais externas."""

    def __init__(self, knowledge_base: str = "umls"):
        if knowledge_base.casefold() != "umls":
            raise ValueError("O EntityLinker local suporta apenas o vocabulário UMLS.")
        self.knowledge_base = "umls"
        self.concepts = {
            "infarto agudo do miocardio": {
                "cui": "C0027051",
                "name": "Myocardial Infarction",
                "aliases": ("iam", "infarto do miocardio"),
            },
            "diabetes mellitus": {
                "cui": "C0011849",
                "name": "Diabetes Mellitus",
                "aliases": ("diabetes",),
            },
            "hipertensao arterial": {
                "cui": "C0020538",
                "name": "Hypertensive disease",
                "aliases": (
                    "hipertensao",
                    "hipertensao arterial sistemica",
                ),
            },
            "acido acetilsalicilico": {
                "cui": "C0004057",
                "name": "Aspirin",
                "aliases": ("aas", "aspirina", "aspirin"),
            },
        }
        self._term_index = {
            self._normalize(term): concept
            for term, concept in self.concepts.items()
        }
        for concept in self.concepts.values():
            for alias in concept["aliases"]:
                self._term_index[self._normalize(alias)] = concept

    @staticmethod
    def _normalize(text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text.casefold())
        normalized = "".join(
            character
            for character in normalized
            if not unicodedata.combining(character)
        )
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", normalized)).strip()

    def link(self, entity_text: str) -> Optional[Dict[str, Any]]:
        """Retorna o conceito UMLS correspondente ou None quando não há correspondência."""
        concept = self._term_index.get(self._normalize(entity_text))
        if concept is None:
            return None

        return {
            "cui": concept["cui"],
            "name": concept["name"],
            "vocabulary": self.knowledge_base.upper(),
            "confidence": 1.0,
            "match": "exact_or_synonym",
        }

    def link_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Adiciona o conceito correspondente a cada entidade sem alterar a entrada."""
        return [
            {**entity, "linked": self.link(entity["text"])}
            for entity in entities
        ]
