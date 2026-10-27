"""
Named Entity Recognition (NER) para textos clínicos.
Núcleo da extração de informações.
"""

from typing import List, Dict, Any, Optional
import spacy
from spacy.tokens import Doc


class ClinicalNER:
    """
    Extração de entidades clínicas usando spaCy / scispaCy.
    
    Entidades típicas:
        - DISEASE / CONDITION
        - MEDICATION / CHEMICAL
        - SYMPTOM
        - PROCEDURE
        - ANATOMY
        - DOSAGE
    """

    def __init__(
        self,
        model_name: str = "en_core_sci_sm",
        target_labels: Optional[List[str]] = None,
    ):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"[Aviso] Modelo '{model_name}' não encontrado. "
                  f"Usando 'en_core_web_sm' como fallback.")
            print("Instale o modelo científico com: "
                  "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz")
            self.nlp = spacy.load("en_core_web_sm")

        self.target_labels = set(target_labels) if target_labels else None

    def extract(self, text: str) -> List[Dict[str, Any]]:
        """
        Extrai entidades de um texto.
        
        Returns:
            Lista de dicionários com:
                - text: texto da entidade
                - label: tipo da entidade
                - start: posição inicial (char)
                - end: posição final (char)
                - confidence: (opcional)
        """
        doc: Doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            if self.target_labels is None or ent.label_ in self.target_labels:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": None,  # spaCy padrão não fornece score facilmente
                })

        return entities

    def extract_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """Processa múltiplos textos."""
        return [self.extract(t) for t in texts]

    def get_entity_counts(self, text: str) -> Dict[str, int]:
        """Retorna contagem de entidades por tipo."""
        entities = self.extract(text)
        counts: Dict[str, int] = {}
        for ent in entities:
            counts[ent["label"]] = counts.get(ent["label"], 0) + 1
        return counts
