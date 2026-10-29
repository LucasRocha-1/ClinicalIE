"""
Extração de relações clínicas baseada em tipos de entidade e proximidade textual.
"""

import re
import unicodedata
from typing import Any, Dict, List


class RelationExtractor:
    """Infere entidades clínicas conhecidas e conecta pares compatíveis próximos."""

    def __init__(self):
        self.label_aliases = {
            "CHEMICAL": "MEDICATION",
            "DRUG": "MEDICATION",
            "MEDICINE": "MEDICATION",
            "DIAGNOSIS": "DISEASE",
            "CONDITION": "DISEASE",
            "PROBLEM": "DISEASE",
            "SYMPTOMS": "SYMPTOM",
            "ANATOMIC_STRUCTURE": "ANATOMY",
            "BODY_PART": "ANATOMY",
            "THERAPY": "PROCEDURE",
        }

        self.relation_types = {
            ("MEDICATION", "DOSAGE"): "has_dosage",
            ("MEDICATION", "FREQUENCY"): "has_frequency",
            ("MEDICATION", "ROUTE"): "administered_via",
            ("MEDICATION", "DISEASE"): "treats",
            ("SYMPTOM", "ANATOMY"): "located_in",
            ("PROCEDURE", "ANATOMY"): "performed_on",
        }

        self.known_terms = {
            "MEDICATION": (
                "acido acetilsalicilico",
                "aspirina",
                "aspirin",
                "clopidogrel",
                "metformina",
                "insulina",
                "enalapril",
                "ibuprofeno",
                "paracetamol",
                "amoxicilina",
                "simvastatina",
            ),
            "DISEASE": (
                "infarto agudo do miocardio",
                "diabetes mellitus tipo 2",
                "diabetes mellitus",
                "hipertensao arterial sistemica",
                "hipertensao arterial",
                "doenca pulmonar obstrutiva cronica",
                "trombose venosa profunda",
                "acidente vascular cerebral",
                "iam",
                "dm2",
                "diabetes",
                "has",
                "dpoc",
                "angina",
            ),
        }

    def _canonicalize_label(self, label: str) -> str:
        normalized = str(label or "UNKNOWN").upper().strip()
        return self.label_aliases.get(normalized, normalized)

    def _normalize_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(entity)
        normalized["label"] = self._canonicalize_label(entity.get("label"))
        return normalized

    @staticmethod
    def _fold_diacritics(text: str) -> str:
        return "".join(
            character
            for character in unicodedata.normalize("NFD", text)
            if not unicodedata.combining(character)
        )

    def infer_entities(
        self,
        text: str,
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Acrescenta entidades do vocabulário local que não foram encontradas pelo NER."""
        inferred = [self._normalize_entity(entity) for entity in entities]
        searchable_text = self._fold_diacritics(text)
        occupied_spans = [
            (entity["start"], entity["end"])
            for entity in inferred
            if isinstance(entity.get("start"), int)
            and isinstance(entity.get("end"), int)
        ]
        candidates = []

        for label, terms in self.known_terms.items():
            for term in terms:
                pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
                for match in pattern.finditer(searchable_text):
                    candidates.append((
                        match.start(),
                        match.end(),
                        label,
                        text[match.start():match.end()],
                    ))

        for start, end, label, matched_text in sorted(
            candidates, key=lambda candidate: (candidate[0], -(candidate[1] - candidate[0]))
        ):
            if any(start < known_end and end > known_start for known_start, known_end in occupied_spans):
                continue
            inferred.append({
                "text": matched_text,
                "label": label,
                "start": start,
                "end": end,
                "source": "clinical_lexicon",
            })
            occupied_spans.append((start, end))

        return inferred

    def extract(
        self,
        text: str,
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        normalized_entities = self.infer_entities(text, entities)
        sorted_entities = sorted(
            normalized_entities,
            key=lambda entity: (entity.get("start", 0), entity.get("end", 0)),
        )
        relations = []
        seen = set()

        for index, first in enumerate(sorted_entities):
            for second in sorted_entities[index + 1:]:
                distance = max(0, second.get("start", 0) - first.get("end", 0))
                if distance > 80:
                    break

                first_label = self._canonicalize_label(first.get("label"))
                second_label = self._canonicalize_label(second.get("label"))
                relation_type = self.relation_types.get((first_label, second_label))
                if relation_type:
                    head, tail = first, second
                else:
                    relation_type = self.relation_types.get((second_label, first_label))
                    if not relation_type:
                        continue
                    head, tail = second, first

                key = (
                    head.get("start"),
                    head.get("end"),
                    tail.get("start"),
                    tail.get("end"),
                    relation_type,
                )
                if key in seen:
                    continue
                seen.add(key)
                relations.append({
                    "head": head,
                    "tail": tail,
                    "relation": relation_type,
                    "distance": distance,
                    "source": "rule_based_relation",
                })

        return relations

    def extract_batch(
        self,
        texts: List[str],
        entities_list: List[List[Dict[str, Any]]],
    ) -> List[List[Dict[str, Any]]]:
        if len(texts) != len(entities_list):
            raise ValueError("texts e entities_list devem ter o mesmo tamanho.")
        return [
            self.extract(text, entities)
            for text, entities in zip(texts, entities_list)
        ]
