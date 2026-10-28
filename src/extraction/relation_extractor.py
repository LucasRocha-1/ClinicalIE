"""
Extração de relações entre entidades clínicas.
Exemplo: MEDICATION → DOSAGE → FREQUENCY
"""

import re
from typing import List, Dict, Any


class RelationExtractor:
    """Extrai relações clínicas com base em regras de proximidade e inferência de termos comuns."""

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

        self.possible_relations = {
            ("MEDICATION", "DOSAGE"): "has_dosage",
            ("MEDICATION", "FREQUENCY"): "has_frequency",
            ("MEDICATION", "ROUTE"): "administered_via",
            ("MEDICATION", "DISEASE"): "treats",
            ("SYMPTOM", "ANATOMY"): "located_in",
            ("PROCEDURE", "ANATOMY"): "performed_on",
        }

        self.medication_terms = {
            "aas": "MEDICATION",
            "acido acetilsalicilico": "MEDICATION",
            "aspirin": "MEDICATION",
            "clopidogrel": "MEDICATION",
            "metformina": "MEDICATION",
            "insulina": "MEDICATION",
            "enalapril": "MEDICATION",
            "ibuprofeno": "MEDICATION",
            "paracetamol": "MEDICATION",
            "amoxicilina": "MEDICATION",
            "simvastatina": "MEDICATION",
        }

        self.disease_terms = {
            "iam": "DISEASE",
            "infarto agudo do miocardio": "DISEASE",
            "dm2": "DISEASE",
            "diabetes": "DISEASE",
            "has": "DISEASE",
            "hipertensao": "DISEASE",
            "angina": "DISEASE",
            "dpoC": "DISEASE",
            "doenca pulmonar obstrutiva cronica": "DISEASE",
        }

    def _canonicalize_label(self, label: str) -> str:
        if label is None:
            return "UNKNOWN"
        normalized = str(label).upper().strip()
        return self.label_aliases.get(normalized, normalized)

    def _pair_relation(self, left_label: str, right_label: str) -> str | None:
        left_label = self._canonicalize_label(left_label)
        right_label = self._canonicalize_label(right_label)

        direct = (left_label, right_label)
        reverse = (right_label, left_label)

        relation = self.possible_relations.get(direct)
        if relation:
            return relation

        return self.possible_relations.get(reverse)

    def _normalize_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(entity)
        normalized["label"] = self._canonicalize_label(entity.get("label", "UNKNOWN"))
        normalized.setdefault("start", 0)
        normalized.setdefault("end", 0)
        return normalized

    def _infer_entities_from_text(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        inferred = [self._normalize_entity(entity) for entity in entities]
        known = {(e["start"], e["end"], e["label"]) for e in inferred}

        for term, label in {**self.medication_terms, **self.disease_terms}.items():
            pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                key = (match.start(), match.end(), label)
                if key in known:
                    continue
                inferred.append({
                    "text": match.group(0),
                    "label": label,
                    "start": match.start(),
                    "end": match.end(),
                    "source": "text_inference",
                })
                known.add(key)

        return inferred

    def extract(
        self,
        text: str,
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not entities:
            entities = []

        seen = set()
        relations: List[Dict[str, Any]] = []
        normalized_entities = self._infer_entities_from_text(text, entities)
        sorted_ents = sorted(normalized_entities, key=lambda e: (e["start"], e["end"]))

        for i, ent1 in enumerate(sorted_ents):
            for ent2 in sorted_ents[i + 1:]:
                distance = ent2["start"] - ent1["end"]
                if distance > 80:
                    break

                relation_type = self._pair_relation(ent1["label"], ent2["label"])
                if not relation_type:
                    continue

                if relation_type in {"has_dosage", "has_frequency", "administered_via"}:
                    head, tail = ent2, ent1
                    relation_type = relation_type
                else:
                    head, tail = ent1, ent2

                key = (
                    min(ent1["start"], ent2["start"]),
                    max(ent1["end"], ent2["end"]),
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
        return [
            self.extract(text, ents)
            for text, ents in zip(texts, entities_list)
        ]
