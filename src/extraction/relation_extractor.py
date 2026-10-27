"""
Extração de relações entre entidades clínicas.
Exemplo: MEDICATION → DOSAGE → FREQUENCY

Este módulo está preparado como esqueleto.
Implementação futura pode usar:
    - Regras (padrões de co-ocorrência)
    - Modelos de classificação de pares de entidades
    - Modelos sequence-to-sequence / transformers
"""

from typing import List, Dict, Any, Tuple


class RelationExtractor:
    """
    Esqueleto para extração de relações.
    
    Relações típicas em textos clínicos:
        - has_dosage (medicamento → dose)
        - has_frequency (medicamento → frequência)
        - treats (medicamento → doença)
        - located_in (sintoma/doença → anatomia)
        - indicates (exame → resultado)
    """

    def __init__(self):
        # Pares de labels que podem formar relações
        self.possible_relations = {
            ("MEDICATION", "DOSAGE"): "has_dosage",
            ("MEDICATION", "FREQUENCY"): "has_frequency",
            ("MEDICATION", "DISEASE"): "treats",
            ("SYMPTOM", "ANATOMY"): "located_in",
            ("PROCEDURE", "ANATOMY"): "performed_on",
        }

    def extract(
        self,
        text: str,
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Extrai relações a partir de entidades já identificadas.
        
        Implementação atual: esqueleto simples baseado em proximidade.
        Substituir por modelo de ML quando disponível.
        """
        relations = []

        # Ordenar entidades por posição
        sorted_ents = sorted(entities, key=lambda e: e["start"])

        for i, ent1 in enumerate(sorted_ents):
            for ent2 in sorted_ents[i + 1:]:
                # Verificar se estão próximas (ex: < 50 caracteres)
                distance = ent2["start"] - ent1["end"]
                if distance > 50:
                    break  # Muito distantes

                pair = (ent1["label"], ent2["label"])
                reverse_pair = (ent2["label"], ent1["label"])

                relation_type = self.possible_relations.get(pair) or \
                                self.possible_relations.get(reverse_pair)

                if relation_type:
                    relations.append({
                        "head": ent1,
                        "tail": ent2,
                        "relation": relation_type,
                        "distance": distance,
                        "source": "proximity_rule",  # Trocar por "model" depois
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
