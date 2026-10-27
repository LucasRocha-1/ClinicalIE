"""
Entity Linking: liga entidades extraídas a ontologias clínicas.
Exemplos: UMLS, SNOMED-CT, CID-10, etc.

Este é um esqueleto. Implementações reais normalmente usam:
    - APIs oficiais (UMLS Metathesaurus)
    - Modelos de embedding + busca aproximada
    - scispaCy EntityLinker
"""

from typing import List, Dict, Any, Optional


class EntityLinker:
    """
    Esqueleto de Entity Linking.
    
    Objetivo futuro: mapear "infarto agudo do miocárdio" → CUI UMLS ou código SNOMED.
    """

    def __init__(self, knowledge_base: str = "umls"):
        self.knowledge_base = knowledge_base
        # Placeholder: em produção carregaria índices ou conectaria a API
        self._dummy_kb = {
            "infarto agudo do miocardio": {"cui": "C0027051", "name": "Myocardial Infarction"},
            "diabetes mellitus": {"cui": "C0011849", "name": "Diabetes Mellitus"},
            "hipertensao arterial": {"cui": "C0020538", "name": "Hypertensive disease"},
        }

    def link(self, entity_text: str) -> Optional[Dict[str, Any]]:
        """
        Tenta ligar uma entidade a um conceito da base de conhecimento.
        
        Returns:
            Dicionário com identificador e nome canônico, ou None.
        """
        key = entity_text.lower().strip()
        return self._dummy_kb.get(key)

    def link_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Adiciona informação de linking às entidades."""
        enriched = []
        for ent in entities:
            linked = self.link(ent["text"])
            new_ent = ent.copy()
            new_ent["linked"] = linked
            enriched.append(new_ent)
        return enriched
