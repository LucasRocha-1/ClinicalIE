"""
Pipeline principal de Extração de Informações Clínicas.
Orquestra pré-processamento → NER → (relações) → pós-processamento.
"""

from typing import List, Dict, Any, Optional
import yaml
from pathlib import Path

from src.preprocessing.cleaner import ClinicalTextCleaner
from src.preprocessing.normalizer import ClinicalNormalizer
from src.extraction.ner import ClinicalNER
from src.extraction.rule_based import RuleBasedExtractor
# from src.extraction.relation_extractor import RelationExtractor


class ClinicalInformationExtractionPipeline:
    """
    Pipeline completo e extensível.
    
    Fluxo atual:
        1. Limpeza do texto
        2. Normalização (abreviações, unidades)
        3. NER (spaCy/scispaCy)
        4. Extração por regras (dosagens, PA, etc.)
        5. (Futuro) Extração de relações
        6. (Futuro) Entity Linking
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)

        # Componentes
        self.cleaner = ClinicalTextCleaner(
            lowercase=self.config.get("preprocessing", {}).get("lowercase", True),
            normalize_spaces=self.config.get("preprocessing", {}).get("normalize_spaces", True),
        )
        self.normalizer = ClinicalNormalizer()
        
        ner_config = self.config.get("nlp", {})
        self.ner = ClinicalNER(
            model_name=ner_config.get("model_name", "en_core_sci_sm"),
            target_labels=ner_config.get("target_labels"),
        )
        
        self.rule_extractor = RuleBasedExtractor()
        
        # Descomentar quando implementar:
        # self.relation_extractor = RelationExtractor()

    def _load_config(self, config_path: Optional[str]) -> dict:
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"[Aviso] Arquivo de configuração não encontrado: {config_path}")
            return {}

    def process(self, text: str) -> Dict[str, Any]:
        """
        Processa um único texto clínico.
        
        Returns:
            Dicionário estruturado com resultados da extração.
        """
        # 1. Pré-processamento
        cleaned = self.cleaner.clean(text)
        normalized = self.normalizer.normalize(cleaned)

        # 2. Extração de entidades (NER + regras)
        ner_entities = self.ner.extract(normalized)
        rule_entities = self.rule_extractor.extract(normalized)

        # Combina e remove possíveis duplicatas simples
        all_entities = ner_entities + rule_entities

        # 3. Relações (esqueleto)
        relations = []
        # if self.config.get("pipeline", {}).get("enable_relation_extraction"):
        #     relations = self.relation_extractor.extract(normalized, all_entities)

        return {
            "original_text": text,
            "cleaned_text": cleaned,
            "normalized_text": normalized,
            "entities": all_entities,
            "relations": relations,
            "metadata": {
                "num_entities": len(all_entities),
                "entity_types": sorted(list({e["label"] for e in all_entities})),
                "num_relations": len(relations),
            },
        }

    def process_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Processa uma lista de textos."""
        return [self.process(t) for t in texts]

    def get_entities_only(self, text: str) -> List[Dict[str, Any]]:
        """Atalho: retorna apenas as entidades."""
        result = self.process(text)
        return result["entities"]
