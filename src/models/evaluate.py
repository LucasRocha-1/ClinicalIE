"""
Avaliação de modelos de NER e extração.
Métricas clássicas: Precision, Recall, F1 (por entidade e micro/macro).
"""

from typing import List, Dict, Any, Tuple
from collections import defaultdict


def compute_ner_metrics(
    gold_entities: List[Dict[str, Any]],
    pred_entities: List[Dict[str, Any]],
    match_type: str = "exact",  # "exact" ou "partial"
) -> Dict[str, Any]:
    """
    Calcula Precision, Recall e F1 para NER.
    
    Args:
        gold_entities: Entidades anotadas (ground truth)
        pred_entities: Entidades preditas pelo modelo
        match_type: "exact" (span + label) ou "partial" (overlap)
    
    Returns:
        Dicionário com métricas globais e por tipo de entidade.
    """
    # Implementação simplificada (esqueleto)
    # Em produção usar seqeval, spacy.scorer ou scikit-learn
    
    def entity_key(e: Dict) -> Tuple:
        return (e["start"], e["end"], e["label"])

    gold_set = {entity_key(e) for e in gold_entities}
    pred_set = {entity_key(e) for e in pred_entities}

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "support": len(gold_set),
    }


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Imprime métricas de forma legível."""
    print("=== Métricas de NER ===")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-Score  : {metrics['f1']:.4f}")
    print(f"TP / FP / FN : {metrics['true_positives']} / {metrics['false_positives']} / {metrics['false_negatives']}")
