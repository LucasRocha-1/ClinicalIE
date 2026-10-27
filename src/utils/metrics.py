"""
Reexporta funções de métricas para facilitar imports.
"""

from src.models.evaluate import compute_ner_metrics, print_metrics

__all__ = ["compute_ner_metrics", "print_metrics"]
