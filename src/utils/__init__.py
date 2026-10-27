from .io import load_texts, save_results
from .metrics import compute_ner_metrics  # reexport se quiser

__all__ = ["load_texts", "save_results"]
