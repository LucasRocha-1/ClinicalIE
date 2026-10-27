"""
Utilitários de entrada/saída.
"""

from typing import List, Dict, Any
from pathlib import Path
import json


def load_texts(path: str, encoding: str = "utf-8") -> List[str]:
    """
    Carrega textos de um arquivo ou diretório.
    
    - Se for arquivo .txt: uma linha = um texto
    - Se for diretório: lê todos os .txt
    - Se for .json: espera lista de strings ou lista de dicts com chave 'text'
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Caminho não encontrado: {path}")

    if p.is_file():
        if p.suffix == ".json":
            with open(p, "r", encoding=encoding) as f:
                data = json.load(f)
            if isinstance(data, list):
                if data and isinstance(data[0], dict):
                    return [item.get("text", "") for item in data]
                return data
            return []
        else:
            with open(p, "r", encoding=encoding) as f:
                return [line.strip() for line in f if line.strip()]

    # Diretório
    texts = []
    for file in sorted(p.glob("*.txt")):
        with open(file, "r", encoding=encoding) as f:
            texts.append(f.read().strip())
    return texts


def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """Salva resultados da extração em JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Resultados salvos em: {output_path}")
