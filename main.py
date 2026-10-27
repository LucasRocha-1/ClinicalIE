"""
Ponto de entrada do sistema de Extração de Informações de Textos Clínicos.

Uso:
    python main.py
"""

from src.pipeline.clinical_pipeline import ClinicalInformationExtractionPipeline
from src.utils.io import save_results
import json


def main():
    print("=" * 60)
    print("Sistema de Extração de Informações Clínicas")
    print("=" * 60)

    # Inicializa o pipeline (lê config/settings.yaml automaticamente)
    pipeline = ClinicalInformationExtractionPipeline()

    # Exemplos de textos clínicos (sintéticos)
    sample_texts = [
        """
        Paciente de 58 anos, sexo masculino, apresentou-se com dor torácica e dispneia.
        Diagnóstico de Infarto Agudo do Miocárdio. Iniciado AAS 100mg e Clopidogrel 75mg.
        Realizado cateterismo cardíaco. PA 140x90 mmHg.
        """,
        """
        Paciente feminina, 72 anos, portadora de HAS e DM2. 
        Queixa de cefaleia e tontura. Glicemia de jejum 180 mg/dl.
        Prescrito Metformina 850mg 2x/dia.
        """,
    ]

    print("\nProcessando textos de exemplo...\n")

    results = pipeline.process_batch(sample_texts)

    for i, result in enumerate(results, 1):
        print(f"--- Texto {i} ---")
        print(f"Original (resumo): {result['original_text'][:80].strip()}...")
        print(f"Entidades encontradas ({result['metadata']['num_entities']}):")
        
        for ent in result["entities"]:
            print(f"  [{ent['label']:12}] {ent['text']}")
        
        print(f"Tipos: {result['metadata']['entity_types']}")
        print()

    # Salva resultados
    save_results(results, "data/processed/example_results.json")
    print("Pipeline executado com sucesso.")


if __name__ == "__main__":
    main()
