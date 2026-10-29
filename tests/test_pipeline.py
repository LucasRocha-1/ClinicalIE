"""
Testes básicos do pipeline.
Execute com: pytest tests/
"""

import pytest
from src.pipeline.clinical_pipeline import ClinicalInformationExtractionPipeline
from src.preprocessing.cleaner import ClinicalTextCleaner
from src.preprocessing.normalizer import ClinicalNormalizer
from src.extraction.rule_based import RuleBasedExtractor
from src.postprocessing.entity_linker import EntityLinker


def test_cleaner():
    cleaner = ClinicalTextCleaner()
    text = "  Texto   com   espaços   extras  "
    cleaned = cleaner.clean(text)
    assert "  " not in cleaned
    assert cleaned == "texto com espaços extras"


def test_rule_based_dosage():
    extractor = RuleBasedExtractor()
    text = "AAS 100mg e Clopidogrel 75 mg"
    results = extractor.extract(text)
    assert len(results) >= 2
    labels = [r["label"] for r in results]
    assert "DOSAGE" in labels


def test_normalizer_expands_abbreviations_preserving_punctuation():
    normalizer = ClinicalNormalizer()
    assert normalizer.normalize_abbreviations("HAS, DM2 e DPOC.") == (
        "hipertensao arterial sistemica, diabetes mellitus tipo 2 e "
        "doenca pulmonar obstrutiva cronica."
    )
    assert normalizer.normalize_abbreviations("HASteria") == "HASteria"


def test_rule_based_extracts_vital_signs_and_measurements():
    extractor = RuleBasedExtractor()
    text = (
        "PA 120x80 mmHg, FC 78 bpm, SpO2 97%, glicemia 110 mg/dL, "
        "FEVE 55%, 38,2 C, AAS 100 mg."
    )
    labels = {entity["label"] for entity in extractor.extract(text)}
    assert {
        "BLOOD_PRESSURE",
        "HEART_RATE",
        "OXYGEN_SATURATION",
        "GLUCOSE",
        "TEMPERATURE",
        "DOSAGE",
        "EJECTION_FRACTION",
    }.issubset(labels)


def test_rule_based_handles_normalized_abbreviations():
    normalizer = ClinicalNormalizer()
    extractor = RuleBasedExtractor()
    text = normalizer.normalize(
        "PA 120x80 mmHg, FC 78 bpm, SpO2 97%, HGT 110 mg/dL, via EV."
    )

    labels = {entity["label"] for entity in extractor.extract(text)}
    assert {
        "BLOOD_PRESSURE",
        "HEART_RATE",
        "OXYGEN_SATURATION",
        "GLUCOSE",
        "ROUTE",
    }.issubset(labels)


def test_pipeline_runs():
    pipeline = ClinicalInformationExtractionPipeline()
    text = "Paciente com dor torácica. Diagnóstico de IAM. AAS 100mg."
    result = pipeline.process(text)

    assert "entities" in result
    assert "metadata" in result
    assert isinstance(result["entities"], list)
    assert result["metadata"]["num_entities"] >= 0


def test_relation_extraction_detects_common_clinical_pairs():
    pipeline = ClinicalInformationExtractionPipeline()
    text = "Paciente com dor torácica. Prescrito AAS 100mg 1x/dia para IAM."
    result = pipeline.process(text)

    relation_labels = {item["relation"] for item in result["relations"]}
    assert {"has_dosage", "has_frequency", "treats"}.issubset(relation_labels)
    assert any(entity["label"] == "MEDICATION" for entity in result["entities"])


def test_entity_linker_matches_accents_and_synonyms():
    linker = EntityLinker()

    linked = linker.link("Ácido acetilsalicílico")
    assert linked is not None
    assert linked["cui"] == "C0004057"
    assert linked["name"] == "Aspirin"
    assert linker.link("IAM")["cui"] == "C0027051"
    assert linker.link("conceito desconhecido") is None


def test_pipeline_adds_entity_linking_results():
    pipeline = ClinicalInformationExtractionPipeline()
    result = pipeline.process("Prescrito AAS para IAM.")

    medication = next(
        entity for entity in result["entities"]
        if entity["label"] == "MEDICATION"
    )
    assert medication["linked"]["cui"] == "C0004057"
    assert result["metadata"]["num_linked_entities"] >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
