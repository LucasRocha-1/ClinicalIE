import spacy

from src.models.evaluate import (
    compute_ner_metrics,
    evaluate_ner_model,
    generate_pdf_report,
)


def test_compute_ner_metrics_returns_micro_and_per_label_scores():
    gold = [
        {"start": 0, "end": 3, "label": "DISEASE"},
        {"start": 8, "end": 14, "label": "SYMPTOM"},
    ]
    predicted = [
        {"start": 0, "end": 3, "label": "DISEASE"},
        {"start": 8, "end": 14, "label": "MEDICATION"},
    ]

    metrics = compute_ner_metrics(gold, predicted)

    assert (metrics["precision"], metrics["recall"], metrics["f1"]) == (
        0.5,
        0.5,
        0.5,
    )
    assert metrics["per_label"]["DISEASE"]["f1"] == 1.0
    assert metrics["per_label"]["SYMPTOM"]["false_negatives"] == 1
    assert metrics["per_label"]["MEDICATION"]["false_positives"] == 1


def test_partial_matching_counts_overlapping_spans():
    metrics = compute_ner_metrics(
        [{"start": 0, "end": 5, "label": "DISEASE"}],
        [{"start": 2, "end": 7, "label": "DISEASE"}],
        match_type="partial",
    )

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0


def test_corpus_metrics_do_not_collapse_entities_at_same_offsets():
    nlp = spacy.blank("pt")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns([{"label": "DISEASE", "pattern": "IAM"}])
    evaluation_data = [
        {"text": "IAM", "entities": [(0, 3, "DISEASE")]},
        {"text": "IAM", "entities": [(0, 3, "DISEASE")]},
    ]

    metrics = evaluate_ner_model(nlp, evaluation_data)

    assert metrics["true_positives"] == 2
    assert metrics["support"] == 2
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 1.0


def test_generate_pdf_report_writes_pdf_with_metrics(tmp_path):
    metrics = compute_ner_metrics(
        [{"start": 0, "end": 3, "label": "DISEASE"}],
        [{"start": 0, "end": 3, "label": "DISEASE"}],
    )
    report_path = tmp_path / "nested" / "report.pdf"

    generated_path = generate_pdf_report(
        metrics,
        model_name="models/clinical_ner_demo",
        annotations_path="data/annotations/evaluation.jsonl",
        document_count=1,
        match_type="exact",
        output_path=str(report_path),
    )

    pdf_content = generated_path.read_bytes()
    assert generated_path == report_path
    assert pdf_content.startswith(b"%PDF-")
    assert pdf_content.rstrip().endswith(b"%%EOF")
