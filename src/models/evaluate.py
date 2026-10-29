"""
Avaliação de modelos de NER e extração.
"""

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from xml.sax.saxutils import escape

import spacy
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _entity_key(entity: Dict[str, Any]) -> Tuple[int, int, str]:
    return (entity["start"], entity["end"], entity["label"])


def _match_entities(
    gold_entities: List[Dict[str, Any]],
    pred_entities: List[Dict[str, Any]],
    match_type: str,
) -> Tuple[int, int, int, Dict[str, Tuple[int, int, int]]]:
    gold = {_entity_key(entity) for entity in gold_entities}
    predicted = {_entity_key(entity) for entity in pred_entities}
    per_label: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])

    if match_type == "exact":
        matched_gold = gold & predicted
        matched_predicted = matched_gold
    elif match_type == "partial":
        candidates = []
        for gold_entity in gold:
            for predicted_entity in predicted:
                if gold_entity[2] != predicted_entity[2]:
                    continue
                overlap = min(gold_entity[1], predicted_entity[1]) - max(
                    gold_entity[0], predicted_entity[0]
                )
                if overlap > 0:
                    candidates.append((overlap, gold_entity, predicted_entity))

        candidates.sort(
            key=lambda candidate: (-candidate[0], candidate[1], candidate[2])
        )
        matched_gold = set()
        matched_predicted = set()
        for _, gold_entity, predicted_entity in candidates:
            if gold_entity not in matched_gold and predicted_entity not in matched_predicted:
                matched_gold.add(gold_entity)
                matched_predicted.add(predicted_entity)
    else:
        raise ValueError("match_type deve ser 'exact' ou 'partial'.")

    for entity in matched_gold:
        per_label[entity[2]][0] += 1
    for entity in predicted - matched_predicted:
        per_label[entity[2]][1] += 1
    for entity in gold - matched_gold:
        per_label[entity[2]][2] += 1

    return (
        len(matched_gold),
        len(predicted - matched_predicted),
        len(gold - matched_gold),
        {label: tuple(counts) for label, counts in per_label.items()},
    )


def _metrics_from_counts(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
    support: int,
) -> Dict[str, Any]:
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "support": support,
    }


def compute_ner_metrics(
    gold_entities: List[Dict[str, Any]],
    pred_entities: List[Dict[str, Any]],
    match_type: str = "exact",
) -> Dict[str, Any]:
    """
    Calcula métricas micro e por classe para entidades de um único texto.

    Use ``evaluate_ner_model`` para avaliar vários textos sem misturar offsets
    iguais pertencentes a documentos diferentes.
    """
    tp, fp, fn, label_counts = _match_entities(
        gold_entities, pred_entities, match_type
    )
    metrics = _metrics_from_counts(tp, fp, fn, len(set(map(_entity_key, gold_entities))))
    metrics["per_label"] = {
        label: _metrics_from_counts(label_tp, label_fp, label_fn, label_tp + label_fn)
        for label, (label_tp, label_fp, label_fn) in sorted(label_counts.items())
    }
    return metrics


def evaluate_ner_model(
    nlp: Any,
    evaluation_data: List[Dict[str, Any]],
    match_type: str = "exact",
) -> Dict[str, Any]:
    """Avalia um modelo spaCy em documentos anotados e agrega contagens corretamente."""
    totals: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])
    total_tp = total_fp = total_fn = support = 0

    for record in evaluation_data:
        doc = nlp(record["text"])
        gold_entities = [
            {"start": start, "end": end, "label": label}
            for start, end, label in record["entities"]
        ]
        predicted_entities = [
            {"start": entity.start_char, "end": entity.end_char, "label": entity.label_}
            for entity in doc.ents
        ]
        tp, fp, fn, label_counts = _match_entities(
            gold_entities, predicted_entities, match_type
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
        support += len(set(map(_entity_key, gold_entities)))
        for label, counts in label_counts.items():
            for index, count in enumerate(counts):
                totals[label][index] += count

    metrics = _metrics_from_counts(total_tp, total_fp, total_fn, support)
    metrics["per_label"] = {
        label: _metrics_from_counts(tp, fp, fn, tp + fn)
        for label, (tp, fp, fn) in sorted(totals.items())
    }
    return metrics


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Imprime métricas micro e por classe."""
    print("=== Métricas de NER (micro) ===")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-Score  : {metrics['f1']:.4f}")
    print(
        "TP / FP / FN : "
        f"{metrics['true_positives']} / {metrics['false_positives']} / "
        f"{metrics['false_negatives']}"
    )
    if metrics.get("per_label"):
        print("\n=== Métricas por classe ===")
        for label, label_metrics in metrics["per_label"].items():
            print(
                f"{label:15} P={label_metrics['precision']:.4f} "
                f"R={label_metrics['recall']:.4f} F1={label_metrics['f1']:.4f} "
                f"(suporte={label_metrics['support']})"
            )


def generate_pdf_report(
    metrics: Dict[str, Any],
    model_name: str,
    annotations_path: str,
    document_count: int,
    match_type: str,
    output_path: str,
) -> Path:
    """Gera um PDF com métricas gerais, por classe e informações da avaliação."""
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#183153"),
        spaceAfter=8 * mm,
    )
    section_style = ParagraphStyle(
        "ReportSection",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#183153"),
        spaceBefore=5 * mm,
        spaceAfter=3 * mm,
    )
    body_style = styles["BodyText"]
    small_style = ParagraphStyle(
        "ReportNote",
        parent=body_style,
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#555555"),
    )

    document = SimpleDocTemplate(
        str(destination),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="Relatório de avaliação de NER",
        author="ClinicalIE",
    )
    story = [
        Paragraph("Relatório de avaliação de NER", title_style),
        Paragraph(
            f"<b>Modelo:</b> {escape(model_name)}<br/>"
            f"<b>Anotações:</b> {escape(annotations_path)}<br/>"
            f"<b>Documentos avaliados:</b> {document_count}<br/>"
            f"<b>Correspondência:</b> {escape(match_type)}<br/>"
            f"<b>Gerado em:</b> "
            f"{datetime.now().astimezone().strftime('%d/%m/%Y %H:%M %Z')}",
            body_style,
        ),
        Paragraph("Resultado geral (micro)", section_style),
    ]

    summary_table = Table(
        [
            ["Precision", "Recall", "F1"],
            [
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1']:.4f}",
            ],
        ],
        colWidths=[(A4[0] - 36 * mm) / 3] * 3,
        hAlign="LEFT",
    )
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#183153")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#eef3f8")),
                ("TEXTCOLOR", (0, 1), (-1, 1), colors.HexColor("#183153")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 11),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.white),
            ]
        )
    )
    story.extend(
        [
            summary_table,
            Spacer(1, 3 * mm),
            Paragraph(
                "Contagens: "
                f"verdadeiros positivos = {metrics['true_positives']}; "
                f"falsos positivos = {metrics['false_positives']}; "
                f"falsos negativos = {metrics['false_negatives']}; "
                f"suporte = {metrics['support']}.",
                body_style,
            ),
            Paragraph("Métricas por classe", section_style),
        ]
    )

    label_rows = [["Classe", "Precision", "Recall", "F1", "Suporte", "TP / FP / FN"]]
    for label, label_metrics in sorted(metrics.get("per_label", {}).items()):
        label_rows.append(
            [
                Paragraph(escape(label), body_style),
                f"{label_metrics['precision']:.4f}",
                f"{label_metrics['recall']:.4f}",
                f"{label_metrics['f1']:.4f}",
                str(label_metrics["support"]),
                (
                    f"{label_metrics['true_positives']} / "
                    f"{label_metrics['false_positives']} / "
                    f"{label_metrics['false_negatives']}"
                ),
            ]
        )
    label_table = Table(
        label_rows,
        colWidths=[35 * mm, 24 * mm, 24 * mm, 20 * mm, 19 * mm, 36 * mm],
        repeatRows=1,
        hAlign="LEFT",
    )
    label_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#183153")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c8d2df")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend(
        [
            label_table,
            Spacer(1, 5 * mm),
            Paragraph(
                "<b>Observação:</b> as métricas descrevem apenas o conjunto "
                "de avaliação informado. Resultados de dados sintéticos ou "
                "pequenos não validam desempenho clínico e não substituem "
                "validação independente por especialistas.",
                small_style,
            ),
        ]
    )
    document.build(story)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia um modelo spaCy de NER")
    parser.add_argument("--model", required=True, help="Diretório do modelo spaCy")
    parser.add_argument("--annotations", required=True, help="Arquivo .json ou .jsonl")
    parser.add_argument(
        "--match-type",
        choices=("exact", "partial"),
        default="exact",
        help="Correspondência estrita ou com sobreposição de spans",
    )
    parser.add_argument(
        "--report",
        default="reports/ner_evaluation_report.pdf",
        help="Caminho do relatório PDF (padrão: reports/ner_evaluation_report.pdf)",
    )
    args = parser.parse_args()

    from src.models.train_ner import prepare_training_data

    evaluation_data = prepare_training_data(args.annotations)
    nlp = spacy.load(args.model)
    metrics = evaluate_ner_model(nlp, evaluation_data, args.match_type)
    print_metrics(metrics)
    report_path = generate_pdf_report(
        metrics,
        model_name=args.model,
        annotations_path=args.annotations,
        document_count=len(evaluation_data),
        match_type=args.match_type,
        output_path=args.report,
    )
    print(f"\nRelatório PDF salvo em: {report_path}")


if __name__ == "__main__":
    main()
