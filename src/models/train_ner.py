"""
Fine-tuning de NER spaCy a partir de anotacoes JSON ou JSONL.

Exemplo:
    python -m src.models.train_ner --annotations data/annotations/train.jsonl
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import spacy
from spacy.training import Example


def prepare_training_data(annotations_path: str) -> List[Dict[str, Any]]:
    """
    Le anotacoes JSON/JSONL no formato:
    {"text": "Paciente com IAM.", "entities": [{"start": 14, "end": 17, "label": "DISEASE"}]}

    Os offsets start/end sao indices de caracteres, com end exclusivo.
    """
    path = Path(annotations_path).expanduser()
    if not path.is_absolute() and not path.is_file():
        project_root = Path(__file__).resolve().parents[2]
        project_path = project_root / path
        if project_path.is_file():
            path = project_path

    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de anotacoes nao encontrado: {path}")

    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"JSON invalido na linha {line_number} de {path}: {error.msg}"
                    ) from error
    elif path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as file:
            records = json.load(file)
        if not isinstance(records, list):
            raise ValueError("O arquivo JSON deve conter uma lista de anotacoes.")
    else:
        raise ValueError("Formato nao suportado; use um arquivo .json ou .jsonl.")

    if not records:
        raise ValueError(
            f"Nenhuma anotacao encontrada em {path}. O arquivo esta vazio ou "
            "contem apenas linhas em branco; adicione um objeto JSON por linha."
        )

    prepared = []
    for record_number, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"A anotacao {record_number} deve ser um objeto JSON.")
        text = record.get("text")
        entities = record.get("entities")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"A anotacao {record_number} precisa de texto nao vazio.")
        if not isinstance(entities, list):
            raise ValueError(
                f"A anotacao {record_number} precisa de uma lista 'entities'."
            )

        spans = []
        for entity_number, entity in enumerate(entities, start=1):
            if not isinstance(entity, dict):
                raise ValueError(
                    f"Entidade {entity_number} da anotacao {record_number} "
                    "deve ser um objeto."
                )
            start = entity.get("start")
            end = entity.get("end")
            label = entity.get("label")
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or not isinstance(label, str)
                or not label.strip()
            ):
                raise ValueError(
                    f"Entidade {entity_number} da anotacao {record_number} "
                    "precisa de start/end inteiros e label nao vazio."
                )
            if start < 0 or end <= start or end > len(text):
                raise ValueError(
                    f"Offsets invalidos na entidade {entity_number} "
                    f"da anotacao {record_number}."
                )
            spans.append((start, end, label.strip().upper()))

        spans.sort(key=lambda span: (span[0], span[1]))
        for previous, current in zip(spans, spans[1:]):
            if current[0] < previous[1]:
                raise ValueError(
                    f"Entidades sobrepostas na anotacao {record_number}: "
                    f"{previous} e {current}."
                )
        prepared.append({"text": text, "entities": spans})

    return prepared


def train_spacy_ner(
    train_data: List[Dict[str, Any]],
    output_dir: str = "models/ner_model",
    n_iter: int = 30,
    model_name: str = "en_core_web_sm",
    seed: int = 13,
) -> Path:
    """Ajusta o componente NER de um modelo spaCy pre-treinado e salva o resultado."""
    if n_iter < 1:
        raise ValueError("n_iter deve ser pelo menos 1.")
    if not train_data:
        raise ValueError("Nao ha anotacoes para treinamento.")
    if not any(record["entities"] for record in train_data):
        raise ValueError("As anotacoes precisam conter ao menos uma entidade.")

    spacy.util.fix_random_seed(seed)
    is_blank_model = model_name.startswith("blank:")
    nlp = (
        spacy.blank(model_name.removeprefix("blank:"))
        if is_blank_model
        else spacy.load(model_name)
    )
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    examples = []
    for record_number, record in enumerate(train_data, start=1):
        text = record["text"]
        spans = record["entities"]
        doc = nlp.make_doc(text)
        for start, end, label in spans:
            span = doc.char_span(start, end, label=label, alignment_mode="strict")
            if span is None:
                raise ValueError(
                    f"Offsets da entidade ({start}, {end}, {label}) na anotacao "
                    f"{record_number} nao estao alinhados aos tokens do modelo."
                )
            ner.add_label(label)
        examples.append(Example.from_dict(doc, {"entities": spans}))

    optimizer = (
        nlp.initialize(get_examples=lambda: examples)
        if is_blank_model
        else nlp.resume_training()
    )
    rng = random.Random(seed)
    other_pipes = [name for name in nlp.pipe_names if name != "ner"]
    with nlp.select_pipes(disable=other_pipes):
        for _ in range(n_iter):
            rng.shuffle(examples)
            losses: Dict[str, float] = {}
            for example in examples:
                nlp.update([example], sgd=optimizer, drop=0.2, losses=losses)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning de NER com spaCy")
    parser.add_argument("--annotations", required=True, help="Arquivo .json ou .jsonl")
    parser.add_argument("--base-model", default="en_core_web_sm")
    parser.add_argument("--output", default="models/ner_model")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    training_data = prepare_training_data(args.annotations)
    output_path = train_spacy_ner(
        training_data,
        output_dir=args.output,
        n_iter=args.iterations,
        model_name=args.base_model,
        seed=args.seed,
    )
    print(f"Modelo treinado salvo em: {output_path}")


if __name__ == "__main__":
    main()
