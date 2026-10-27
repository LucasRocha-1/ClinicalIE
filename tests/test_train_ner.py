import json

import spacy
import pytest

import src.models.train_ner as train_ner
from src.models.train_ner import prepare_training_data, train_spacy_ner


def test_prepare_training_data_reads_character_offsets(tmp_path):
    annotations_path = tmp_path / "annotations.jsonl"
    annotations_path.write_text(
        json.dumps(
            {
                "text": "Paciente com IAM.",
                "entities": [{"start": 13, "end": 16, "label": "disease"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    data = prepare_training_data(str(annotations_path))

    assert data == [
        {"text": "Paciente com IAM.", "entities": [(13, 16, "DISEASE")]}
    ]


def test_prepare_training_data_resolves_relative_path_from_outside_project(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    module_path = project_root / "src" / "models" / "train_ner.py"
    module_path.parent.mkdir(parents=True)
    module_path.touch()
    annotations_path = project_root / "data" / "annotations" / "train.jsonl"
    annotations_path.parent.mkdir(parents=True)
    annotations_path.write_text(
        '{"text":"Sem sintomas.","entities":[]}\n',
        encoding="utf-8",
    )
    outside_project = tmp_path / "outside"
    outside_project.mkdir()
    monkeypatch.setattr(train_ner, "__file__", str(module_path))
    monkeypatch.chdir(outside_project)

    data = prepare_training_data("data/annotations/train.jsonl")

    assert data == [{"text": "Sem sintomas.", "entities": []}]


def test_prepare_training_data_reports_empty_file(tmp_path):
    annotations_path = tmp_path / "empty.jsonl"
    annotations_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="arquivo esta vazio"):
        prepare_training_data(str(annotations_path))


def test_train_spacy_ner_saves_loadable_model(tmp_path):
    output_path = tmp_path / "trained_model"
    data = [{"text": "Paciente com IAM.", "entities": [(13, 16, "DISEASE")]}]

    train_spacy_ner(data, str(output_path), n_iter=1, model_name="blank:pt")

    trained_model = spacy.load(output_path)
    assert "ner" in trained_model.pipe_names
    assert "DISEASE" in trained_model.get_pipe("ner").labels
