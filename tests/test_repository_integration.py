import json
from datetime import date
from pathlib import Path

from clinical_re_bilingual.cli import main
from clinical_re_bilingual.inference.io import (
    read_predictions_jsonl,
    write_predictions_jsonl,
)
from clinical_re_bilingual.schema import Language, PredictionRecord, RelationLabel


def prediction() -> PredictionRecord:
    return PredictionRecord(
        example_id="synthetic-test-1",
        gold_label=RelationLabel.TrAP,
        predicted_label=RelationLabel.TrAP,
        model="synthetic-model",
        language=Language.ENGLISH,
        retriever="rar",
        num_shots=5,
        prompt_strategy="standard",
        temperature=0.0,
        run_date=date(2026, 10, 30),
    )


def test_prediction_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    assert write_predictions_jsonl([prediction()], path) == 1
    assert read_predictions_jsonl(path) == [prediction()]


def test_notebooks_are_unexecuted_entry_points() -> None:
    root = Path(__file__).parents[1]
    expected = [
        "01_prepare_authorized_data.ipynb",
        "02_train_rar.ipynb",
        "03_run_llm_experiments.ipynb",
        "04_train_baselines.ipynb",
        "05_evaluate_results.ipynb",
    ]
    for filename in expected:
        notebook = json.loads((root / "notebooks" / filename).read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert code_cells
        assert all(cell["execution_count"] is None for cell in code_cells)
        assert all(cell["outputs"] == [] for cell in code_cells)


def test_model_ignore_rule_targets_only_root_artifacts() -> None:
    root = Path(__file__).parents[1]
    rules = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/models/" in rules
    assert "models/" not in rules


def test_cli_help_lists_complete_workflow(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["clinical-re", "--help"])
    try:
        main()
    except SystemExit as error:
        assert error.code == 0
    output = capsys.readouterr().out
    assert "run-rar" in output
    assert "run-inference" in output
    assert "run-baseline" in output
    assert "evaluate" in output
    assert "summarize-results" in output
