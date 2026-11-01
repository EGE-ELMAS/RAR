"""Small command-line entry point for environment verification."""

from __future__ import annotations

import argparse
from pathlib import Path

from clinical_re_bilingual import __version__
from clinical_re_bilingual.data.i2b2 import (
    I2B2Layout,
    parse_i2b2_directory,
    read_relation_examples_tsv,
    write_tsv,
)
from clinical_re_bilingual.data.manifests import create_split_manifest
from clinical_re_bilingual.evaluation.io import write_evaluation_outputs
from clinical_re_bilingual.inference.io import read_predictions_jsonl
from clinical_re_bilingual.prompting.builder import PromptStrategy
from clinical_re_bilingual.schema import Language
from clinical_re_bilingual.workflows.inference import run_inference_workflow
from clinical_re_bilingual.workflows.rar import run_rar_workflow


def main() -> None:
    parser = argparse.ArgumentParser(prog="clinical-re")
    parser.add_argument("--version", action="store_true")
    subparsers = parser.add_subparsers(dest="command")
    prepare = subparsers.add_parser("prepare-i2b2", help="parse authorized local i2b2 files")
    prepare.add_argument("--concepts-dir", type=Path, required=True)
    prepare.add_argument("--relations-dir", type=Path, required=True)
    prepare.add_argument("--text-dir", type=Path, required=True)
    prepare.add_argument("--language", choices=[item.value for item in Language], required=True)
    prepare.add_argument("--output", type=Path, required=True)
    manifest = subparsers.add_parser(
        "create-split-manifest", help="create a public-safe train/dev/test ID manifest"
    )
    manifest.add_argument("--training-pool", type=Path, required=True)
    manifest.add_argument("--official-test", type=Path, required=True)
    manifest.add_argument("--dataset-name", required=True)
    manifest.add_argument("--dev-fraction", type=float, default=0.2)
    manifest.add_argument("--seed", type=int, default=42)
    manifest.add_argument("--output", type=Path, required=True)
    rar = subparsers.add_parser("run-rar", help="train RAR and create ID-only test rankings")
    rar.add_argument("--training-pool", type=Path, required=True)
    rar.add_argument("--official-test", type=Path, required=True)
    rar.add_argument("--manifest", type=Path, required=True)
    rar.add_argument("--language", choices=[item.value for item in Language], required=True)
    rar.add_argument("--config", type=Path, default=Path("config/rar.yaml"))
    rar.add_argument("--models-config", type=Path, default=Path("config/models.yaml"))
    rar.add_argument("--output-dir", type=Path, required=True)
    rar.add_argument("--device", default=None)
    inference = subparsers.add_parser(
        "run-inference", help="run one few-shot experiment and save label-only predictions"
    )
    inference.add_argument("--training-pool", type=Path, required=True)
    inference.add_argument("--official-test", type=Path, required=True)
    inference.add_argument(
        "--rankings",
        type=Path,
        help="ID-only rankings; omit only for zs_static_cot",
    )
    inference.add_argument("--provider", choices=["openai", "gemini", "deepseek"], required=True)
    inference.add_argument("--model", required=True)
    inference.add_argument("--num-shots", type=int, required=True)
    inference.add_argument(
        "--strategy", choices=[item.value for item in PromptStrategy], required=True
    )
    inference.add_argument("--temperature", type=float, default=0.0)
    inference.add_argument("--rationales", type=Path)
    inference.add_argument("--output", type=Path, required=True)
    baseline = subparsers.add_parser(
        "run-baseline", help="train and evaluate a supervised BERT or PURE baseline"
    )
    baseline.add_argument("--baseline", choices=["bert", "pure"], required=True)
    baseline.add_argument("--training-pool", type=Path, required=True)
    baseline.add_argument("--official-test", type=Path, required=True)
    baseline.add_argument("--manifest", type=Path, required=True)
    baseline.add_argument("--language", choices=[item.value for item in Language], required=True)
    baseline.add_argument("--config", type=Path, default=Path("config/baselines.yaml"))
    baseline.add_argument("--models-config", type=Path, default=Path("config/models.yaml"))
    baseline.add_argument("--output-dir", type=Path, required=True)
    baseline.add_argument("--device")
    evaluate = subparsers.add_parser(
        "evaluate", help="compute metrics from label-only prediction records"
    )
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--metrics", type=Path, required=True)
    evaluate.add_argument("--confusion-matrix", type=Path, required=True)
    summarize = subparsers.add_parser(
        "summarize-results", help="validate and aggregate label-only prediction files"
    )
    summarize.add_argument("--predictions-dir", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.version:
        print(__version__)
    elif args.command == "prepare-i2b2":
        layout = I2B2Layout(args.concepts_dir, args.relations_dir, args.text_dir)
        examples = parse_i2b2_directory(layout, Language(args.language))
        write_tsv(examples, args.output)
        print(f"Wrote {len(examples)} protected examples to {args.output}")
    elif args.command == "create-split-manifest":
        training_pool = read_relation_examples_tsv(args.training_pool)
        official_test = read_relation_examples_tsv(args.official_test)
        split_manifest = create_split_manifest(
            training_pool,
            official_test,
            dataset_name=args.dataset_name,
            dev_fraction=args.dev_fraction,
            seed=args.seed,
        )
        split_manifest.write_json(args.output)
        print(f"Wrote public-safe split manifest to {args.output}")
    elif args.command == "run-rar":
        result = run_rar_workflow(
            training_pool_path=args.training_pool,
            official_test_path=args.official_test,
            manifest_path=args.manifest,
            language=Language(args.language),
            rar_config_path=args.config,
            models_config_path=args.models_config,
            output_dir=args.output_dir,
            device=args.device,
        )
        print(f"Saved RAR checkpoint to {result.checkpoint_dir}")
        print(f"Saved {result.test_examples} ID-only rankings to {result.rankings_path}")
    elif args.command == "run-inference":
        count = run_inference_workflow(
            training_pool_path=args.training_pool,
            official_test_path=args.official_test,
            rankings_path=args.rankings,
            provider=args.provider,
            model_name=args.model,
            num_shots=args.num_shots,
            strategy=PromptStrategy(args.strategy),
            temperature=args.temperature,
            output_path=args.output,
            rationales_path=args.rationales,
        )
        print(f"Wrote {count} label-only predictions to {args.output}")
    elif args.command == "run-baseline":
        from clinical_re_bilingual.workflows.baselines import run_baseline_workflow

        count = run_baseline_workflow(
            baseline=args.baseline,
            training_pool_path=args.training_pool,
            official_test_path=args.official_test,
            manifest_path=args.manifest,
            language=Language(args.language),
            baselines_config_path=args.config,
            models_config_path=args.models_config,
            output_dir=args.output_dir,
            device=args.device,
        )
        print(f"Wrote {count} label-only baseline predictions to {args.output_dir}")
    elif args.command == "evaluate":
        records = read_predictions_jsonl(args.predictions)
        write_evaluation_outputs(records, args.metrics, args.confusion_matrix)
        print(f"Wrote metrics to {args.metrics}")
    elif args.command == "summarize-results":
        from clinical_re_bilingual.evaluation.aggregate import write_results_summary

        count = write_results_summary(args.predictions_dir, args.output_dir)
        print(f"Validated and summarized {count} experiments")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
