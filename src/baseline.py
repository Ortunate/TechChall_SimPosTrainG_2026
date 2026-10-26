from pathlib import Path
from datetime import datetime
import json
import time

from baseline_config import CONFIG
from drgrpo_grader import extract_boxed_answer, r1_zero_reward_fn

ROOT = Path(__file__).resolve().parents[1]


def build_prompt(question: str) -> str:
    """Build the prompt using the provided templates."""
    system = (ROOT / "prompts/system.txt").read_text(encoding="utf-8")
    user = (ROOT / "prompts/user.txt").read_text(encoding="utf-8")

    user = user.replace("{question}", question)
    return system.replace("{instruction}", user)


def load_test_records() -> list[dict]:
    """Load test records and extract each ground-truth answer once."""
    from datasets import load_dataset

    dataset_config = CONFIG["dataset"]
    categories = dataset_config["categories"]
    limit = dataset_config["limit_per_category"]

    if not categories:
        raise ValueError("At least one category is required")

    if limit is not None and limit <= 0:
        raise ValueError("limit_per_category must be positive or None")

    test_records = []

    for category in categories:
        dataset = load_dataset(
            path=dataset_config["path"],
            name=category,
            split=dataset_config["split"],
        )

        # A limit applies independently to each subject category.
        if limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))

        for dataset_index, item in enumerate(dataset):
            ground_truth = extract_boxed_answer(item["solution"])

            # Validate labels before any expensive model inference.
            if ground_truth is None:
                raise ValueError(f"Cannot extract answer: {category}[{dataset_index}]")

            test_records.append(
                {
                    "category": category,
                    "dataset_index": dataset_index,
                    "problem": item["problem"],
                    "ground_truth": ground_truth,
                    "level": item["level"],
                }
            )

        print(f"Loaded {len(dataset)} test records from {category}")

    return test_records


def generate_response_vllm(
    llm,
    sampling_params,
    prompts: list[str],
) -> list[str]:
    """Generate one response per prompt using vLLM."""
    request_outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
    )

    return [request_output.outputs[0].text for request_output in request_outputs]


def evaluate_one(response: str, ground_truth: str) -> dict:
    """Grade a complete response against a pre-extracted answer."""
    return r1_zero_reward_fn(
        response,
        ground_truth,
        fast=True,
    )


def save_results(
    test_records: list[dict],
    generated_responses: list[str],
    generation_seconds: float,
) -> dict:
    """Save per-question results, category metrics and TensorBoard logs."""
    from torch.utils.tensorboard import SummaryWriter

    evaluation_records = []

    # Accumulate statistics separately for each subject category.
    category_totals = {}

    for index, (test_record, response) in enumerate(
        zip(test_records, generated_responses, strict=True)
    ):
        result = evaluate_one(
            response=response,
            ground_truth=test_record["ground_truth"],
        )

        category = test_record["category"]

        if category not in category_totals:
            category_totals[category] = {
                "count": 0,
                "correct": 0.0,
                "formatted": 0.0,
            }

        totals = category_totals[category]
        totals["count"] += 1
        totals["correct"] += result["answer_reward"]
        totals["formatted"] += result["format_reward"]

        evaluation_records.append(
            {
                "index": index,
                "category": category,
                "dataset_index": test_record["dataset_index"],
                "level": test_record["level"],
                "question": test_record["problem"],
                "ground_truth": test_record["ground_truth"],
                "response": response,
                "format_reward": result["format_reward"],
                "answer_reward": result["answer_reward"],
            }
        )

    sample_count = len(evaluation_records)

    if sample_count == 0:
        raise ValueError("No test records were evaluated")

    accuracy = (
        sum(record["answer_reward"] for record in evaluation_records) / sample_count
    )

    format_rate = (
        sum(record["format_reward"] for record in evaluation_records) / sample_count
    )

    # Compute accuracy and format rate for each category.
    category_metrics = {}

    for category, totals in category_totals.items():
        count = totals["count"]

        category_metrics[category] = {
            "sample_count": count,
            "accuracy": totals["correct"] / count,
            "format_rate": totals["formatted"] / count,
        }

    summary = {
        "sample_count": sample_count,
        "accuracy": accuracy,
        "format_rate": format_rate,
        "by_category": category_metrics,
        "generation_seconds": generation_seconds,
        "config": CONFIG,
    }

    # Keep each run in a separate directory.
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-" + CONFIG["mode"]

    output_dir = ROOT / "outputs" / "baseline" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSONL stores one evaluated question per line.
    with (output_dir / "records.jsonl").open("w", encoding="utf-8") as file:
        for record in evaluation_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    # JSON stores aggregate metrics and the full run configuration.
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    log_dir = ROOT / "runs" / "baseline" / run_id

    with SummaryWriter(log_dir=str(log_dir)) as writer:
        writer.add_scalar("eval/accuracy", accuracy, 0)
        writer.add_scalar("eval/format_rate", format_rate, 0)
        writer.add_scalar("eval/generation_seconds", generation_seconds, 0)

        for category, metrics in category_metrics.items():
            writer.add_scalar(
                f"eval/category_accuracy/{category}",
                metrics["accuracy"],
                0,
            )

    print("\n=== Baseline Results ===")
    print(f"Samples:     {sample_count}")
    print(f"Accuracy:    {accuracy:.2%}")
    print(f"Format Rate: {format_rate:.2%}")
    print(f"Time:        {generation_seconds:.2f}s")

    print("\nAccuracy by category:")
    for category, metrics in category_metrics.items():
        print(
            f"  {category}: "
            f"{metrics['accuracy']:.2%} "
            f"({metrics['sample_count']} questions)"
        )

    print(f"\nResults: {output_dir}")
    print(f"TensorBoard: {log_dir}")

    # Display only a small number of responses in the terminal.
    for record in evaluation_records[:2]:
        print(f"\nQuestion {record['index']}: {record['question'][:150]}")
        print(f"Ground truth: {record['ground_truth']}")
        print(f"Response: {record['response'][:400]}")

    return summary


def run_self_test() -> None:
    """Test the evaluation and logging pipeline without GPU inference."""
    test_records = [
        {
            "category": "synthetic",
            "dataset_index": 0,
            "level": "test",
            "problem": "What is 2+3?",
            "ground_truth": "5",
        },
        {
            "category": "synthetic",
            "dataset_index": 1,
            "level": "test",
            "problem": "What is 2+2?",
            "ground_truth": "4",
        },
    ]

    generated_responses = [
        "2+3=5.</think> <answer>5</answer>",
        "2+2=5.</think> <answer>5</answer>",
    ]

    prompt = build_prompt(test_records[0]["problem"])
    assert "What is 2+3?" in prompt
    assert "{question}" not in prompt
    assert "{instruction}" not in prompt

    summary = save_results(
        test_records=test_records,
        generated_responses=generated_responses,
        generation_seconds=0.0,
    )

    assert summary["accuracy"] == 0.5
    assert summary["format_rate"] == 1.0

    print("\nLocal self-test PASSED")


def main() -> None:
    if CONFIG["mode"] == "self_test":
        run_self_test()
        return

    if CONFIG["mode"] != "evaluate":
        raise ValueError(f"Unknown mode: {CONFIG['mode']}")

    # Extract and validate all reference answers before loading the LLM.
    test_records = load_test_records()

    prompts = [build_prompt(record["problem"]) for record in test_records]

    # vLLM is only required during real GPU inference.
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=CONFIG["model_id"],
        seed=CONFIG["seed"],
        **CONFIG["llm"],
    )

    sampling_params = SamplingParams(
        seed=CONFIG["seed"],
        **CONFIG["sampling"],
    )

    start_time = time.perf_counter()

    generated_responses = generate_response_vllm(
        llm=llm,
        sampling_params=sampling_params,
        prompts=prompts,
    )

    generation_seconds = time.perf_counter() - start_time

    save_results(
        test_records=test_records,
        generated_responses=generated_responses,
        generation_seconds=generation_seconds,
    )


if __name__ == "__main__":
    main()
