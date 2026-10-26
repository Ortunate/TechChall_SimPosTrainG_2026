# src/baseline_config.py

CONFIG = {
    # "self_test": test the evaluation pipeline without loading a model.
    # "evaluate": run inference with the actual model and dataset.
    "mode": "evaluate",
    # Hugging Face model repository ID or local model directory.
    "model_id": "Qwen/Qwen2.5-Math-1.5B",
    # Random seed used by the inference engine and sampler.
    "seed": 42,
    "dataset": {
        "path": "EleutherAI/hendrycks_math",
        "split": "test",
        # MATH dataset subject categories to evaluate.
        "categories": [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ],
        # Maximum records per category; None means all records.
        "limit_per_category": 5,
    },
    "llm": {
        # Floating-point precision for model weights and computation.
        "dtype": "bfloat16",
        # Number of GPUs used for tensor parallelism.
        "tensor_parallel_size": 1,
        # Fraction of GPU memory vLLM aims to use.
        "gpu_memory_utilization": 0.65,
        # Maximum total sequence length: prompt + generated tokens.
        "max_model_len": 4096,
    },
    "sampling": {
        # Sampling temperature; 1.0 means no temperature scaling.
        "temperature": 1.0,
        # Nucleus sampling threshold; 1.0 disables top-p truncation.
        "top_p": 1.0,
        # Maximum number of generated tokens per response.
        "max_tokens": 1024,
        # Stop generation after the answer closing tag.
        "stop": ["</answer>"],
        # Keep the closing tag for strict response-format grading.
        "include_stop_str_in_output": True,
    },
}
