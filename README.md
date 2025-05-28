# mlx-grpo

**Group Reletive Policy Optimization (GRPO) Trainer using MLX**

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

`mlx-grpo` provides a robust, extensible trainer for Generalized Reinforcement Policy Optimization (GRPO) on top of Apple’s [MLX](https://github.com/ml-explore/mlx) framework. The trainer is designed for RLHF (Reinforcement Learning from Human Feedback) and related tasks, supporting format/content reward evaluation, checkpointing, and seamless integration with Hugging Face datasets and MLX-LM models.

Key features:
- **Flexible Reward Functions**: Includes format, math evaluation, Jaccard, and multiple-choice correctness rewards.
- **Robust Checkpointing**: Atomic save and resume, output directory management, and save-on-exit.
- **Speculative Decoding**: (Optional) For faster rollouts with a draft model.
- **Rich Logging & Progress**: Uses the [rich](https://github.com/Textualize/rich) library for beautiful CLI visualization.
- **Integration with MLX-LM**: Leverages MLX-LM for models, tokenizers, and utilities.
- **Extensible Training Configuration**: Uses dataclasses for configuration management.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Arguments](#arguments)
- [Reward Functions](#reward-functions)
- [Citations](#citations)
- [License](#license)

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/adeelahmad/mlx-grpo.git
   cd mlx-grpo
   ```

2. **Install dependencies:**
   - Python >=3.8 is recommended.
   - Required packages (install with pip):
     ```bash
     pip install mlx mlx-lm numpy rich datasets
     ```
   - (Optional) For enhanced logging and resource monitoring:
     ```bash
     pip install psutil wandb
     ```

   - If using code from `llama_rl`, ensure it is accessible in your `PYTHONPATH`.

3. **Hardware:**  
   Requires Apple Silicon or any hardware supported by [MLX](https://github.com/ml-explore/mlx).

---

## Usage

This repo provides a trainer script (see `mlx_grpo_trainer_aligned.py`) for RLHF/GRPO training with flexible configuration.

### Basic Example

```bash
python mlx_grpo_trainer_aligned.py \
  --output_dir ./output_model \
  --model_path ./base_model \
  --train_dataset_path ./data/train.jsonl \
  --val_dataset_path ./data/valid.jsonl \
  --num_training_steps 10000 \
  --reward_content_type jaccard
```

### With Hugging Face datasets

```bash
python mlx_grpo_trainer_aligned.py \
  --dataset_name "your-hf-dataset" \
  --dataset_config "main"
```

### Arguments

The trainer supports many arguments (see `TrainingArgs` dataclass in the code). Some key ones:

| Argument                  | Description                                           | Default                |
|---------------------------|-------------------------------------------------------|------------------------|
| `--output_dir`            | Directory for checkpoints and outputs                 | `../OutputModel`       |
| `--model_path`            | Path or ID of the base MLX model                      | `../Model`             |
| `--train_dataset_path`    | Local training JSONL file (overrides HF dataset)      | `../dataset_512/train.jsonl` |
| `--val_dataset_path`      | Local validation JSONL file                           | `../dataset_512/valid.jsonl` |
| `--num_training_steps`    | Number of optimizer steps                             | `8500`                 |
| `--reward_content_type`   | Content reward type: `jaccard`, `math_eval`, `choice_correctness` | `jaccard`      |
| `--reward_format_weight`  | Weight for format reward (`0.0` - `1.0`)              | `0.5`                  |
| `--reward_content_weight` | Weight for content reward (`0.0` - `1.0`)             | `0.5`                  |

See the code for a full list (`TrainingArgs`).

---

## Reward Functions

The trainer supports several reward strategies for RLHF:

- **Format Reward:** Checks if output follows the required `<think>...</think><answer>...</answer>` structure.
- **Jaccard Reward:** Measures word overlap between generated and reference answers.
- **Math Evaluation Reward:** Evaluates numeric/math expressions for correctness.
- **Choice Correctness Reward:** For multiple-choice, measures overlap between predicted and reference options.

You can choose the content reward function via `--reward_content_type`.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Citations & References

- [MLX: Apple Machine Learning Framework](https://github.com/ml-explore/mlx)
- [mlx-lm: Language Model tools for MLX](https://github.com/ml-explore/mlx-lm)
- [rich: Python rich text and beautiful formatting](https://github.com/Textualize/rich)
- [datasets: Hugging Face Datasets](https://github.com/huggingface/datasets)

---

**Contributions are welcome!**  
Feel free to open issues or pull requests.

---

**Contact:**  
For questions or support, open an issue on [GitHub](https://github.com/adeelahmad/mlx-grpo/issues).
