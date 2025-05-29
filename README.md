# 🧠 MLX-GRPO: Train Your Own DeepSeek-R1 on Mac

<div align="center">

![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-Native-blue?style=for-the-badge&logo=apple)
![MLX](https://img.shields.io/badge/MLX-Optimized-orange?style=for-the-badge)
![GRPO](https://img.shields.io/badge/GRPO-DeepSeek_R1_Style-red?style=for-the-badge)
![MIT License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**🔥 The FIRST MLX implementation of GRPO - Train reasoning models like DeepSeek-R1 🔥**

*Build your own o1-style reasoning AI using the same technique that powers DeepSeek-R1*

[🚀 Quick Start](#-quick-start) • [🧠 What is GRPO?](#-what-is-grpo) • [⚡ Performance](#-why-mlx--apple-silicon) • [🎯 Examples](#-training-examples)

</div>

---

## 🎯 Why This Matters Right Now

**DeepSeek-R1 just shocked the AI world** by matching o1 performance using GRPO (Group Relative Policy Optimization). Now you can:

- 🧠 **Train o1-style reasoning models** - Same technique as DeepSeek-R1
- ⚡ **On your Mac** - Native Apple Silicon optimization via MLX
- 💰 **No human feedback needed** - Programmable rewards instead of expensive RLHF
- 🎯 **Multi-step reasoning** - Perfect for math, coding, and complex problems
- 🚀 **Production ready** - Robust checkpointing and speculative decoding

> *"GRPO is the technique behind DeepSeek-R1's breakthrough performance"* - Recent AI research shows GRPO enables direct optimization using programmable reward functions, making it more scalable than traditional RLHF approaches

## 🧠 What is GRPO?

**Group Relative Policy Optimization** is the secret sauce behind DeepSeek-R1's reasoning abilities:

- 📊 **Compares multiple responses** to the same question within each batch
- 🎯 **Learns from relative quality** - promotes better answers, demotes worse ones  
- 🔄 **Online learning** - improves iteratively using the model's own generated data
- 🎛️ **Programmable rewards** - no need for expensive human preference data
- 🧮 **Perfect for reasoning** - excels at multi-step problems like math and coding

The GRPO update compares multiple answers to a single question within a batch, teaching the model to become more like correct answers and less like incorrect ones.

## 🚀 Quick Start

Get your GRPO reasoning model running in **3 minutes**:

```bash
# 1. Clone and install
git clone https://github.com/adeelahmad/mlx-grpo.git
cd mlx-grpo
pip install mlx mlx-lm numpy rich datasets

# 2. Train a math reasoning model (like DeepSeek-R1)
python mlx_grpo_trainer_aligned.py \
  --model_path microsoft/DialoGPT-medium \
  --train_dataset_path ./data/math_problems.jsonl \
  --reward_content_type math_eval \
  --num_training_steps 5000

# 3. Test your reasoning model
python test_reasoning.py --model ./output_model
```

**That's it!** 🎉 You now have a reasoning model trained with the same technique as DeepSeek-R1.

## ⚡ Why MLX + Apple Silicon?

| Traditional Training | MLX-GRPO on Mac | Advantage |
|---------------------|-----------------|-----------|
| Requires expensive GPUs | Runs on any Mac with Apple Silicon | **💰 Cost savings** |
| Complex CUDA setup | Zero configuration needed | **🚀 Easy setup** |
| High memory usage | MLX optimized memory management | **📱 Efficient** |
| Slow on consumer hardware | Native Apple Silicon acceleration | **⚡ Fast training** |

*MLX is Apple's machine learning framework designed specifically for efficient training and inference on Apple Silicon.*

## 🎯 Training Examples

### 🧮 **Mathematics Reasoning (DeepSeek-R1 style)**
```bash
python mlx_grpo_trainer_aligned.py \
  --model_path microsoft/DialoGPT-medium \
  --train_dataset_path ./data/math_qa.jsonl \
  --reward_content_type math_eval \
  --reward_format_weight 0.3 \
  --reward_content_weight 0.7 \
  --num_training_steps 8500
```
*Trains a model to show step-by-step mathematical reasoning*

### 💭 **Chain-of-Thought Reasoning**
```bash
python mlx_grpo_trainer_aligned.py \
  --model_path microsoft/DialoGPT-large \
  --train_dataset_path ./data/reasoning.jsonl \
  --reward_content_type jaccard \
  --num_training_steps 10000
```
*Optimizes for the `<think>...</think><answer>...</answer>` format used by o1 and R1*

### 🎯 **Multiple Choice Questions**
```bash
python mlx_grpo_trainer_aligned.py \
  --dataset_name "your-mcq-dataset" \
  --reward_content_type choice_correctness \
  --num_training_steps 6000
```
*Perfect for training on standardized tests and benchmarks*

## 🛠️ Advanced Features

### 🎯 **Smart Reward System**
- **📝 Format Rewards**: Ensures proper `<think>...</think><answer>...</answer>` structure
- **🧮 Math Evaluation**: Automatically checks mathematical correctness  
- **📊 Jaccard Similarity**: Measures word overlap with reference answers
- **✅ Choice Correctness**: Perfect for multiple-choice problems
- **🔧 Custom Rewards**: Build your own reward functions

### 🚀 **Production Features**
- **💾 Atomic Checkpointing**: Never lose training progress
- **⚡ Speculative Decoding**: 2x faster inference with draft models
- **🎨 Rich CLI**: Beautiful progress bars and logging
- **🔄 Auto-Resume**: Continues exactly where you left off
- **📊 Weights & Biases**: Optional experiment tracking

### 🎛️ **Flexible Configuration**
```python
# All training parameters
@dataclass
class TrainingArgs:
    model_path: str = "../Model"
    output_dir: str = "../OutputModel" 
    num_training_steps: int = 8500
    reward_content_type: str = "jaccard"  # jaccard, math_eval, choice_correctness
    reward_format_weight: float = 0.5
    reward_content_weight: float = 0.5
    # ... and many more!
```

## 📊 Complete Configuration Options

<details>
<summary>📋 All Training Parameters</summary>

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--output_dir` | Directory for checkpoints and outputs | `../OutputModel` |
| `--model_path` | Path or ID of the base MLX model | `../Model` |
| `--train_dataset_path` | Local training JSONL file | `../dataset_512/train.jsonl` |
| `--val_dataset_path` | Local validation JSONL file | `../dataset_512/valid.jsonl` |
| `--num_training_steps` | Number of optimizer steps | `8500` |
| `--reward_content_type` | Content reward: `jaccard`, `math_eval`, `choice_correctness` | `jaccard` |
| `--reward_format_weight` | Weight for format reward (0.0 - 1.0) | `0.5` |
| `--reward_content_weight` | Weight for content reward (0.0 - 1.0) | `0.5` |

*See `TrainingArgs` dataclass in the code for the complete list*

</details>

## 🔥 What's Hot About This

### 🎯 **Trending AI Techniques**
- ✅ **GRPO** - Same as DeepSeek-R1 (trending #1 on Twitter)
- ✅ **Chain-of-Thought** - o1-style reasoning format
- ✅ **Apple Silicon ML** - Fastest growing ML platform
- ✅ **Reward-Free RL** - No expensive human feedback needed

### 🚀 **Perfect Timing**
- 🔥 **DeepSeek-R1** just dominated benchmarks using GRPO
- 📈 **Apple MLX** adoption growing rapidly  
- 💡 **Reasoning models** are the hottest topic in AI
- 💰 **Cost-effective** alternative to GPT-4/Claude for reasoning

## 🤝 Community & Support

<div align="center">

**Join the MLX + GRPO Revolution**

[![Issues](https://img.shields.io/github/issues/adeelahmad/mlx-grpo?style=for-the-badge)](https://github.com/adeelahmad/mlx-grpo/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/adeelahmad/mlx-grpo?style=for-the-badge)](https://github.com/adeelahmad/mlx-grpo/pulls)
[![Stars](https://img.shields.io/github/stars/adeelahmad/mlx-grpo?style=for-the-badge)](https://github.com/adeelahmad/mlx-grpo/stargazers)

</div>

### 🚀 Resources
- 📚 [GRPO Explained](https://www.deeplearning.ai/short-courses/reinforcement-fine-tuning-llms-grpo/) - DeepLearning.AI Course
- 🔬 [DeepSeek-R1 Technical Report](https://arxiv.org/abs/2501.12948) - How they used GRPO
- 🍎 [MLX Documentation](https://github.com/ml-explore/mlx) - Apple's ML framework
- 💬 [HuggingFace GRPO Guide](https://huggingface.co/docs/trl/main/en/grpo_trainer) - Alternative implementation

## 🛠️ Requirements

- 🍎 **Apple Silicon Mac** (M1, M2, M3, M4) or any MLX-supported hardware
- 🐍 **Python ≥3.8**
- 📦 **Dependencies**: `mlx`, `mlx-lm`, `numpy`, `rich`, `datasets`
- 💾 **Optional**: `psutil`, `wandb` for enhanced monitoring

## 🤝 Contributing

We ❤️ contributions! This is a hot research area with lots of room for improvement:

1. 🍴 **Fork the repo**
2. 🌿 **Create feature branch** (`git checkout -b amazing-feature`)
3. 💫 **Commit changes** (`git commit -m 'Add amazing feature'`)
4. 🚀 **Push to branch** (`git push origin amazing-feature`)
5. 🎉 **Open Pull Request**

### 🎯 Contribution Ideas
- 🔧 New reward functions for specific domains
- ⚡ Performance optimizations for MLX
- 📊 Better evaluation metrics
- 🎨 Enhanced CLI visualization
- 📝 More training examples and tutorials

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- 🍎 **Apple** for the incredible [MLX framework](https://github.com/ml-explore/mlx)
- 🤗 **HuggingFace** for [MLX-LM](https://github.com/ml-explore/mlx-lm) and [datasets](https://github.com/huggingface/datasets)
- 🎨 **Textualize** for the beautiful [Rich](https://github.com/Textualize/rich) library
- 🧠 **DeepSeek** for pioneering GRPO in their R1 model
- 🔬 **Research community** advancing reinforcement learning for LLMs

---

<div align="center">

**⭐ Star us if you're excited about training reasoning models on Mac! ⭐**

*Built with 🧠 for the future of AI reasoning*

**🔥 Trending:** `#GRPO` `#DeepSeekR1` `#MLX` `#AppleSilicon` `#ReasoningAI` `#MachineLearning`

</div>
