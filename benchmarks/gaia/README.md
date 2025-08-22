# GAIA Benchmark Integration for Agent-S

This directory contains the integration of the GAIA (General AI Assistant) benchmark with Agent-S.

## Overview

GAIA is a benchmark designed to test AI systems on real-world tasks that require reasoning, web browsing, and tool use. This integration focuses on the 2023 validation set with Level 1 tasks for initial testing.

## Setup

### 1. Install Dependencies

The required dependencies will be automatically installed when running the scripts. Main requirements:
- `datasets` - For loading GAIA dataset from Hugging Face
- `huggingface_hub` - For dataset access

### 2. Download Dataset

Run the download script to fetch the GAIA 2023 validation Level 1 tasks:

```bash
python benchmarks/gaia/download_gaia.py
```

This will:
- Download the GAIA dataset from Hugging Face
- Filter for Level 1 tasks only
- Save processed tasks to `data/gaia_2023_validation_level1.json`

## Running the Benchmark

### Quick Test

To verify the setup with a single task:

```bash
python benchmarks/gaia/quick_test.py
```

### Full Benchmark Run

To run the full benchmark suite:

```bash
python benchmarks/gaia/run_gaia_benchmark.py [OPTIONS]
```

#### Available Options

**Benchmark Configuration:**
- `--max_tasks N` - Run only the first N tasks
- `--task_ids ID1,ID2,...` - Run specific tasks by ID
 - `--max_steps N` - Maximum steps per task (default: 20)
- `--delay_between_tasks N` - Delay in seconds between tasks (default: 2)

**Model Configuration:**
- `--provider PROVIDER` - Main model provider (default: openai)
- `--model MODEL` - Main model name (default: gpt-5)
- `--temperature FLOAT` - Temperature for generation (default: 1.0; 0.0 = deterministic, higher = more creative)
- `--model_url URL` - Custom model API URL
- `--model_api_key KEY` - Custom model API key

**Grounding Model Configuration:**
- `--ground_provider PROVIDER` - Grounding model provider (default: open_router)
- `--ground_model MODEL` - Grounding model name (default: bytedance/ui-tars-1.5-7b)
- `--ground_url URL` - Grounding model URL
- `--ground_api_key KEY` - Grounding model API key
- `--grounding_width WIDTH` - Grounding image width (default: 1728)
- `--grounding_height HEIGHT` - Grounding image height (default: 1117)

**Agent Configuration:**
- `--enable_reflection` - Enable reflection agent
- `--max_trajectory_length N` - Maximum trajectory length (default: 8)

### Example Commands

Run first 5 tasks with GPT-4:
```bash
python benchmarks/gaia/run_gaia_benchmark.py --max_tasks 5 --model gpt-4o --temperature 0.7 --enable_reflection
```

Run specific tasks:
```bash
python benchmarks/gaia/run_gaia_benchmark.py --task_ids gaia_val_l1_0001,gaia_val_l1_0002
```

Set a deterministic run (temperature 0.0) with a custom provider/model:
```bash
python benchmarks/gaia/run_gaia_benchmark.py \
  --max_tasks 3 \
  --provider openai \
  --model gpt-5 \
  --temperature 1.0
```

## Results

Results are saved in the `results/` directory with timestamps:
- `gaia_results_YYYYMMDD_HHMMSS.json` - Full results with task details
- `gaia_run_YYYYMMDD_HHMMSS.log` - Execution logs

Each result file includes:
- Task details (ID, question, expected answer)
- Agent's answer and trajectory
- Execution time
- Correctness evaluation
- Overall accuracy metrics

## Dataset Information

The GAIA benchmark includes tasks that test:
- Question answering
- Basic reasoning
- Tool use and web browsing
- Multi-step problem solving

Level 1 tasks are the simplest category, suitable for initial testing and validation.

## Troubleshooting

1. **Dataset download fails**: Ensure you have internet connection and Hugging Face is accessible
2. **API key errors**: Set environment variables for your model providers (e.g., `OPENAI_API_KEY`)
3. **Import errors**: Ensure you're running from the Agent-S root directory

## Citation

If you use GAIA benchmark, please cite:
```
@article{mialon2023gaia,
  title={GAIA: a benchmark for General AI Assistants},
  author={Mialon, Grégoire and others},
  journal={arXiv preprint arXiv:2311.12983},
  year={2023}
}
```