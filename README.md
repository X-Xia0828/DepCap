# DepCap

Official repository for the paper "DepCap: Adaptive Block-Wise Parallel Decoding for Efficient Diffusion LM Inference". This project introduces a novel algorithm that accelerates inference in Diffusion Language Models (DLMs) through adaptive block-wise parallel decoding.

- `dream/`: evaluation and adaptive block generation code for Dream
- `llada/`: evaluation and adaptive block generation code for LLaDA

The repository mainly contains evaluation entry points, generation utilities, and a few post-processing scripts for running experiments with `lm-eval`.

## Installation

Install the environment directly from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Dream Evaluation

Main entry points:

- `dream/eval.py`
- `dream/eval.sh`

Example:

```bash
cd dream
accelerate launch eval.py \
  --model dream \
  --model_args pretrained=Dream-org/Dream-v0-Base-7B,max_new_tokens=256,add_bos_token=true,L_min=8,L_max=128,lambda_u=1.2,topk_cache=64,alg=confidence_threshold,tau_low=0.8,tau_high=0.95,gamma=-16.0,show_speed=True,save_dir=results_dream/depcap/gsm8k \
  --tasks gsm8k \
  --batch_size 1 \
  --confirm_run_unsafe_code \
  --num_fewshot 5 \
  --output_path results_dream/depcap/gsm8k
```

Important arguments:

- `max_new_tokens`: generation length
- `L_min` / `L_max`: minimum and maximum adaptive block size
- `lambda_u`: uncertainty penalty weight
- `topk_cache`: top-k cache size
- `tau_low`, `tau_high`, `gamma`: candidate selection and conflict-control parameters

## LLaDA Evaluation

Main entry points:

- `llada/eval_llada.py`
- `llada/generate.py`
- `llada/eval_gsm8k.sh`

The provided shell script includes three common modes:

- `depcap`
- `prefix cache`
- `dual cache`

Example:

```bash
cd llada
accelerate launch eval_llada.py \
  --tasks gsm8k \
  --num_fewshot 5 \
  --confirm_run_unsafe_code \
  --model llada_dist \
  --output_path evals_results_depcap/gsm8k \
  --model_args model_path=GSAI-ML/LLaDA-8B-Instruct,gen_length=256,L_max=128,L_min=8,lambda_u=1.2,topk_cache=64,tau_low=0.8,tau_high=0.95,gamma=-16.0,show_speed=True,save_dir=evals_results_depcap/gsm8k
```

To enable cache-based variants, add:

- `use_cache=True`
- `dual_cache=True`

## Utility Scripts

- `postprocess_code.py`: post-processing for humaneval
- `postprocess_mbpp.py`: post-processing for MBPP
- `sanitize.py`: output cleanup and normalization
