# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true


task=gsm8k
length=256
num_fewshot=5
model_path="GSAI-ML/LLaDA-8B-Instruct"
L_max=128
L_min=8
lambda_u=1.2
topk_cache=64
tau_low=0.8
tau_high=0.95
gamma=-16.0

# depcap
for length in 256; do
  echo "Running: gen_length=${length}"
  accelerate launch eval_llada.py \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code \
    --model llada_dist \
    --output_path "evals_results_depcap/${task}_fewshot_${num_fewshot}/genlength_${length}_L_min_${L_min}_L_max_${L_max}_lambda_u_${lambda_u}_topk_cache_${topk_cache}_tau_low_${tau_low}_tau_high_${tau_high}_gamma_${gamma}" \
    --model_args model_path=${model_path},gen_length=${length},L_max=${L_max},L_min=${L_min},lambda_u=${lambda_u},topk_cache=${topk_cache},tau_low=${tau_low},tau_high=${tau_high},gamma=${gamma},show_speed=True,save_dir="evals_results_depcap/${task}_fewshot_${num_fewshot}/genlength_${length}_L_min_${L_min}_L_max_${L_max}_lambda_u_${lambda_u}_topk_cache_${topk_cache}_tau_low_${tau_low}_tau_high_${tau_high}_gamma_${gamma}"
done

# prefix cache
for length in 256; do
  echo "Running: gen_length=${length}"
  accelerate launch eval_llada.py \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code \
    --model llada_dist \
    --output_path "evals_results_depcap/${task}_fewshot_${num_fewshot}/genlength_${length}_L_min_${L_min}_L_max_${L_max}_lambda_u_${lambda_u}_topk_cache_${topk_cache}_tau_low_${tau_low}_tau_high_${tau_high}_gamma_${gamma}_use_cache" \
    --model_args model_path=${model_path},gen_length=${length},L_max=${L_max},L_min=${L_min},lambda_u=${lambda_u},topk_cache=${topk_cache},tau_low=${tau_low},tau_high=${tau_high},gamma=${gamma},use_cache=True,show_speed=True,save_dir="evals_results_depcap/${task}_fewshot_${num_fewshot}/genlength_${length}_L_min_${L_min}_L_max_${L_max}_lambda_u_${lambda_u}_topk_cache_${topk_cache}_tau_low_${tau_low}_tau_high_${tau_high}_gamma_${gamma}_use_cache"
done


# dual cache
for length in 256; do
  echo "Running: gen_length=${length}"
  accelerate launch eval_llada.py \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code \
    --model llada_dist \
    --output_path "evals_results_depcap/${task}_fewshot_${num_fewshot}/genlength_${length}_L_min_${L_min}_L_max_${L_max}_lambda_u_${lambda_u}_topk_cache_${topk_cache}_tau_low_${tau_low}_tau_high_${tau_high}_gamma_${gamma}_dual_cache" \
    --model_args model_path=${model_path},gen_length=${length},L_max=${L_max},L_min=${L_min},lambda_u=${lambda_u},topk_cache=${topk_cache},tau_low=${tau_low},tau_high=${tau_high},gamma=${gamma},use_cache=True,dual_cache=True,show_speed=True,save_dir="evals_results_depcap/${task}_fewshot_${num_fewshot}/genlength_${length}_L_min_${L_min}_L_max_${L_max}_lambda_u_${lambda_u}_topk_cache_${topk_cache}_tau_low_${tau_low}_tau_high_${tau_high}_gamma_${gamma}_dual_cache"
done


# task=humaneval
# length=256
# num_fewshot=0
# model_path="GSAI-ML/LLaDA-8B-Instruct"
# L_max=128
# L_min=8
# lambda_u=1.2
# topk_cache=64
# tau_low=0.8
# tau_high=0.95
# gamma=-16.0

# # depcap
# for length in 256; do
#   echo "Running: gen_length=${length}"
#   accelerate launch eval_llada.py \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --confirm_run_unsafe_code \
#     --model llada_dist \
#     --output_path "evals_results_depcap/${task}_fewshot_${num_fewshot}/genlength_${length}_L_min_${L_min}_L_max_${L_max}_lambda_u_${lambda_u}_topk_cache_${topk_cache}_tau_low_${tau_low}_tau_high_${tau_high}_gamma_${gamma}" \
#     --model_args model_path=${model_path},gen_length=${length},L_max=${L_max},L_min=${L_min},lambda_u=${lambda_u},topk_cache=${topk_cache},tau_low=${tau_low},tau_high=${tau_high},gamma=${gamma},show_speed=True,save_dir="evals_results_depcap/${task}_fewshot_${num_fewshot}/genlength_${length}_L_min_${L_min}_L_max_${L_max}_lambda_u_${lambda_u}_topk_cache_${topk_cache}_tau_low_${tau_low}_tau_high_${tau_high}_gamma_${gamma}"
#     --log_samples
# done



