# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true


task=gsm8k
length=256
L_min=8
L_max=128
lambda_u=1.2
topk_cache=64
num_fewshot=5
model="Dream-org/Dream-v0-Base-7B"

for length in 256; do
    output_dir="results_dream/depcap/${task}_fewshot_${num_fewshot}/genlen_${length}_L_min_${L_min}_L_max_${L_max}_lambda_u_${lambda_u}_topk_cache_${topk_cache}_tau_low_${tau_low}_tau_high_${tau_high}_gamma_${gamma}"
    accelerate launch eval.py --model dream \
        --model_args pretrained=${model},max_new_tokens=${length},add_bos_token=true,L_min=${L_min},L_max=${L_max},lambda_u=${lambda_u},topk_cache=${topk_cache},alg=confidence_threshold,tau_low=${tau_low},tau_high=${tau_high},gamma=${gamma},show_speed=True,save_dir=${output_dir} \
        --tasks ${task} \
        --batch_size 1 \
        --confirm_run_unsafe_code \
        --num_fewshot ${num_fewshot} \
        --output_path ${output_dir} 
done
