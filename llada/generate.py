# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.cuda import nvtx
from transformers import AutoTokenizer

from model.modeling_llada import LLaDAModelLM

def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, low-precision Gumbel Max improves
    perplexity score but reduces generation quality. Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def entropy_from_logits_batched(logits_2d: torch.Tensor) -> torch.Tensor:
    """
    Exact entropy H(p) from logits (S, V), batched over S.
    Returns: (S,)
    """
    logits_f = logits_2d.to(torch.float64)
    logZ = torch.logsumexp(logits_f, dim=-1)  # (S,)
    p = torch.exp(logits_f - logZ.unsqueeze(-1))  # (S, V)
    H = logZ - torch.sum(p * logits_f, dim=-1)  # (S,)
    return H


def robust_norm(x: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
    """
    Robust normalization within a window:
      (x - median) / IQR
    x: (L,)
    """
    x = x.to(torch.float64)
    med = x.median()
    L = x.numel()
    k1 = max(int(0.25 * (L - 1)), 0) + 1
    k3 = max(int(0.75 * (L - 1)), 0) + 1
    q1 = x.kthvalue(k1).values
    q3 = x.kthvalue(k3).values
    iqr = (q3 - q1).abs().clamp(min=eps)
    return (x - med) / iqr


@dataclass
class TopKDist:
    ids: torch.Tensor  # (K,)
    logp: torch.Tensor  # (K,)
    logp_floor: torch.Tensor  # scalar tensor


@dataclass
class TopKCacheWindow:
    """
    Cache a contiguous window of positions [start, start+S).
    Stored in batched tensors for speed.
    """
    start: int
    ids: torch.Tensor  # (S, K) long
    logp: torch.Tensor  # (S, K) float
    logp_floor: torch.Tensor  # (S,) float

    def at(self, abs_pos: int) -> TopKDist:
        i = abs_pos - self.start
        return TopKDist(
            ids=self.ids[i],
            logp=self.logp[i],
            logp_floor=self.logp_floor[i],
        )


def topk_cache_window_from_logits(
    logits_2d: torch.Tensor, K: int, start: int
) -> TopKCacheWindow:
    """
    Batched TopKDist for a block of positions.
    logits_2d: (S, V)
    Returns TopKCacheWindow with:
      ids: (S,K), logp: (S,K), logp_floor: (S,)
    """
    logits_f = logits_2d.to(torch.float64)
    topk_logits, ids = torch.topk(logits_f, k=K, dim=-1)

    logZ = torch.logsumexp(logits_f, dim=-1)
    logp = topk_logits.to(torch.float64) - logZ.unsqueeze(-1)  # (S, K)

    topk_logmass = torch.logsumexp(logp, dim=-1)  # (S,)
    topk_mass = torch.exp(topk_logmass).clamp(max=1 - 1e-12)
    rem_logmass = torch.log1p(-topk_mass)  # (S,)

    V = logits_2d.shape[-1]
    denom = max(V - K, 1)
    logp_floor = rem_logmass - rem_logmass.new_tensor(math.log(denom))  # (S,)

    return TopKCacheWindow(start=start, ids=ids, logp=logp, logp_floor=logp_floor)


def approx_kl_topk_batched(
    p_ids: torch.Tensor,
    p_logp: torch.Tensor,
    q_ids: torch.Tensor,
    q_logp: torch.Tensor,
    q_logp_floor: torch.Tensor,
) -> torch.Tensor:
    """
    Batched approx KL for multiple positions at once.
    Shapes:
      p_ids:  (S, K)
      p_logp: (S, K)
      q_ids:  (S, K)
      q_logp: (S, K)
      q_logp_floor: (S,)
    Returns:
      kl: (S,)
    """
    p_prob = torch.exp(p_logp)

    eq = (p_ids.unsqueeze(2) == q_ids.unsqueeze(1))
    matched = eq.any(dim=2)

    q_logp_matched = (eq.to(p_logp.dtype) * q_logp.unsqueeze(1)).sum(dim=2)

    q_floor = q_logp_floor.unsqueeze(1).expand_as(p_logp)  # (S, K)
    q_lp = torch.where(matched, q_logp_matched, q_floor)  # (S, K)

    kl = torch.sum(p_prob * (p_logp - q_lp), dim=1)  # (S,)
    return kl

@torch.no_grad()
def generate_adaptive_block(
    model,
    prompt,
    gen_length=128,
    L_max: int = 128,                # max adaptive block length
    L_min: int = 8,                  # min adaptive block length
    temperature=0.,
    remasking='low_confidence',
    mask_id=126336,
    lambda_u: float = 1.0,  # Weight for uncertainty penalty.
    topk_cache: int = 64,            # store topK for baseline cache
    tau_low: float = 0.8,
    tau_high: float = 0.95,
    gamma: float = -16.0,
):
    """
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        gen_length: Generated answer length.
        L_max: Maximum block length.
        L_min: Minimum block length.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        threshold: Confidence threshold for unmasking.
        lambda_u: Weight for uncertainty penalty.
        tau_low: Lower threshold for confidence.
        tau_high: Upper threshold for confidence.
        gamma: Gamma parameter for conflict detect.
        topk_cache: Top-K cache size for baseline model.
    Returns:
        x: Generated sequence of shape (1, L + gen_length).
        nfe: Number of function evaluations.
        transfer_steps: Tensor indicating the step at which each token was transferred.
    """

    device = model.device
    B = prompt.shape[0]

    prompt_len = int(prompt.shape[1])
    total_len = int(prompt_len + gen_length)

    x = torch.full((B, total_len), mask_id, dtype=torch.long).to(device)
    x[:, :prompt_len] = prompt.clone()

    assert 1 <= L_min <= L_max
    assert B == 1  # Currently only support batch size 1.

    base_cache: Optional[TopKCacheWindow] = None
    block_sizes: List[int] = []

    transfer_steps = torch.zeros((B, total_len), dtype=torch.long, device=device)
    global_step = 0
    nfe = 0

    t = prompt_len
    block_idx = 0

    while t < total_len:
        block_idx += 1
        remaining = total_len - t

        out = model(x)
        logits = out.logits
        nfe += 1

        cache_end = min(t + 2 * L_max, total_len)
        cur_cache = topk_cache_window_from_logits(
            logits[0, t:cache_end], K=topk_cache, start=t
        )

        L_window = min(L_max, remaining)

        if block_idx == 1:
            L = min(L_min, remaining)
        else:
            p0 = t - cur_cache.start
            q0 = t - base_cache.start
            p_ids = cur_cache.ids[p0:p0 + L_window]
            p_logp = cur_cache.logp[p0:p0 + L_window]
            q_ids = base_cache.ids[q0:q0 + L_window]
            q_logp = base_cache.logp[q0:q0 + L_window]
            q_floor = base_cache.logp_floor[q0:q0 + L_window]

            I = approx_kl_topk_batched(p_ids, p_logp, q_ids, q_logp, q_floor).to(torch.float64)

            H = entropy_from_logits_batched(logits[0, t:t + L_window]).to(torch.float64)

            def smooth_centered_mean(S: torch.Tensor, W: int) -> torch.Tensor:
                """
                Centered moving average with zero padding.
                S: (L,)
                W: odd window size (e.g., 3, 5)
                return: (L,)
                """
                assert W % 2 == 1, "Window size W must be odd"
                r = W // 2

                kernel = torch.ones(W, device=S.device, dtype=S.dtype) / W

                S_pad = F.pad(S[None, None, :], (r, r))

                S_smooth = F.conv1d(S_pad, kernel[None, None, :])

                return S_smooth[0, 0]

            I_smooth = smooth_centered_mean(I, W=3)
            H_smooth = smooth_centered_mean(H, W=3)
            I_t = robust_norm(I_smooth)
            H_t = robust_norm(H_smooth)
            S = I_t - lambda_u * H_t

            pos_mask = S >= 0
            if pos_mask.all():
                L = S.size(0)
            else:
                L = int((~pos_mask).to(torch.long).argmax().item())

            if remaining < L_min:
                L = remaining
            else:
                L = max(L, L_min)
                L = min(L, L_max, remaining)

        block_sizes.append(int(L))

        base_cache = cur_cache

        block_start = t
        block_end = t + L
        i = 0

        while True:
            global_step += 1

            mask_index = (x == mask_id)
            mask_index[:, block_end:] = 0

            if i > 0:
                logits = model(x).logits
                nfe += 1

            x0, transfer_index = get_transfer_index(
                logits,
                temperature,
                remasking,
                mask_index,
                x,
                None,
                tau_low=tau_low,
                tau_high=tau_high,
                gamma=gamma,
            )

            x[transfer_index] = x0[transfer_index]
            transfer_steps[transfer_index] = global_step
            i += 1

            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break

        t = block_end

    return x, nfe, transfer_steps[:, prompt_len:], block_sizes

@torch.no_grad()
def generate_adaptive_block_with_prefix_cache(
    model,
    prompt,
    gen_length=128,
    L_max: int = 128,
    L_min: int = 8,
    temperature=0.,
    remasking='low_confidence',
    mask_id=126336,
    lambda_u: float = 1.0,
    topk_cache: int = 64,
    tau_low: float = 0.8,
    tau_high: float = 0.95,
    gamma: float = -16.0,
):
    device = model.device
    B = prompt.shape[0]

    prompt_len = int(prompt.shape[1])
    total_len = int(prompt_len + gen_length)

    x = torch.full((B, total_len), mask_id, dtype=torch.long).to(device)
    x[:, :prompt_len] = prompt.clone()

    assert 1 <= L_min <= L_max
    assert B == 1

    base_cache: Optional[TopKCacheWindow] = None
    block_sizes: List[int] = []

    transfer_steps = torch.zeros((B, total_len), dtype=torch.long, device=device)
    global_step = 0
    nfe = 0

    t = prompt_len
    block_idx = 0

    while t < total_len:
        block_idx += 1
        remaining = total_len - t

        out = model(x, use_cache=True)
        logits = out.logits
        past_key_values = out.past_key_values
        nfe += 1

        cache_end = min(t + 2 * L_max, total_len)
        cur_cache = topk_cache_window_from_logits(
            logits[0, t:cache_end], K=topk_cache, start=t
        )

        L_window = min(L_max, remaining)

        if block_idx == 1:
            L = min(L_min, remaining)
        else:
            p0 = t - cur_cache.start
            q0 = t - base_cache.start
            p_ids = cur_cache.ids[p0:p0 + L_window]
            p_logp = cur_cache.logp[p0:p0 + L_window]
            q_ids = base_cache.ids[q0:q0 + L_window]
            q_logp = base_cache.logp[q0:q0 + L_window]
            q_floor = base_cache.logp_floor[q0:q0 + L_window]

            I = approx_kl_topk_batched(
                p_ids, p_logp, q_ids, q_logp, q_floor
            ).to(torch.float64)
            H = entropy_from_logits_batched(
                logits[0, t:t + L_window]
            ).to(torch.float64)

            def smooth_centered_mean(S: torch.Tensor, W: int) -> torch.Tensor:
                assert W % 2 == 1, "Window size W must be odd"
                r = W // 2
                kernel = torch.ones(W, device=S.device, dtype=S.dtype) / W
                S_pad = F.pad(S[None, None, :], (r, r))
                S_smooth = F.conv1d(S_pad, kernel[None, None, :])
                return S_smooth[0, 0]

            I_smooth = smooth_centered_mean(I, W=3)
            H_smooth = smooth_centered_mean(H, W=3)
            I_t = robust_norm(I_smooth)
            H_t = robust_norm(H_smooth)
            S = I_t - lambda_u * H_t

            pos_mask = S >= 0
            if pos_mask.all():
                L = S.size(0)
            else:
                L = int((~pos_mask).to(torch.long).argmax().item())

            if remaining < L_min:
                L = remaining
            else:
                L = max(L, L_min)
                L = min(L, L_max, remaining)

        block_sizes.append(int(L))
        base_cache = cur_cache

        block_start = t
        block_end = t + L

        trimmed_past_key_values = []
        for layer_idx in range(len(past_key_values)):
            trimmed_past_key_values.append(())
            for kv_idx in range(len(past_key_values[layer_idx])):
                trimmed_past_key_values[layer_idx] += (
                    past_key_values[layer_idx][kv_idx][:, :, :block_start],
                )
        past_key_values = trimmed_past_key_values

        global_step += 1
        mask_index = (x == mask_id)
        mask_index[:, block_end:] = 0
        x0, transfer_index = get_transfer_index(
            logits,
            temperature,
            remasking,
            mask_index,
            x,
            None,
            tau_low=tau_low,
            tau_high=tau_high,
            gamma=gamma,
        )
        x[transfer_index] = x0[transfer_index]
        transfer_steps[transfer_index] = global_step

        while (x[:, block_start:block_end] == mask_id).sum() > 0:
            global_step += 1

            mask_index = (x[:, block_start:] == mask_id)
            mask_index[:, L:] = 0

            logits = model(
                x[:, block_start:],
                past_key_values=past_key_values,
                use_cache=True,
            ).logits
            nfe += 1

            x0, transfer_index = get_transfer_index(
                logits,
                temperature,
                remasking,
                mask_index,
                x[:, block_start:],
                None,
                tau_low=tau_low,
                tau_high=tau_high,
                gamma=gamma,
            )
            x[:, block_start:][transfer_index] = x0[transfer_index]
            transfer_steps[:, block_start:][transfer_index] = global_step

        t = block_end

    return x, nfe, transfer_steps[:, prompt_len:], block_sizes


@torch.no_grad()
def generate_adaptive_block_with_dual_cache(
    model,
    prompt,
    gen_length=128,
    L_max: int = 128,
    L_min: int = 8,
    temperature=0.,
    remasking='low_confidence',
    mask_id=126336,
    lambda_u: float = 1.0,
    topk_cache: int = 64,
    tau_low: float = 0.8,
    tau_high: float = 0.95,
    gamma: float = -16.0,
):
    device = model.device
    B = prompt.shape[0]

    prompt_len = int(prompt.shape[1])
    total_len = int(prompt_len + gen_length)

    x = torch.full((B, total_len), mask_id, dtype=torch.long).to(device)
    x[:, :prompt_len] = prompt.clone()

    assert 1 <= L_min <= L_max
    assert B == 1

    base_cache: Optional[TopKCacheWindow] = None
    block_sizes: List[int] = []

    transfer_steps = torch.zeros((B, total_len), dtype=torch.long, device=device)
    global_step = 0
    nfe = 0

    t = prompt_len
    block_idx = 0

    while t < total_len:
        block_idx += 1
        remaining = total_len - t

        out = model(x, use_cache=True)
        logits = out.logits
        past_key_values = out.past_key_values
        nfe += 1

        cache_end = min(t + 2 * L_max, total_len)
        cur_cache = topk_cache_window_from_logits(
            logits[0, t:cache_end], K=topk_cache, start=t
        )

        L_window = min(L_max, remaining)

        if block_idx == 1:
            L = min(L_min, remaining)
        else:
            p0 = t - cur_cache.start
            q0 = t - base_cache.start
            p_ids = cur_cache.ids[p0:p0 + L_window]
            p_logp = cur_cache.logp[p0:p0 + L_window]
            q_ids = base_cache.ids[q0:q0 + L_window]
            q_logp = base_cache.logp[q0:q0 + L_window]
            q_floor = base_cache.logp_floor[q0:q0 + L_window]

            I = approx_kl_topk_batched(
                p_ids, p_logp, q_ids, q_logp, q_floor
            ).to(torch.float64)
            H = entropy_from_logits_batched(
                logits[0, t:t + L_window]
            ).to(torch.float64)

            def smooth_centered_mean(S: torch.Tensor, W: int) -> torch.Tensor:
                assert W % 2 == 1, "Window size W must be odd"
                r = W // 2
                kernel = torch.ones(W, device=S.device, dtype=S.dtype) / W
                S_pad = F.pad(S[None, None, :], (r, r))
                S_smooth = F.conv1d(S_pad, kernel[None, None, :])
                return S_smooth[0, 0]

            I_smooth = smooth_centered_mean(I, W=3)
            H_smooth = smooth_centered_mean(H, W=3)
            I_t = robust_norm(I_smooth)
            H_t = robust_norm(H_smooth)
            S = I_t - lambda_u * H_t

            pos_mask = S >= 0
            if pos_mask.all():
                L = S.size(0)
            else:
                L = int((~pos_mask).to(torch.long).argmax().item())

            if remaining < L_min:
                L = remaining
            else:
                L = max(L, L_min)
                L = min(L, L_max, remaining)

        block_sizes.append(int(L))
        base_cache = cur_cache

        block_start = t
        block_end = t + L

        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, block_start:block_end] = True

        global_step += 1
        mask_index = (x == mask_id)
        mask_index[:, block_end:] = 0
        x0, transfer_index = get_transfer_index(
            logits,
            temperature,
            remasking,
            mask_index,
            x,
            None,
            tau_low=tau_low,
            tau_high=tau_high,
            gamma=gamma,
        )
        x[transfer_index] = x0[transfer_index]
        transfer_steps[transfer_index] = global_step

        while (x[:, block_start:block_end] == mask_id).sum() > 0:
            global_step += 1

            logits_blk = model(
                x[:, block_start:block_end],
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=replace_position,
            ).logits
            nfe += 1

            mask_blk = (x[:, block_start:block_end] == mask_id)
            x0_blk, transfer_idx_blk = get_transfer_index(
                logits_blk,
                temperature,
                remasking,
                mask_blk,
                x[:, block_start:block_end],
                None,
                tau_low=tau_low,
                tau_high=tau_high,
                gamma=gamma,
            )

            blk_old = x[:, block_start:block_end]
            blk_new = torch.where(transfer_idx_blk, x0_blk, blk_old)
            x = torch.cat([x[:, :block_start], blk_new, x[:, block_end:]], dim=1)

            step_old = transfer_steps[:, block_start:block_end]
            step_fill = torch.full_like(step_old, global_step)
            step_new = torch.where(transfer_idx_blk, step_fill, step_old)
            transfer_steps = torch.cat(
                [transfer_steps[:, :block_start], step_new, transfer_steps[:, block_end:]],
                dim=1,
            )

        t = block_end

    return x, nfe, transfer_steps[:, prompt_len:], block_sizes

def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,  # (B, L) bool
    x: torch.Tensor,  # (B, L) long
    num_transfer_tokens,  # (B,) or (B,1) long tensor, or None when threshold is used
    tau_low: float = 0.8,
    tau_high: float = 0.95,
    gamma: float = -16.0,  # conflict threshold
):
    """
    Returns:
        x0: (B, L) long - proposed tokens
        transfer_index: (B, L) bool - positions to update in this step
    """
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == "low_confidence":
        log_p = F.log_softmax(logits.to(torch.float64), dim=-1)
        x0_logp = torch.gather(log_p, -1, x0.unsqueeze(-1)).squeeze(-1)
        confidence = torch.exp(x0_logp)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)

    neg_inf = torch.finfo(confidence.dtype).min
    confidence = torch.where(
        mask_index,
        confidence,
        torch.full_like(confidence, neg_inf)
    )

    candidate_mask = mask_index & (confidence >= tau_low)
    transfer_index = torch.zeros_like(candidate_mask)
    if candidate_mask.any():
        idx = candidate_mask.nonzero(as_tuple=False)[:, 1]
        n = idx.shape[0]

        log_p_sub = log_p[:, idx, :]
        x0_sub = x0[:, idx]
        conf_sub = confidence[:, idx]

        xj = x0_sub.unsqueeze(1).expand(-1, n, -1)

        log_p_i_xj = torch.gather(
            log_p_sub.unsqueeze(2).expand(-1, -1, n, -1),
            -1,
            xj.unsqueeze(-1)
        ).squeeze(-1)

        D = log_p_i_xj + log_p_i_xj.transpose(1, 2)
        conflict = (D > gamma)
        eye = torch.eye(n, device=conflict.device, dtype=torch.bool).unsqueeze(0)
        conflict = conflict & (~eye)

        selected_sub = torch.zeros_like(conf_sub, dtype=torch.bool)

        high_mask = conf_sub >= tau_high
        selected_sub |= high_mask

        remaining = torch.ones_like(selected_sub, dtype=torch.bool)
        remaining &= ~selected_sub

        if high_mask.any():
            high_conflict = (conflict & high_mask.unsqueeze(1)).any(dim=2)
            remaining &= ~high_conflict

        while remaining.any():
            masked_conf = torch.where(
                remaining,
                conf_sub,
                torch.full_like(conf_sub, neg_inf)
            )

            max_idx = torch.argmax(masked_conf, dim=1, keepdim=True)

            new_select = torch.zeros_like(selected_sub).scatter_(1, max_idx, True)

            selected_sub |= new_select

            remaining &= ~new_select
            conflict_with_new = (conflict & new_select.unsqueeze(1)).any(dim=2)

            remaining &= ~conflict_with_new

        transfer_index[:, idx] = selected_sub

    max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)
    force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)

    transfer_index = transfer_index | force_mask

    transfer_index = transfer_index & mask_index

    return x0, transfer_index

    



def main():
    device = 'cuda'

    model = LLaDAModelLM.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    prompt = (
        "Lily can run 12 kilometers per hour for 4 hours. After that, she runs "
        "6 kilometers per hour. How many kilometers can she run in 8 hours?"
    )

    m = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    with torch.inference_mode():
        nvtx.range_push("INFER")

        out = generate(
            model,
            input_ids,
            steps=128,
            gen_length=128,
            block_length=32,
            temperature=0.,
            remasking='low_confidence',
        )

        torch.cuda.synchronize()
        nvtx.range_pop()
    print(tokenizer.batch_decode(out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0])

if __name__ == '__main__':
    main()
