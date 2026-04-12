import copy
import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils import ModelOutput, is_torchdynamo_compiling, logging

logger = logging.get_logger(__name__)


# Code for DepCap

def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
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
    logZ = torch.logsumexp(logits_f, dim=-1)                  # (S,)
    p = torch.exp(logits_f - logZ.unsqueeze(-1))              # (S,V)
    H = logZ - torch.sum(p * logits_f, dim=-1)                # (S,)
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
    # topk over logits (S,V) -> (S,K)
    logits_f = logits_2d.to(torch.float64)
    topk_logits, ids = torch.topk(logits_f, k=K, dim=-1)

    # one logZ per position (S,)
    logZ = torch.logsumexp(logits_f, dim=-1)
    logp = topk_logits.to(torch.float64) - logZ.unsqueeze(-1)  # (S,K)

    topk_logmass = torch.logsumexp(logp, dim=-1)  # (S,)
    topk_mass = torch.exp(topk_logmass).clamp(max=1 - 1e-12)
    rem_logmass = torch.log1p(-topk_mass)         # (S,)

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
    # p_prob (S,K)
    p_prob = torch.exp(p_logp)

    # eq: (S,K,K) where p_ids match q_ids
    eq = (p_ids.unsqueeze(2) == q_ids.unsqueeze(1))
    matched = eq.any(dim=2)  # (S,K)

    # For matched ones, pick corresponding q_logp by masked sum over K
    # This assumes at most one match; ids are unique within each top-k set.
    q_logp_matched = (eq.to(p_logp.dtype) * q_logp.unsqueeze(1)).sum(dim=2)  # (S,K)

    q_floor = q_logp_floor.unsqueeze(1).expand_as(p_logp)  # (S,K)
    q_lp = torch.where(matched, q_logp_matched, q_floor)   # (S,K)

    kl = torch.sum(p_prob * (p_logp - q_lp), dim=1)        # (S,)
    return kl

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


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None
    nfe_history: Optional[list] = None


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", 'origin')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass

class DreamGenerationMixin:
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # Do not call torch.repeat_interleave if expand_size is 1 because it clones
        # the input tensor and thus requires more memory although no change is applied
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        # Can't throw warnings/exceptions during compilation
        if is_torchdynamo_compiling():
            return

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        elif has_default_max_length:
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DreamGenerationConfig], **kwargs: Dict
    ) -> DreamGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        using_model_generation_config = False
        if generation_config is None:
            generation_config = DreamGenerationConfig.from_model_config(self.config)
            using_model_generation_config = True

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled -- an
        # exception will be raised in `_validate_model_kwargs`
        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            _kwargs = generation_config.update(**kwargs)
            # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.generation_config.mask_token_id

        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: DreamGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.
        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config = self._prepare_generation_config(generation_config, **kwargs)

        # 2. Define model inputs
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        
        # 4. Check input_ids
        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )
        if (
            hasattr(generation_config, "pad_token_id") and
            torch.any(input_ids == generation_config.pad_token_id) and 
            attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask 
        )
        threshold = kwargs.get("threshold", 0.9)
        dual_cache = kwargs.get("dual_cache", False)
        L_min = kwargs.get("L_min", 8)
        L_max = kwargs.get("L_max", 128)
        lambda_u = kwargs.get("lambda_u", 1.2)
        topk_cache = kwargs.get("topk_cache", 64)
        tau_low = kwargs.get("tau_low", 0.8)
        tau_high = kwargs.get("tau_high", 0.95)
        gamma = kwargs.get("gamma", -16.0)

        result = self._sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            dual_cache=dual_cache,
            L_min=L_min,
            L_max=L_max,
            lambda_u=lambda_u,
            topk_cache=topk_cache,
            tau_low=tau_low,
            tau_high=tau_high,
            gamma=gamma
        )
        return result


    def _sample_parallel_depcap(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        threshold: Optional[float] = None,
        dual_cache: bool = False,
        L_min: Optional[int] = None,
        L_max: Optional[int] = None,
        lambda_u: Optional[float] = None,
        topk_cache: Optional[int] = None,
        tau_low: Optional[float] = 0.8,
        tau_high: Optional[float] = 0.95,
        gamma: Optional[float] = -16.0,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        '''
        Semi-Autoregressive Dream
        '''
        # init values
        # print(f"Using BLOCK GENERATION W/ PARALLEL")
        # Note: dual_cache parameter is ignored in parallel mode

        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        alg = generation_config.alg

        histories = [] if (return_dict_in_generate and output_history) else None
        nfe_history = []

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        gen_length = max_length - input_ids.shape[1]
        
        prompt_len = int(input_ids.shape[1])
        total_len = int(prompt_len + gen_length)

        # absolute position -> TopKDist
        base_cache: Optional[TopKCacheWindow] = None
        # record chosen block sizes
        block_sizes: List[int] = []

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        # Initialize cache for the prompt
        # past_key_values = None
        t = prompt_len
        block_idx = 0

        # Process each block
        while t < total_len:
            block_idx += 1
            remaining = total_len - t
            # -------------------------------a forward before decoding each block-------------------------
            # Prepare attention mask for cached generation
            if attention_mask != "full":
                # Adjust attention mask for current position
                current_attention_mask = attention_mask[:, :, :, t:]
            else:
                current_attention_mask = attention_mask

            model_output = self(x, current_attention_mask, tok_idx if tok_idx is not None else None)
            
            logits = model_output.logits
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
            
            block_nfe = 1
            # -----------------------------------------------------
            
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

            current_block_start = t
            current_block_end = t + L
            i = 0
            while True:
                # mask_index = (x == mask_token_id)
                mask_index = (x == mask_token_id)
                mask_index[:, current_block_end:] = False
                if i > 0:
                    # Prepare attention mask for cached generation
                    if attention_mask != "full":
                        # Adjust attention mask for current position
                        current_attention_mask = attention_mask[:, :, :, current_block_start:]
                    else:
                        current_attention_mask = attention_mask

                    model_output = self(x, current_attention_mask, tok_idx if tok_idx is not None else None)
                    
                    logits = model_output.logits
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                    block_nfe += 1

                if alg == "confidence_threshold":
                    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                    x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L), long
                    log_p = F.log_softmax(logits.to(torch.float64), dim=-1)
                    x0_logp = torch.gather(log_p, -1, x0.unsqueeze(-1)).squeeze(-1)
                    confidence = torch.exp(x0_logp)
                    
                    # Only modify masked spots; keep others as original x and set their confidence to -inf.
                    x0 = torch.where(mask_index, x0, x)

                    neg_inf = torch.finfo(confidence.dtype).min
                    confidence = torch.where(
                        mask_index,
                        confidence,
                        torch.full_like(confidence, neg_inf),
                    )  # (B, L)

                
                    candidate_mask = confidence >= tau_low
                    transfer_index = torch.zeros_like(candidate_mask)

                    if candidate_mask.any():
                        # B == 1 here, so all candidates are in the same batch item.
                        idx = candidate_mask.nonzero(as_tuple=False)[:, 1]  # (n,)
                        n = idx.shape[0]

                        log_p_sub = log_p[:, idx, :]      # (B, n, V)
                        x0_sub = x0[:, idx]               # (B, n)
                        conf_sub = confidence[:, idx]     # (B, n)

                        # Build D_ij.
                        xj = x0_sub.unsqueeze(1).expand(-1, n, -1)  # (B, n, n)

                        log_p_i_xj = torch.gather(
                            log_p_sub.unsqueeze(2).expand(-1, -1, n, -1),
                            -1,
                            xj.unsqueeze(-1)
                        ).squeeze(-1)  # (B, n, n)

                        D = log_p_i_xj + log_p_i_xj.transpose(1, 2)  # (B, n, n)
                        conflict = (D > gamma)  # (B, n, n)
                        eye = torch.eye(n, device=conflict.device, dtype=torch.bool).unsqueeze(0)
                        conflict = conflict & (~eye)

                        # -------------------------------------------------
                        # 4) selection on subgraph
                        # -------------------------------------------------
                        selected_sub = torch.zeros_like(conf_sub, dtype=torch.bool)

                        high_mask = conf_sub >= tau_high
                        selected_sub |= high_mask

                        remaining = torch.ones_like(selected_sub, dtype=torch.bool)
                        remaining &= ~selected_sub

                        # Remove positions that conflict with selected ones.
                        if high_mask.any():
                            high_conflict = (conflict & high_mask.unsqueeze(1)).any(dim=2)
                            remaining &= ~high_conflict

                        # 4.2 greedy
                        while remaining.any():

                            masked_conf = torch.where(
                                remaining,
                                conf_sub,
                                torch.full_like(conf_sub, neg_inf)
                            )

                            max_idx = torch.argmax(masked_conf, dim=1, keepdim=True)

                            new_select = torch.zeros_like(selected_sub).scatter_(1, max_idx, True)

                            selected_sub |= new_select

                            # Explicitly remove the newly selected position.
                            remaining &= ~new_select
                            conflict_with_new = (
                                conflict & new_select.unsqueeze(1)
                            ).any(dim=2)

                            remaining &= ~conflict_with_new

                        # Scatter back to original positions.
                        transfer_index[:, idx] = selected_sub

                    # at least one token is transferred "always unmask max c^i"
                    max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)  # (B, 1)
                    force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)

                    # (Above Threshold) OR (Is Max Confidence)
                    transfer_index = transfer_index | force_mask

                    # Safety: do not unmask something that was not masked (consider fully unmasked rows)
                    transfer_index = transfer_index & mask_index

                    
                    x[transfer_index] = x0[transfer_index]
                i += 1
                if (x[:, current_block_start:current_block_end] == mask_token_id).sum() == 0:
                    break

            # Record block statistics
            nfe_history.append(block_nfe)
            t = current_block_end
        
        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
                nfe_history=nfe_history,
            ), block_sizes
        else:
            return x, block_sizes
        

    
    def _sample_cache(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        threshold: Optional[float] = 0.9,
        block_length: Optional[int] = 32,
        dual_cache: bool = False,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        '''
        Original Fast-dLLM Implementation
        '''
        pass
    # TODO
        
