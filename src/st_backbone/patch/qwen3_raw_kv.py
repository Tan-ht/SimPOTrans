from __future__ import annotations

from typing import Optional

import torch


def patch_qwen3_raw_kv_cache() -> None:
    """Patch Transformers Qwen3 to store unrotated K/V in generation cache.

    This is opt-in and idempotent. Evaluation code calls it before model
    construction when ``llm_raw_kv_cache`` is enabled; training does not import
    this module by default.
    """
    from transformers.models.qwen3 import modeling_qwen3 as qwen3

    if getattr(qwen3.Qwen3Attention, "_simpotrans_raw_kv_patched", False):
        return

    qwen3.Qwen3Model._simpotrans_original_forward = qwen3.Qwen3Model.forward
    qwen3.Qwen3Attention._simpotrans_original_forward = qwen3.Qwen3Attention.forward
    qwen3.Qwen3Model.forward = _qwen3_model_raw_kv_forward
    qwen3.Qwen3Attention.forward = _qwen3_attention_raw_kv_forward
    qwen3.Qwen3Model._simpotrans_raw_kv_patched = True
    qwen3.Qwen3Attention._simpotrans_raw_kv_patched = True


def _qwen3_model_raw_kv_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    from transformers.models.qwen3 import modeling_qwen3 as qwen3

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = qwen3.DynamicCache(config=self.config)

    if cache_position is None:
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    if not isinstance(causal_mask_mapping := attention_mask, dict):
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": qwen3.create_causal_mask(**mask_kwargs),
        }
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = (
                qwen3.create_sliding_window_causal_mask(**mask_kwargs)
            )

    hidden_states = inputs_embeds
    query_position_embeddings = self.rotary_emb(hidden_states, position_ids)
    raw_key_position_ids = torch.arange(
        0,
        cache_position[-1].item() + 1,
        dtype=torch.long,
        device=inputs_embeds.device,
    ).unsqueeze(0)
    raw_key_position_embeddings = self.rotary_emb(hidden_states, raw_key_position_ids)

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping[decoder_layer.attention_type],
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=query_position_embeddings,
            raw_key_position_embeddings=raw_key_position_embeddings,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)
    return qwen3.BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
    )


def _qwen3_attention_raw_kv_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values=None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    from transformers.models.qwen3 import modeling_qwen3 as qwen3

    raw_key_position_embeddings = kwargs.pop("raw_key_position_embeddings", None)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states = qwen3.apply_rotary_pos_emb(
        query_states,
        query_states,
        cos,
        sin,
    )[0]

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states,
            value_states,
            self.layer_idx,
            cache_kwargs,
        )
        if raw_key_position_embeddings is None:
            raise RuntimeError(
                "Qwen3 raw KV patch requires raw_key_position_embeddings."
            )
        key_cos, key_sin = raw_key_position_embeddings
        key_states = qwen3.apply_rotary_pos_emb(
            key_states,
            key_states,
            key_cos,
            key_sin,
        )[1]
    else:
        key_states = qwen3.apply_rotary_pos_emb(
            key_states,
            key_states,
            cos,
            sin,
        )[1]

    attention_interface = qwen3.eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = qwen3.ALL_ATTENTION_FUNCTIONS[
            self.config._attn_implementation
        ]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
