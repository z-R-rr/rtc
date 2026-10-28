# Source excerpts; require original nn/nnx/jnp/at/nnx_utils definitions.
# Read alongside IMPLEMENTATION.md. Not imported by rtc_reference.

@at.typecheck
class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x, cond):
        dtype = x.dtype  # original dtype, could be half-precision
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)  # compute variance in float32
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))  # compute normalization in float32
        if cond is None:
            # regular RMSNorm
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            normed_inputs = normed_inputs * (
                1 + scale
            )  # scale by learned parameter in float32 (matches Flax implementation)
            return normed_inputs.astype(dtype), None  # return in original dtype

        # adaptive RMSNorm
        modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
        scale, shift, gate = jnp.split(modulation[:, None, :] if modulation.ndim == 2 else modulation, 3, axis=-1)
        normed_inputs = normed_inputs * (1 + scale) + shift  # scale and shift in float32
        return normed_inputs.astype(dtype), gate

class Pi0RTCConfigMethodExcerpt:
    def get_rtc_freeze_filter(self) -> nnx.filterlib.Filter:
        """Resolve the explicitly selected full or action-side LoRA RTC training mode."""
        if self.rtc_finetune_mode == "full":
            return nnx.Nothing
        if not self.pi05 or "lora" not in self.action_expert_variant:
            raise ValueError("RTC adaptation requires pi05 and an action expert LoRA variant")
        # _1 identifies the action expert in the two-expert Gemma module.
        trainable = nnx.Any(
            nnx_utils.PathRegex(r"PaliGemma/llm/.*_1/.*lora.*"),
            nnx_utils.PathRegex(
                r"PaliGemma/llm/(layers/(pre_attention_norm_1|pre_ffw_norm_1)|final_norm_1)/Dense_0/(kernel|bias)"
            ),
            nnx_utils.PathRegex(r"(time_mlp_in|time_mlp_out|action_in_proj|action_out_proj)/(kernel|bias)"),
        )
        return nnx.All(nnx.Param, nnx.Not(trainable))
