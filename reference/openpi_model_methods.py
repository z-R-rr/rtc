"""RTC method excerpts from the source OpenPI Pi0 class, for reading/porting only.

Requires the original model initialization, embed_prefix, make_attn_mask,
posemb_sincos, OpenPI/Flax modules, and parameter names. This is NOT an
independently importable model or a tested drop-in mixin. See IMPLEMENTATION.md.
The method bodies below are preserved verbatim from the reviewed source.
"""

# Original module-level symbols needed by these method bodies:
# einops, nnx, jax, jnp, override, _model, rtc, at,
# posemb_sincos and make_attn_mask.

class Pi0RTCMethodExcerpts:
    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, "b *t"],  # noqa: F821
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b *t emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        if timestep.ndim not in (1, 2):
            raise ValueError("timestep must be [B] or [B,H]")
        if timestep.ndim == 2 and (not self.pi05 or timestep.shape != noisy_actions.shape[:2]):
            raise ValueError("Per-action time requires pi05 and matching [B,H]")
        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def flow_errors(self, rng, observation, actions, *, train=False, rtc_delay=None, noise=None, time=None):
        """Elementwise errors and active token mask; RTC is independent of augmentation mode."""
        preprocess_rng, noise_rng, time_rng, delay_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        batch_shape = actions.shape[:-2]
        if noise is None:
            noise = jax.random.normal(noise_rng, actions.shape)
        if time is None:
            if self.simulated_delay is not None and self.rtc_time_distribution == "uniform":
                time = jax.random.uniform(time_rng, batch_shape)
            else:
                # Preserve the official OpenPI pi0.5 time distribution by default.
                time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        active = jnp.ones(actions.shape[:-1], dtype=jnp.bool_)
        if self.simulated_delay is not None:
            delay = rtc.draw_delay(delay_rng, batch_shape, self.simulated_delay, self.simulated_delay_weights)
            if rtc_delay is not None:
                delay = jnp.broadcast_to(jnp.asarray(rtc_delay, dtype=jnp.int32), batch_shape)
            prefix = jnp.arange(self.action_horizon) < delay[..., None]
            model_time = jnp.where(prefix, 0.0, time[..., None])
            x_t = model_time[..., None] * noise + (1 - model_time[..., None]) * actions
            active = ~prefix
        else:
            if rtc_delay is not None:
                raise ValueError("rtc_delay requires an RTC model")
            model_time = time
            x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
        if observation.action_is_pad is not None:
            active = active & ~observation.action_is_pad
        u_t = noise - actions
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, model_time)
        if self.simulated_delay is not None and observation.action_is_pad is not None:
            # Padding must not be visible as clean prefix context or noisy keys.
            # Loss masking alone still leaks repeated end-of-episode targets into valid tokens.
            suffix_mask = suffix_mask & ~observation.action_is_pad
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return jnp.square(v_t - u_t), active

    @override
    def compute_loss(self, rng, observation, actions, *, train=False, rtc_delay=None, noise=None, time=None):
        errors, active = self.flow_errors(
            rng, observation, actions, train=train, rtc_delay=rtc_delay, noise=noise, time=time
        )
        if self.simulated_delay is None:
            return jnp.mean(errors, axis=-1)
        # The trainer takes mean([B,H]). Scale globally, NEVER per sample.
        # Thus mean(loss) == sum(error over active B,H,D) / count(active B,H).
        return rtc.loss_grid(errors, active, self.rtc_loss_reduction)

    def rtc_validation_sums(self, rng, observation, actions, *, delay, raw_action_dim):
        """Accumulate numerator/count across batches before dividing, avoiding mean-of-means."""
        errors, active = self.flow_errors(rng, observation, actions, train=False, rtc_delay=delay)
        masked = jnp.where(active[..., None], errors, 0.0)
        return {
            "squared_error": jnp.sum(masked),
            "raw_squared_error": jnp.sum(masked[..., :raw_action_dim]),
            "active_tokens": jnp.sum(active),
        }

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        rtc_prefix: at.Float[at.Array, "b ah ad"] | None = None,
        rtc_delay: at.Int[at.Array, "b"] | None = None,  # noqa: F821
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        if rtc_prefix is not None:
            if self.simulated_delay is None or rtc_delay is None:
                raise ValueError("Hard prefix sampling requires an RTC model and rtc_delay")
            if rtc_prefix.shape != noise.shape or rtc_delay.shape != (batch_size,):
                raise ValueError("RTC prefix must be [B,H,D], delay must be [B]")
            prefix_mask_rtc = jnp.arange(self.action_horizon)[None, :] < rtc_delay[:, None]
        else:
            if rtc_delay is not None:
                raise ValueError("rtc_delay requires rtc_prefix")
            prefix_mask_rtc = jnp.zeros((batch_size, self.action_horizon), dtype=jnp.bool_)
            rtc_prefix = jnp.zeros_like(noise)

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            x_t = jnp.where(prefix_mask_rtc[..., None], rtc_prefix, x_t)
            model_time = jnp.broadcast_to(time, (batch_size,))
            if self.simulated_delay is not None:
                model_time = jnp.where(prefix_mask_rtc, 0.0, model_time[:, None])
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, model_time)
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            updated = x_t + dt * v_t
            return jnp.where(prefix_mask_rtc[..., None], rtc_prefix, updated), time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
