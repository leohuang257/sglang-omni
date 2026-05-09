# SPDX-License-Identifier: Apache-2.0
"""Z-Image diffusion backend with semantic + ByT5 text encoding.

Loads the ZImage pipeline components (transformer, VAE, scheduler) and
optionally the MingSemanticEncoder (LLM + connector) and/or ByT5 text
encoder.  Semantic conditioning (LLM-derived) produces meaningful images;
ByT5 provides supplementary text rendering control.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import torch
from PIL import Image

from sglang_omni.models.ming_omni.diffusion.backend import (
    DiffusionBackend,
    ImageGenParams,
)

logger = logging.getLogger(__name__)

# Patterns for extracting quoted text from prompts (matching
# processing_bailingmm2.py:get_text_from_prompt).
_QUOTE_PATTERNS = [
    r"\"(.*?)\"",  # straight double quotes
    r"\u201c(.*?)\u201d",  # curly double quotes ""
    r"\u2018(.*?)\u2019",  # curly single quotes ''
]


def _extract_render_text(prompt: str) -> str:
    """Extract text-to-render from a prompt by finding quoted substrings.

    Mirrors Ming's ``processing_bailingmm2.py:get_text_from_prompt``.
    Returns the last quoted substring found, or empty string if none.
    """
    texts: list[str] = []
    for pattern in _QUOTE_PATTERNS:
        texts.extend(re.findall(pattern, prompt))
    return texts[-1] if texts else ""


class ZImageBackend(DiffusionBackend):
    """Z-Image diffusion backend with semantic text conditioning.

    With ``sp_size > 1``, runs the transformer under Ulysses-style sequence
    parallelism (diffusers Context Parallel): every rank holds full weights,
    the unified token sequence is sharded across ranks inside attention.
    Leader drives the request; followers loop in :meth:`sp_follower_serve`.
    """

    def __init__(self) -> None:
        self._pipe = None
        self._text_encoder = None  # ByT5 (supplementary)
        self._tokenizer = None  # ByT5 tokenizer
        self._semantic_encoder = None  # Ming LLM + connector (primary)
        self._device: torch.device | None = None

        # Sequence-parallel state (sp_size == 1 means SP disabled).
        self._sp_size: int = 1
        self._sp_rank: int = 0
        self._sp_group = None  # torch.distributed ProcessGroup for SP ranks

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models(
        self,
        model_path: str,
        device: torch.device,
        *,
        skip_semantic_encoder: bool = False,
        sp_size: int = 1,
        sp_rank: int = 0,
        nccl_port: int | None = None,
    ) -> None:
        self._device = device
        self._sp_size = max(int(sp_size), 1)
        self._sp_rank = int(sp_rank)

        if self._sp_size > 1:
            self._init_sp_distributed(nccl_port)

        from diffusers import (
            AutoencoderKL,
            FlowMatchEulerDiscreteScheduler,
            ZImagePipeline,
            ZImageTransformer2DModel,
        )

        logger.info(
            "[ZImage] Loading pipeline components from %s (sp_size=%d, sp_rank=%d)",
            model_path,
            self._sp_size,
            self._sp_rank,
        )

        # 1. Scheduler
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
        scheduler.config["use_dynamic_shifting"] = True

        # 2. VAE
        vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae", torch_dtype=torch.bfloat16
        )

        # 3. Transformer (ZImageTransformer2DModel)
        transformer = ZImageTransformer2DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        logger.info(
            "[ZImage] Transformer loaded (cap_feat_dim=%d)",
            transformer.config.cap_feat_dim,
        )

        # 4. Assemble pipeline (text encoding handled separately)
        self._pipe = ZImagePipeline(
            scheduler=scheduler,
            vae=vae,
            transformer=transformer,
            text_encoder=None,
            tokenizer=None,
        )
        self._pipe = self._pipe.to(device)
        logger.info("[ZImage] Pipeline assembled on %s", device)

        if self._sp_size > 1:
            self._enable_sp_on_transformer()

        # 5. Load semantic encoder (LLM + connector) — primary
        if skip_semantic_encoder:
            logger.info(
                "[ZImage] Skipping semantic encoder loading "
                "(skip_semantic_encoder=True)"
            )
            self._semantic_encoder = None
        else:
            try:
                from sglang_omni.models.ming_omni.diffusion.semantic_encoder import (
                    MingSemanticEncoder,
                )

                self._semantic_encoder = MingSemanticEncoder()
                self._semantic_encoder.load(model_path, device)
                logger.info("[ZImage] Semantic encoder (LLM + connector) ready")
            except Exception as e:
                logger.warning(
                    "[ZImage] Failed to load semantic encoder: %s. "
                    "Falling back to ByT5-only mode.",
                    e,
                )
                self._semantic_encoder = None

        # 6. Load ByT5 text encoder + mapper (supplementary)
        byt5_dir = Path(model_path) / "byt5"
        if byt5_dir.exists():
            from sglang_omni.models.ming_omni.diffusion.byt5_encoder import (
                load_byt5_text_encoder,
            )

            self._text_encoder, self._tokenizer = load_byt5_text_encoder(
                model_path, device, dtype=torch.bfloat16
            )
            logger.info("[ZImage] ByT5 text encoder ready")
        else:
            logger.warning(
                "[ZImage] No byt5/ directory found at %s — "
                "ByT5 text encoding will not be available.",
                model_path,
            )

    # ------------------------------------------------------------------
    # Sequence-parallel helpers
    # ------------------------------------------------------------------

    def _init_sp_distributed(self, nccl_port: int | None) -> None:
        """Initialize a flat ``torch.distributed`` world of size ``sp_size``.

        Bypasses ``parallel_state.initialize_model_parallel`` (its
        ``world_size == TP * PP`` assertion would mislabel us as TP) and
        always tears down any pre-existing process group: a stale group
        (e.g. thinker TP) is bound to other GPUs / a dead TCP store, so
        reusing it deadlocks the new SP follower even when world sizes match.
        """
        import os

        import torch.distributed as dist

        # Pin current device BEFORE any NCCL work: the communicator binds to
        # ``torch.cuda.current_device()`` at creation, so a stale current
        # device (cuda:0 default, or thinker leftover) silently hangs the
        # first all-to-all. Followers do this in sp_follower_loop.
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.set_device(self._device)

        # Always tear down — see docstring: matching world_size is not a
        # valid reuse signal across stages.
        if dist.is_initialized():
            existing_ws = dist.get_world_size()
            logger.warning(
                "[ZImage SP] torch.distributed already initialized with "
                "world_size=%d; tearing down to set up SP world_size=%d on "
                "nccl_port=%s",
                existing_ws,
                self._sp_size,
                nccl_port,
            )
            # Drop any SGLang TP state first, then the default process group.
            try:
                from sglang.srt.distributed import parallel_state

                if parallel_state.model_parallel_is_initialized():
                    parallel_state.destroy_model_parallel()
                parallel_state.destroy_distributed_environment()
            except Exception as exc:
                logger.warning(
                    "[ZImage SP] SGLang parallel_state teardown failed: %s",
                    exc,
                )
            if dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except Exception as exc:
                    logger.warning("[ZImage SP] destroy_process_group failed: %s", exc)

        local_rank = self._sp_rank
        if self._device is not None and self._device.type == "cuda":
            local_rank = self._device.index if self._device.index is not None else 0

        # Explicit tcp:// on our own port; clear MASTER_* so we don't pick up
        # a previous stage's stale store and connect to dead ranks.
        init_method = (
            f"tcp://127.0.0.1:{nccl_port}" if nccl_port is not None else "env://"
        )
        for stale_var in ("MASTER_ADDR", "MASTER_PORT"):
            os.environ.pop(stale_var, None)
        os.environ["LOCAL_RANK"] = str(local_rank)

        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            world_size=self._sp_size,
            rank=self._sp_rank,
        )

        # WORLD is the SP group; diffusers builds its DeviceMesh from it.
        self._sp_group = dist.group.WORLD
        logger.info(
            "[ZImage SP] torch.distributed initialized: rank=%d/%d",
            self._sp_rank,
            self._sp_size,
        )

    def _enable_sp_on_transformer(self) -> None:
        """Apply Diffusers Context Parallelism (Ulysses) to the transformer.

        cp_plan (see tests/test_zimage_cp_plan.py):
          - split ``x`` once at ``layers.0`` and ``freqs_cis`` on every layer
          - gather ``all_final_layer.<key>.linear`` back to full length
          - ``attn_mask`` stays full (all-to-all restores S=full at SDPA)
          - refiners run on full sequences; their processors are detached below
        """
        from diffusers.models._modeling_parallel import (
            ContextParallelConfig,
            ContextParallelInput,
            ContextParallelOutput,
        )

        transformer = self._pipe.transformer

        # Fail at load time instead of an opaque shape error in step 0.
        n_heads = transformer.config.n_heads
        if n_heads % self._sp_size != 0:
            valid = [s for s in range(1, n_heads + 1) if n_heads % s == 0]
            raise ValueError(
                f"ZImage n_heads={n_heads} not divisible by sp_size={self._sp_size}; "
                f"valid SP degrees up to n_heads: {valid}"
            )

        plan: dict = {
            "layers.0": {
                "x": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
            },
            "layers.*": {
                "freqs_cis": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
            },
        }
        for key in transformer.all_final_layer.keys():
            plan[f"all_final_layer.{key}.linear"] = ContextParallelOutput(
                gather_dim=1, expected_dims=3
            )

        # `native` (templated SDPA) is the most portable CP-capable backend.
        transformer.set_attention_backend("native")
        cp_cfg = ContextParallelConfig(ulysses_degree=self._sp_size, ring_degree=1)
        transformer.enable_parallelism(config=cp_cfg, cp_plan=plan)

        # enable_parallelism stamps every attention processor; refiners see
        # full-length inputs (no shard plan), so opt them out manually.
        for refiner in (
            transformer.noise_refiner,
            transformer.context_refiner,
            getattr(transformer, "siglip_refiner", None),
        ):
            if refiner is None:
                continue
            for blk in refiner:
                blk.attention.processor._parallel_config = None

        logger.info(
            "[ZImage SP] Context parallelism enabled (ulysses=%d, plan=%d entries)",
            self._sp_size,
            len(plan),
        )

    def _broadcast_request_payload(self, payload: dict | None) -> dict | None:
        """Broadcast a request payload from rank 0 to all SP ranks.

        Returns the same payload object on every rank.  ``None`` is the
        poison-pill that signals followers to exit their serve loop.
        """
        from sglang.srt.utils import broadcast_pyobj

        result = broadcast_pyobj(
            [payload] if self._sp_rank == 0 else [None],
            self._sp_rank,
            self._sp_group,
            src=0,
            force_cpu_device=False,
        )
        return result[0] if result else None

    def sp_follower_serve(self) -> None:
        """Serve loop for SP rank > 0.

        Blocks waiting for broadcast payloads from rank 0; for each one,
        runs the diffusion pipeline collectively (all ranks participate
        in the NCCL collectives inside the cp_plan hooks).  Returns when
        rank 0 broadcasts ``None``.
        """
        if self._sp_rank == 0:
            raise RuntimeError("sp_follower_serve must only run on rank > 0")
        if self._pipe is None:
            raise RuntimeError("ZImage pipeline not loaded")

        step = 0
        while True:
            payload = self._broadcast_request_payload(None)
            if payload is None:
                logger.info(
                    "[ZImage SP rank %d] received stop signal after %d steps",
                    self._sp_rank,
                    step,
                )
                return
            with torch.no_grad():
                # Followers run pipe to participate in collectives but discard
                # the resulting image.
                self._run_pipe_from_payload(payload)
            step += 1

    def sp_shutdown(self) -> None:
        """Send poison pill to followers (called by leader on shutdown)."""
        if self._sp_size > 1 and self._sp_rank == 0:
            try:
                self._broadcast_request_payload(None)
            except Exception as exc:
                logger.warning("[ZImage SP] shutdown broadcast failed: %s", exc)

    def _preflight_check_sp_seq_len(
        self, prompt_embeds: list[torch.Tensor], height: int, width: int
    ) -> None:
        """Check unified seq_len % sp_size before entering the pipeline.

        Without this, the EquipartitionSharder asserts deep inside the first
        forward with an opaque "not divisible by mesh size" message. Called
        on every rank so leader and followers fail symmetrically.
        """
        if self._sp_size <= 1:
            return

        # Total spatial stride = VAE downsample x transformer patch_size (16 for ZImage).
        vae_stride = self._pipe.vae_scale_factor * 2
        img_tokens = (height // vae_stride) * (width // vae_stride)
        cap_tokens = sum(int(t.shape[0]) for t in prompt_embeds)
        seq_len = img_tokens + cap_tokens

        if seq_len % self._sp_size != 0:
            raise ValueError(
                f"ZImage SP pre-flight failed: unified_seq_len={seq_len} "
                f"(image_patches={img_tokens} from {height}x{width} @ stride={vae_stride}, "
                f"cap_tokens={cap_tokens}) is not divisible by sp_size={self._sp_size}.\n"
                f"Ulysses requires BOTH n_heads % sp == 0 AND seq_len % sp == 0. "
                f"For ZImage at this resolution, change sp_size to a divisor of "
                f"{seq_len} that also divides n_heads={self._pipe.transformer.config.n_heads} "
                f"(common working choice: sp_size=2 for 1024x1024 + 256-token caption)."
            )

    def _run_pipe_from_payload(self, payload: dict):
        """Invoke ``self._pipe`` with the broadcast payload (collective)."""
        prompt_embeds = [t.to(self._device) for t in payload["prompt_embeds"]]
        neg_embeds = [t.to(self._device) for t in payload["negative_prompt_embeds"]]

        self._preflight_check_sp_seq_len(
            prompt_embeds, payload["height"], payload["width"]
        )

        seed = payload.get("seed")
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(int(seed))
        return self._pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_embeds,
            height=payload["height"],
            width=payload["width"],
            num_inference_steps=payload["num_inference_steps"],
            guidance_scale=payload["guidance_scale"],
            generator=generator,
            max_sequence_length=payload.get("max_sequence_length", 512),
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        params: ImageGenParams,
        *,
        condition_embeds: list[torch.Tensor] | None = None,
        negative_condition_embeds: list[torch.Tensor] | None = None,
    ) -> Image.Image:
        if self._pipe is None:
            raise RuntimeError("ZImage pipeline not loaded")

        generator = None
        if params.seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(params.seed)

        # --- Build condition embeddings ---
        prompt_embeds: list[torch.Tensor]
        neg_embeds: list[torch.Tensor]

        if condition_embeds is not None:
            # Pre-computed embeddings provided (e.g., from thinker hidden states)
            prompt_embeds = condition_embeds
            neg_embeds = (
                negative_condition_embeds
                if negative_condition_embeds is not None
                else [e * 0.0 for e in condition_embeds]
            )

            # Optionally concatenate ByT5 embeddings for text rendering
            if self._text_encoder is not None and self._tokenizer is not None:
                render_text = _extract_render_text(prompt)
                if render_text:
                    byt5_pos, byt5_neg = self._text_encoder.encode(
                        render_text,
                        tokenizer=self._tokenizer,
                        device=self._device,
                        max_length=256,
                    )
                    prompt_embeds = [
                        torch.cat([sem, byt.to(sem.device)], dim=0)
                        for sem, byt in zip(prompt_embeds, byt5_pos)
                    ]
                    neg_embeds = [
                        torch.cat([nsem, nbyt.to(nsem.device)], dim=0)
                        for nsem, nbyt in zip(neg_embeds, byt5_neg)
                    ]

        elif self._semantic_encoder is not None:
            # Semantic encoding via LLM + connector
            prompt_embeds, neg_embeds = self._semantic_encoder.encode(prompt)

            # Optionally concatenate ByT5 embeddings for text rendering.
            # ByT5 encodes only the RENDER TEXT (text between quotes in the
            # prompt), not the full scene description.  This matches Ming's
            # production flow: processing_bailingmm2.py:get_text_from_prompt
            # extracts quoted text, and encode() wraps it as 'Text "...". '.
            if self._text_encoder is not None and self._tokenizer is not None:
                render_text = _extract_render_text(prompt)
                if render_text:
                    byt5_pos, byt5_neg = self._text_encoder.encode(
                        render_text,
                        tokenizer=self._tokenizer,
                        device=self._device,
                        max_length=256,
                    )
                    prompt_embeds = [
                        torch.cat([sem, byt.to(sem.device)], dim=0)
                        for sem, byt in zip(prompt_embeds, byt5_pos)
                    ]
                    neg_embeds = [
                        torch.cat([nsem, nbyt.to(nsem.device)], dim=0)
                        for nsem, nbyt in zip(neg_embeds, byt5_neg)
                    ]

        elif self._text_encoder is not None and self._tokenizer is not None:
            # Fallback: ByT5-only encoding (text rendering mode)
            logger.warning(
                "[ZImage] Using ByT5-only encoding (no semantic encoder). "
                "Images may show text rendering instead of semantic content."
            )
            render_text = _extract_render_text(prompt) or prompt
            prompt_embeds, neg_embeds = self._text_encoder.encode(
                render_text,
                tokenizer=self._tokenizer,
                device=self._device,
                max_length=256,
            )

        else:
            # No text encoder at all — random embeddings
            logger.warning(
                "[ZImage] No text encoder — generating with random embeddings"
            )
            cap_feat_dim = self._pipe.transformer.config.cap_feat_dim
            prompt_embeds = [
                torch.randn(77, cap_feat_dim, device=self._device, dtype=torch.bfloat16)
            ]
            neg_embeds = [
                torch.zeros(77, cap_feat_dim, device=self._device, dtype=torch.bfloat16)
            ]

        if self._sp_size > 1:
            # Broadcast payload, then every rank collectively runs the pipe;
            # cp_plan hooks split/gather inside transformer forward.
            payload = {
                "prompt_embeds": [t.detach().cpu() for t in prompt_embeds],
                "negative_prompt_embeds": [t.detach().cpu() for t in neg_embeds],
                "height": params.height,
                "width": params.width,
                "num_inference_steps": params.num_inference_steps,
                "guidance_scale": params.guidance_scale,
                "seed": params.seed,
                "max_sequence_length": 512,
            }
            self._broadcast_request_payload(payload)
            result = self._run_pipe_from_payload(payload)
        else:
            result = self._pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=neg_embeds,
                height=params.height,
                width=params.width,
                num_inference_steps=params.num_inference_steps,
                guidance_scale=params.guidance_scale,
                generator=generator,
                max_sequence_length=512,
            )

        return result.images[0]

    def unload(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        if self._text_encoder is not None:
            del self._text_encoder
            self._text_encoder = None
        self._tokenizer = None
        if self._semantic_encoder is not None:
            self._semantic_encoder.unload()
            self._semantic_encoder = None
        torch.cuda.empty_cache()
