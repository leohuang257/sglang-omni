# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Z-Image diffusion sequence parallelism.

These tests cover the pure-logic surface added for Ulysses-style SP:
preflight divisibility checks, follower-process GPU placement, and
executor-level guards. The actual NCCL/forward path is covered by
``tests/test_production_image_gen_e2e.py`` (requires real GPUs).
"""
from __future__ import annotations

import socket
import types
import unittest
from unittest.mock import MagicMock, patch

import torch


def _make_backend(sp_size: int, n_heads: int = 30, vae_scale_factor: int = 8):
    """Construct a ZImageBackend with just enough stubbing for arithmetic checks."""
    from sglang_omni.models.ming_omni.diffusion.zimage_backend import ZImageBackend

    backend = ZImageBackend()
    backend._sp_size = sp_size
    backend._pipe = types.SimpleNamespace(
        vae_scale_factor=vae_scale_factor,
        transformer=types.SimpleNamespace(
            config=types.SimpleNamespace(n_heads=n_heads),
        ),
    )
    return backend


class TestPreflightSeqLen(unittest.TestCase):
    """``_preflight_check_sp_seq_len``: unified_seq_len % sp_size."""

    def _embeds(self, cap_tokens: int, dim: int = 1):
        return [torch.zeros(cap_tokens, dim)]

    def test_sp_disabled_is_noop(self):
        backend = _make_backend(sp_size=1)
        backend._preflight_check_sp_seq_len(self._embeds(3), height=7, width=11)

    def test_1024_sq_sp2_passes(self):
        # img=64*64=4096, cap=256, total=4352, 4352 % 2 == 0
        backend = _make_backend(sp_size=2)
        backend._preflight_check_sp_seq_len(self._embeds(256), height=1024, width=1024)

    def test_1024_sq_sp3_raises(self):
        # 4352 % 3 == 2  → expected to fail
        backend = _make_backend(sp_size=3)
        with self.assertRaises(ValueError) as cm:
            backend._preflight_check_sp_seq_len(
                self._embeds(256), height=1024, width=1024
            )
        msg = str(cm.exception)
        self.assertIn("4352", msg)
        self.assertIn("sp_size=3", msg)
        self.assertIn("n_heads=30", msg)

    def test_caption_makes_seq_len_indivisible(self):
        # img=4096, cap=257, total=4353 (odd) → fails for sp=2
        backend = _make_backend(sp_size=2)
        with self.assertRaises(ValueError) as cm:
            backend._preflight_check_sp_seq_len(
                self._embeds(257), height=1024, width=1024
            )
        self.assertIn("cap_tokens=257", str(cm.exception))

    def test_multiple_caption_chunks_summed(self):
        # Two embed chunks of 100 + 156 = 256 cap_tokens
        backend = _make_backend(sp_size=2)
        embeds = [torch.zeros(100, 1), torch.zeros(156, 1)]
        backend._preflight_check_sp_seq_len(embeds, height=1024, width=1024)


class TestEnableSpHeadCheck(unittest.TestCase):
    """``_enable_sp_on_transformer``: n_heads % sp_size guard."""

    def test_n_heads_indivisible_raises(self):
        # n_heads=30, sp_size=4 → ValueError listing valid divisors
        backend = _make_backend(sp_size=4, n_heads=30)
        with self.assertRaises(ValueError) as cm:
            backend._enable_sp_on_transformer()
        msg = str(cm.exception)
        self.assertIn("n_heads=30", msg)
        self.assertIn("sp_size=4", msg)
        # Divisors of 30 must appear in the suggestion list
        for divisor in (1, 2, 3, 5, 6, 10, 15, 30):
            self.assertIn(str(divisor), msg)


class TestSpawnSpFollowers(unittest.TestCase):
    """``spawn_sp_followers``: process count, GPU placement, kwargs."""

    def _patched_spawn(self, **kwargs):
        """Run spawn_sp_followers with mp.get_context patched out."""
        from sglang_omni.models.ming_omni.diffusion import sp_follower

        started: list[dict] = []

        class DummyProcess:
            def __init__(self, target, args, kwargs, name, daemon):
                self.target = target
                self.args = args
                self.kwargs = kwargs
                self.name = name
                self.daemon = daemon
                self.pid = 1000 + len(started)

            def start(self):
                started.append(
                    dict(
                        target=self.target,
                        args=self.args,
                        kwargs=self.kwargs,
                        name=self.name,
                        daemon=self.daemon,
                    )
                )

        class DummyContext:
            def Process(self, target, args, kwargs, name, daemon):
                return DummyProcess(target, args, kwargs, name, daemon)

        with patch.object(sp_follower.mp, "get_context", return_value=DummyContext()):
            procs = sp_follower.spawn_sp_followers(**kwargs)
        return procs, started

    def test_sp_size_1_returns_empty(self):
        procs, started = self._patched_spawn(
            sp_size=1, base_gpu_id=0, model_path="m", nccl_port=1234
        )
        self.assertEqual(procs, [])
        self.assertEqual(started, [])

    def test_default_step_walks_upward(self):
        procs, started = self._patched_spawn(
            sp_size=3, base_gpu_id=5, model_path="m", nccl_port=29500
        )
        self.assertEqual(len(procs), 2)
        # rank, sp_size, gpu_id, model_path, nccl_port
        self.assertEqual(started[0]["args"], (1, 3, 6, "m", 29500))
        self.assertEqual(started[1]["args"], (2, 3, 7, "m", 29500))

    def test_negative_step_walks_downward(self):
        procs, started = self._patched_spawn(
            sp_size=3,
            base_gpu_id=5,
            model_path="m",
            nccl_port=29500,
            gpu_id_step=-1,
        )
        self.assertEqual(len(procs), 2)
        self.assertEqual(started[0]["args"][2], 4)
        self.assertEqual(started[1]["args"][2], 3)

    def test_followers_are_non_daemon(self):
        # daemon=True would forbid spawning grandchildren and prevent the
        # follower from receiving the leader's poison-pill cleanly.
        _, started = self._patched_spawn(
            sp_size=2, base_gpu_id=0, model_path="m", nccl_port=29500
        )
        self.assertFalse(started[0]["daemon"])

    def test_skip_semantic_encoder_kwarg_propagates(self):
        _, started = self._patched_spawn(
            sp_size=2,
            base_gpu_id=0,
            model_path="m",
            nccl_port=29500,
            skip_semantic_encoder=True,
        )
        self.assertEqual(started[0]["kwargs"], {"skip_semantic_encoder": True})


class TestResolveNcclPort(unittest.TestCase):
    def test_returns_bindable_port(self):
        from sglang_omni.models.ming_omni.diffusion.sp_follower import (
            _resolve_nccl_port,
        )

        port = _resolve_nccl_port()
        self.assertIsInstance(port, int)
        self.assertGreater(port, 0)
        # Sanity: we can immediately rebind it (i.e. it was released).
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))


class TestExecutorSpGuard(unittest.TestCase):
    """``MingImageGenExecutor`` rejects sp_size>1 for non-Z-Image backends."""

    def test_sp_size_gt_1_only_for_zimage(self):
        from sglang_omni.models.ming_omni.components import image_gen_executor

        executor = image_gen_executor.MingImageGenExecutor(
            model_path="m",
            dit_type="sd3",
            sp_size=2,
        )
        # Stub _create_backend so we don't import the real SD3Backend
        # (which would pull in heavy diffusers deps the guard never reaches).
        with patch.object(
            image_gen_executor, "_create_backend", return_value=MagicMock()
        ):
            with self.assertRaises(ValueError) as cm:
                executor._load_models()
        self.assertIn("zimage", str(cm.exception))
        self.assertIn("sd3", str(cm.exception))

    def test_sp_size_clamped_to_at_least_1(self):
        # sp_size=0 should not enable SP (avoids accidental enablement).
        from sglang_omni.models.ming_omni.components.image_gen_executor import (
            MingImageGenExecutor,
        )

        executor = MingImageGenExecutor(model_path="m", dit_type="zimage", sp_size=0)
        self.assertEqual(executor._sp_size, 1)


if __name__ == "__main__":
    unittest.main()
