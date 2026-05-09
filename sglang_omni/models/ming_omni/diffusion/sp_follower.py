# SPDX-License-Identifier: Apache-2.0
"""Sequence-parallel follower processes for the Z-Image diffusion stage.

Mirrors the pattern used for the thinker stage in
``sglang_omni/engines/tp/follower.py``:

* The leader (rank 0) lives inside the normal ``MingImageGenExecutor``
  process, owns the request lifecycle and the ZMQ connections.
* Each follower (rank > 0) runs in its own OS process, holds a full copy
  of the diffusion components on a different GPU, and joins every NCCL
  collective issued from rank 0.
* Follower processes are spawned **before** rank 0 calls
  ``ZImageBackend.load_models``, because ``init_distributed_environment``
  is collective and would deadlock otherwise.
"""

from __future__ import annotations

import logging
import multiprocessing as mp

logger = logging.getLogger(__name__)


def sp_follower_loop(
    rank: int,
    world_size: int,
    gpu_id: int,
    model_path: str,
    nccl_port: int,
    *,
    skip_semantic_encoder: bool = False,
) -> None:
    """Entry point for an SP follower process.

    Loads the diffusion pipeline on ``cuda:{gpu_id}``, joins the SP
    process group, and serves broadcast requests from rank 0 until a
    ``None`` poison-pill is received.
    """
    import torch

    torch.cuda.set_device(gpu_id)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [SP{rank}] %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger(f"zimage_sp_follower.{rank}")
    log.info(
        "Starting SP follower on GPU %d (rank %d/%d, model=%s)",
        gpu_id,
        rank,
        world_size,
        model_path,
    )

    # Followers don't need the semantic encoder / ByT5 — only rank 0 ever
    # builds prompt embeddings; followers receive them via broadcast.
    from sglang_omni.models.ming_omni.diffusion.zimage_backend import ZImageBackend

    backend = ZImageBackend()
    backend.load_models(
        model_path,
        torch.device("cuda", gpu_id),
        skip_semantic_encoder=True,
        sp_size=world_size,
        sp_rank=rank,
        nccl_port=nccl_port,
    )

    log.info("Follower ready, entering serve loop")
    try:
        backend.sp_follower_serve()
    finally:
        backend.unload()
    log.info("Follower exiting")


def spawn_sp_followers(
    sp_size: int,
    base_gpu_id: int,
    model_path: str,
    nccl_port: int,
    *,
    gpu_id_step: int = 1,
    skip_semantic_encoder: bool = False,
) -> list[mp.Process]:
    """Spawn ``sp_size - 1`` follower processes on the GPUs immediately
    following ``base_gpu_id``.

    Returns the list of started ``mp.Process`` instances.  The caller is
    responsible for keeping them alive (and for sending the shutdown
    poison-pill via :meth:`ZImageBackend.sp_shutdown` before joining).
    """
    if sp_size <= 1:
        return []

    ctx = mp.get_context("spawn")
    procs: list[mp.Process] = []
    for rank in range(1, sp_size):
        gpu_id = base_gpu_id + rank * gpu_id_step
        proc = ctx.Process(
            target=sp_follower_loop,
            args=(rank, sp_size, gpu_id, model_path, nccl_port),
            kwargs={"skip_semantic_encoder": skip_semantic_encoder},
            name=f"zimage-sp-{rank}",
            # daemon=False so the follower can stay alive long enough to
            # receive the shutdown poison-pill from the leader. The leader
            # (image_gen stage process) is already non-daemon when sp_size
            # > 1 — see mp_runner.MultiProcessPipelineRunner.
            daemon=False,
        )
        proc.start()
        procs.append(proc)
        logger.info(
            "Spawned ZImage SP follower rank %d on GPU %d (pid=%d)",
            rank,
            gpu_id,
            proc.pid,
        )
    return procs


def _resolve_nccl_port(default: int = 29500) -> int:
    """Pick an open TCP port for NCCL bootstrap.

    Mirrors ``sglang_omni.engines.ar.sglang_backend.model_worker._resolve_nccl_port``
    but kept local to avoid importing the AR backend just for one helper.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
