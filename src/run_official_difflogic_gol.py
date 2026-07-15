"""Attempt the official DiffLogic-CA Game-of-Life training path.

This adapter executes the model/data/training definitions from selected cells
of Google's immutable Apache-2.0 notebook instead of silently reimplementing a
different architecture.  It removes only plotting/progress cells and writes a
machine-readable multi-level semantic check.  The source notebook is verified
by SHA-256 before any cell is executed.

Upstream source:
  google-research/self-organising-systems, notebooks/diffLogic_CA.ipynb
  commit 3d5547ca48b60ecac459834e2c05c9ff5df87991

This is an optional GPU experiment.  Frozen hard-circuit replay is handled by
``run_cellular_automata.py`` and has no JAX dependency.
"""

from __future__ import annotations

import argparse
from collections import namedtuple
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from urllib.request import urlopen


NOTEBOOK_COMMIT = "3d5547ca48b60ecac459834e2c05c9ff5df87991"
NOTEBOOK_SHA256 = "a9b3829db0d9fe0eb46148d516c18358aa648e3538aeaa682003b77cfbe757c8"
NOTEBOOK_URL = (
    "https://raw.githubusercontent.com/google-research/self-organising-systems/"
    + NOTEBOOK_COMMIT
    + "/notebooks/diffLogic_CA.ipynb"
)
SELECTED_CELLS = (6, 8, 10, 12, 15, 16)
OFFICIAL_STEP_CELL_INDEX = 14
OFFICIAL_STEP_CELL_SHA256 = "6e13a2df48a7ead2db303097724680e543dd5988ade20ef33f79a3c53be86461"
DETERMINISTIC_XLA_FLAG = "--xla_gpu_deterministic_ops=true"


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_verified_notebook(path: Path | None) -> tuple[dict, bytes, str]:
    if path is None:
        with urlopen(NOTEBOOK_URL, timeout=120) as response:
            payload = response.read()
        source = NOTEBOOK_URL
    else:
        payload = path.read_bytes()
        source = str(path.resolve())
    actual = digest_bytes(payload)
    if actual != NOTEBOOK_SHA256:
        raise ValueError(f"official notebook SHA-256 mismatch: {actual}")
    document = json.loads(payload.decode("utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("cells"), list):
        raise ValueError("invalid notebook structure")
    return document, payload, source


def official_step(jnp, jax):
    @jax.jit
    def step(board):
        neighbors = sum(
            jnp.roll(board, delta, (0, 1))
            for delta in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1))
        )
        return (neighbors == 3) | (board & (neighbors == 2))

    return step


def execute_official_definitions(notebook: dict) -> tuple[dict, dict[int, str]]:
    import flax.linen as nn
    from einops import rearrange
    import jax
    from jax.lax import conv_general_dilated_patches
    import jax.numpy as jnp
    import jax.random as random
    import optax

    namespace = {
        "namedtuple": namedtuple,
        "partial": partial,
        "nn": nn,
        "rearrange": rearrange,
        "jax": jax,
        "jnp": jnp,
        "random": random,
        "conv_general_dilated_patches": conv_general_dilated_patches,
        "optax": optax,
    }
    step_source = "".join(notebook["cells"][OFFICIAL_STEP_CELL_INDEX].get("source", []))
    step_hash = digest_bytes(step_source.encode("utf-8"))
    if step_hash != OFFICIAL_STEP_CELL_SHA256:
        raise ValueError(f"official GoL step cell SHA-256 mismatch: {step_hash}")
    cell_hashes = {OFFICIAL_STEP_CELL_INDEX: step_hash}
    for index in SELECTED_CELLS:
        source = "".join(notebook["cells"][index].get("source", []))
        if not source.strip():
            raise ValueError(f"official notebook cell {index} is empty")
        cell_hashes[index] = digest_bytes(source.encode("utf-8"))
        if index == 15:
            namespace["step"] = official_step(jnp, jax)
        exec(compile(source, f"official_diffLogic_CA.ipynb:cell_{index}", "exec"), namespace)
    return namespace, cell_hashes


def save_pytree_npz(path: Path, jax, params: object, wires: object) -> dict:
    import numpy as np

    arrays = {}
    inventory = {"format": "diagnostic_flat_pytree_leaves_npz", "reusable_checkpoint": False, "trees": {}}
    for prefix, tree in (("param", params), ("wire", wires)):
        leaves = []
        for index, leaf in enumerate(jax.tree_util.tree_leaves(tree)):
            array = np.asarray(leaf)
            key = f"{prefix}_{index:03d}"
            arrays[key] = array
            leaves.append({"key": key, "shape": list(array.shape), "dtype": str(array.dtype)})
        inventory["trees"][prefix] = {
            "treedef": str(jax.tree_util.tree_structure(tree)),
            "leaves": leaves,
        }
    np.savez_compressed(path, **arrays)
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("results/difflogic_ca_training_attempt.json"))
    parser.add_argument("--model-output", type=Path, default=Path("results/difflogic_ca_training_snapshot.npz"))
    parser.add_argument("--nondeterministic", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.log_every < 1:
        raise ValueError("epochs, batch size, and log interval must be positive")
    xla_tokens = os.environ.get("XLA_FLAGS", "").split()
    if not args.nondeterministic:
        xla_tokens = [token for token in xla_tokens if not token.startswith("--xla_gpu_deterministic_ops=")]
        xla_tokens.append(DETERMINISTIC_XLA_FLAG)
        os.environ["XLA_FLAGS"] = " ".join(xla_tokens)
    actual_xla_flags = os.environ.get("XLA_FLAGS", "")
    deterministic_gpu_ops = DETERMINISTIC_XLA_FLAG in actual_xla_flags.split()

    # JAX is imported only after the deterministic option is fixed.
    import einops
    import flax
    import jax
    import jaxlib
    import jax.numpy as jnp
    import jax.random as random
    import numpy as np
    import optax

    notebook, payload, source = load_verified_notebook(args.notebook)
    started = time.perf_counter()
    namespace, cell_hashes = execute_official_definitions(notebook)
    hyperparams = namespace["hyperparams"]
    hyperparams["num_epochs"] = args.epochs
    hyperparams["batch_size"] = args.batch_size
    trajectories = namespace["trajectories"]
    sample_batch = namespace["sample_batch"]
    init_state = namespace["init_state"]
    train_step = namespace["train_step"]
    opt = namespace["opt"]
    train_state, wires = init_state(hyperparams, opt, hyperparams["seed"])
    key = random.PRNGKey(hyperparams["seed"])
    history = []
    for epoch in range(args.epochs):
        key, sample_key = random.split(key, 2)
        train_x, train_y = sample_batch(sample_key, trajectories, args.batch_size, hyperparams["channels"])
        train_state, soft_loss, hard_loss = train_step(
            train_state,
            train_x,
            train_y[:, 0, :, :],
            wires,
            hyperparams["periodic"],
            hyperparams["num_steps"],
            hyperparams["async_training"],
        )
        if epoch % args.log_every == 0 or epoch + 1 == args.epochs:
            row = {
                "epoch": epoch + 1,
                "soft_loss": float(soft_loss),
                "hard_loss": float(hard_loss["hard"]),
            }
            history.append(row)
            print(json.dumps(row))

    params, _, key = train_state
    evaluation_input = jnp.zeros((512, 3, 3, 1), dtype=jnp.float32)
    evaluation_input = evaluation_input.at[..., 0].set(namespace["initial_boards"])
    prediction = namespace["v_run_iter_nca"](
        evaluation_input,
        params,
        wires,
        False,
        hyperparams["periodic"],
        1,
        False,
        key,
    )
    target = trajectories[:, 1, ..., None]
    prediction_np = np.asarray(prediction, dtype=np.uint8)
    target_np = np.asarray(target, dtype=np.uint8)
    hardened = jax.tree_util.tree_map(lambda value: jnp.argmax(value, axis=-1), params)
    active = jax.tree_util.tree_map(lambda value: ((value != 3) & (value != 5)).sum(), hardened)
    active_gates = int(sum(int(value) for value in jax.tree_util.tree_leaves(active)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    snapshot_inventory = save_pytree_npz(args.model_output, jax, params, wires)
    snapshot_payload = args.model_output.read_bytes()
    snapshot_inventory.update(
        {
            "path": str(args.model_output),
            "bytes": len(snapshot_payload),
            "sha256": digest_bytes(snapshot_payload),
            "note": "This flat leaf archive supports diagnostics but intentionally is not claimed as a directly reloadable optimizer checkpoint.",
        }
    )
    backend = jax.lib.xla_bridge.get_backend()
    result = {
        "status": "completed",
        "reproduction_level": "official_notebook_definition_training_attempt",
        "notebook": {
            "source": source,
            "commit": NOTEBOOK_COMMIT,
            "sha256": digest_bytes(payload),
            "selected_cells": list(SELECTED_CELLS),
            "validated_support_cells": [OFFICIAL_STEP_CELL_INDEX],
            "selected_cell_sha256": {str(key): value for key, value in cell_hashes.items()},
        },
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "flax": flax.__version__,
            "optax": optax.__version__,
            "einops": einops.__version__,
            "numpy": np.__version__,
            "backend": jax.default_backend(),
            "backend_platform_version": getattr(backend, "platform_version", "unknown"),
            "devices": [str(device) for device in jax.devices()],
            "device_kinds": [getattr(device, "device_kind", "unknown") for device in jax.devices()],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "xla_flags": actual_xla_flags,
            "requested_nondeterministic": args.nondeterministic,
            "deterministic_gpu_ops": deterministic_gpu_ops,
        },
        "config": hyperparams,
        "epochs": args.epochs,
        "all_512_boards_full_grid_accuracy": float(np.mean(prediction_np == target_np)),
        "all_512_center_transition_accuracy": float(np.mean(prediction_np[:, 1, 1, 0] == target_np[:, 1, 1, 0])),
        "all_512_full_grid_exact_rate": float(np.mean(np.all(prediction_np == target_np, axis=(1, 2, 3)))),
        "hard_error_bits": int(np.sum(prediction_np != target_np)),
        "reported_active_non_passthrough_gates": active_gates,
        "model_snapshot": snapshot_inventory,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": "One configured seed; this does not establish a multi-seed success rate or hardware advantage.",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "model": str(args.model_output), "accuracy": result["all_512_center_transition_accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
