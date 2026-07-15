"""Explicit M04a neural suite; excluded from default ``test_*.py`` discovery."""

from __future__ import annotations

import gc
import unittest

import torch

from afts_arc.m04a_contract import D_MODEL, MODEL_PARAMETER_COUNT
from afts_arc.m04a_model import (
    DEMO_INPUT_ROLE_ID,
    DEMO_OUTPUT_ROLE_ID,
    GRID_CLS_POSITION_ID,
    GRID_CLS_TOKEN_ID,
    MASK_TOKEN_ID,
    PAD_TOKEN_ID,
    QUERY_INPUT_ROLE_ID,
    TARGET_ROLE_ID,
    GridCMLM,
    TokenBatch,
    _CUDATransferAttestation,
    _version_tracked_tensor_copies,
    mask_target_grid,
    model_config,
    parameter_state_sha256,
    tokenize_target_batch,
    tokenize_task_memory,
    trainable_parameter_count,
)


def _small_task_tokens() -> TokenBatch:
    demonstrations = (
        (((0, 1), (2, 3)), ((1, 0), (3, 2))),
        (((4, 5, 6),), ((6, 5, 4),)),
    )
    return tokenize_task_memory(demonstrations, ((7,), (8,), (9,)))


class GridCMLMContractCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))

    def test_task_tokenization_is_one_canonical_grid_batch(self) -> None:
        batch = _small_task_tokens()
        self.assertEqual(batch.batch_size, 5)
        self.assertEqual(batch.semantic_kind, "encoder")
        self.assertTrue(batch.has_padding)
        self.assertEqual(batch.lengths, (5, 5, 4, 4, 4))
        self.assertEqual(batch.padded_length, 5)
        self.assertEqual(batch.token_ids[:, 0].tolist(), [GRID_CLS_TOKEN_ID] * 5)
        self.assertEqual(
            batch.role_ids[:, 0].tolist(),
            [
                DEMO_INPUT_ROLE_ID,
                DEMO_OUTPUT_ROLE_ID,
                DEMO_INPUT_ROLE_ID,
                DEMO_OUTPUT_ROLE_ID,
                QUERY_INPUT_ROLE_ID,
            ],
        )
        self.assertEqual(batch.pair_slot_ids[:, 0].tolist(), [0, 0, 1, 1, 15])
        self.assertEqual(batch.row_ids[0].tolist(), [30, 0, 0, 1, 1])
        self.assertEqual(batch.column_ids[0].tolist(), [30, 0, 1, 0, 1])
        self.assertEqual(batch.height_ids[0].tolist(), [2] * 5)
        self.assertEqual(batch.width_ids[0].tolist(), [2] * 5)
        self.assertTrue(batch.padding_mask[2, -1].item())
        self.assertEqual(batch.token_ids[2, -1].item(), PAD_TOKEN_ID)
        self.assertEqual(batch.row_ids[2, -1].item(), GRID_CLS_POSITION_ID)
        self.assertEqual(batch.height_ids[2, -1].item(), 0)
        self.assertEqual(batch.role_ids[2, -1].item(), DEMO_INPUT_ROLE_ID)
        self.assertEqual(batch.pair_slot_ids[2, -1].item(), 1)

    def test_ten_demonstrations_form_exactly_twenty_one_encoder_grids(self) -> None:
        grid = ((0,),)
        batch = tokenize_task_memory(tuple((grid, grid) for _ in range(10)), grid)
        self.assertEqual(batch.batch_size, 21)
        self.assertEqual(batch.pair_slot_ids[18, 0].item(), 9)
        self.assertEqual(batch.pair_slot_ids[19, 0].item(), 9)
        self.assertEqual(batch.pair_slot_ids[20, 0].item(), 15)
        with self.assertRaisesRegex(ValueError, "1..10"):
            tokenize_task_memory(tuple((grid, grid) for _ in range(11)), grid)

    def test_encoder_query_role_requires_canonical_query_slot(self) -> None:
        canonical = _small_task_tokens()
        fields = {
            name: getattr(canonical, name).clone()
            for name in (
                "token_ids",
                "row_ids",
                "column_ids",
                "height_ids",
                "width_ids",
                "role_ids",
                "pair_slot_ids",
                "padding_mask",
            )
        }
        fields["pair_slot_ids"][-1, :] = 0
        with self.assertRaisesRegex(ValueError, "query-input.*slot 15"):
            TokenBatch(
                **fields,
                lengths=canonical.lengths,
                semantic_kind="encoder",
            )
        reordered = {
            name: getattr(canonical, name)[[2, 3, 0, 1, 4]].clone()
            for name in fields
        }
        with self.assertRaisesRegex(ValueError, "demo_in0,demo_out0"):
            TokenBatch(
                **reordered,
                lengths=tuple(canonical.lengths[index] for index in (2, 3, 0, 1, 4)),
                semantic_kind="encoder",
            )

    def test_attested_cpu_tensor_mutation_cannot_transfer(self) -> None:
        self.assertNotIn(
            "sealed_tensors", _CUDATransferAttestation.__dataclass_fields__
        )
        batch = tokenize_target_batch((((MASK_TOKEN_ID,),),), device="cpu")
        batch.token_ids[0, 0] = 3  # Still in-domain: only the fingerprint catches it.
        with self.assertRaisesRegex(RuntimeError, "changed after semantic validation"):
            batch.to("cuda")
        with self.assertRaisesRegex(RuntimeError, "changed after semantic validation"):
            GridCMLM().embed(batch)

    def test_attested_copies_retain_versions_inside_inference_mode(self) -> None:
        source = (
            torch.arange(4, dtype=torch.long).reshape(2, 2),
            torch.tensor(((False, True), (True, False)), dtype=torch.bool),
        )
        with torch.inference_mode():
            copied = _version_tracked_tensor_copies(source)
            moved = _version_tracked_tensor_copies(
                copied, device=torch.device("cpu"), non_blocking=False
            )
            self.assertTrue(torch.is_inference_mode_enabled())
        self.assertTrue(all(not tensor.is_inference() for tensor in copied))
        self.assertTrue(all(not tensor.is_inference() for tensor in moved))
        versions = tuple(int(tensor._version) for tensor in copied)
        copied[0].fill_(0)
        self.assertEqual(int(copied[0]._version), versions[0] + 1)

    def test_target_tokenization_has_no_cls_and_masks_only_declared_cells(self) -> None:
        target = ((0, 1), (2, 3))
        partial = mask_target_grid(target, (1, 3))
        self.assertEqual(partial, ((0, MASK_TOKEN_ID), (2, MASK_TOKEN_ID)))
        batch = tokenize_target_batch((partial, ((4,),)))
        self.assertEqual(batch.semantic_kind, "target")
        self.assertTrue(batch.has_padding)
        self.assertEqual(batch.lengths, (4, 1))
        self.assertNotIn(GRID_CLS_TOKEN_ID, batch.token_ids[0, :4].tolist())
        self.assertEqual(batch.token_ids[0, :4].tolist(), [0, 10, 2, 10])
        self.assertEqual(batch.role_ids[0, :4].tolist(), [TARGET_ROLE_ID] * 4)
        self.assertEqual(batch.pair_slot_ids[0, :4].tolist(), [15] * 4)
        self.assertTrue(torch.all(batch.padding_mask[1, 1:]))
        with self.assertRaises(ValueError):
            mask_target_grid(target, (3, 1))

    def test_parameter_count_layer_independence_and_output_domain(self) -> None:
        model = GridCMLM()
        self.assertEqual(trainable_parameter_count(model), 8_733_706)
        self.assertEqual(trainable_parameter_count(model), MODEL_PARAMETER_COUNT)
        self.assertEqual(len(model.encoder_layers), 3)
        self.assertEqual(len(model.decoder_layers), 6)
        self.assertEqual(len({id(layer) for layer in model.encoder_layers}), 3)
        self.assertEqual(len({id(layer) for layer in model.decoder_layers}), 6)
        self.assertEqual(
            len({layer.self_attn.in_proj_weight.data_ptr() for layer in model.encoder_layers}),
            3,
        )
        self.assertFalse(
            torch.equal(
                model.encoder_layers[0].self_attn.in_proj_weight,
                model.encoder_layers[1].self_attn.in_proj_weight,
            )
        )
        self.assertEqual(model.output_head.in_features, D_MODEL)
        self.assertEqual(model.output_head.out_features, 10)
        self.assertEqual(model_config()["parameter_count"], 8_733_706)
        self.assertFalse(model_config()["target_has_grid_cls"])

    def test_frozen_reset_is_independent_of_ambient_rng_and_hashable(self) -> None:
        torch.manual_seed(99)
        torch.rand(17)
        first = GridCMLM()
        first_hash = parameter_state_sha256(first)
        self.assertRegex(first_hash, r"^[0-9a-f]{64}$")
        del first
        gc.collect()

        torch.manual_seed(123456)
        torch.rand(31)
        second = GridCMLM()
        second_hash = parameter_state_sha256(second)
        self.assertEqual(first_hash, second_hash)
        self.assertTrue(torch.all(second.encoder_final_norm.weight == 1))
        self.assertTrue(torch.all(second.encoder_final_norm.bias == 0))
        self.assertGreater(second.token_embedding.weight[PAD_TOKEN_ID].abs().sum().item(), 0.0)
        before_data_mutation = parameter_state_sha256(second)
        next(second.parameters()).data.add_(1.0)
        self.assertNotEqual(before_data_mutation, parameter_state_sha256(second))

    def test_embeddings_are_a_direct_unscaled_sum_and_pad_is_zero(self) -> None:
        model = GridCMLM().eval()
        batch = _small_task_tokens()
        with torch.no_grad():
            embedded = model.embed(batch)
            expected = (
                model.token_embedding(batch.token_ids)
                + model.row_embedding(batch.row_ids)
                + model.column_embedding(batch.column_ids)
                + model.height_embedding(batch.height_ids)
                + model.width_embedding(batch.width_ids)
                + model.role_embedding(batch.role_ids)
                + model.pair_slot_embedding(batch.pair_slot_ids)
            )
            expected = expected.masked_fill(batch.padding_mask.unsqueeze(-1), 0.0)
        self.assertTrue(torch.equal(embedded, expected))
        self.assertTrue(torch.all(embedded.masked_select(batch.padding_mask.unsqueeze(-1)) == 0))

    def test_encoder_padding_is_zero_and_does_not_change_unpadded_memory(self) -> None:
        model = GridCMLM().eval()
        batch = _small_task_tokens()
        extra_padded = batch.right_pad_to(batch.padded_length + 4)
        with torch.no_grad():
            encoded = model.encode_grid_batch(batch)
            encoded_padded = model.encode_grid_batch(extra_padded)
            memory = model.encode_task_memory(batch)
            memory_padded = model.encode_task_memory(extra_padded)
        self.assertTrue(torch.all(encoded.masked_select(batch.padding_mask.unsqueeze(-1)) == 0))
        self.assertTrue(
            torch.all(encoded_padded.masked_select(extra_padded.padding_mask.unsqueeze(-1)) == 0)
        )
        for index, length in enumerate(batch.lengths):
            self.assertTrue(
                torch.allclose(
                    encoded[index, :length], encoded_padded[index, :length], atol=1e-5, rtol=1e-5
                )
            )
        self.assertEqual(memory.memory.shape, (1, sum(batch.lengths), D_MODEL))
        self.assertEqual(memory.grid_lengths, batch.lengths)
        self.assertFalse(torch.any(memory.key_padding_mask).item())
        self.assertTrue(torch.allclose(memory.memory, memory_padded.memory, atol=1e-5, rtol=1e-5))
        with self.assertRaisesRegex(ValueError, "CUDA"):
            memory.to_bf16_cuda_cache()

    def test_bidirectional_decoder_outputs_only_ten_logits_and_zeros_pad(self) -> None:
        model = GridCMLM().eval()
        task = _small_task_tokens()
        targets = tokenize_target_batch((((MASK_TOKEN_ID, 1), (2, 3)), ((MASK_TOKEN_ID,),)))
        with torch.no_grad():
            memory = model.encode_task_memory(task)
            decoded = model.decode_target(targets, memory)
        self.assertEqual(decoded.logits.shape, (2, 4, 10))
        self.assertEqual(decoded.hidden_states.shape, (2, 4, D_MODEL))
        self.assertTrue(torch.all(decoded.logits[1, 1:] == 0))
        self.assertTrue(torch.all(decoded.hidden_states[1, 1:] == 0))
        self.assertTrue(torch.equal(decoded.padding_mask, targets.padding_mask))
        self.assertNotEqual(
            decoded.padding_mask.data_ptr(), targets.padding_mask.data_ptr()
        )
        baseline_logits = decoded.logits.clone()
        decoded.padding_mask.fill_(False)
        with torch.no_grad():
            replayed = model.decode_target(targets, memory)
        self.assertTrue(torch.equal(replayed.logits, baseline_logits))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the explicit neural checks")
class GridCMLMContractCudaTests(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA BF16 is required by the frozen M04a contract")

    def test_cuda_token_batch_requires_cpu_validation_attestation(self) -> None:
        grid = ((MASK_TOKEN_ID,),)
        cpu = tokenize_target_batch((grid,), device="cpu")
        fields = {
            name: getattr(cpu, name).to("cuda")
            for name in (
                "token_ids",
                "row_ids",
                "column_ids",
                "height_ids",
                "width_ids",
                "role_ids",
                "pair_slot_ids",
                "padding_mask",
            )
        }
        with self.assertRaisesRegex(ValueError, "CPU-validated"):
            TokenBatch(
                **fields,
                lengths=cpu.lengths,
                semantic_kind=cpu.semantic_kind,
            )
        validated = cpu.to("cuda")
        self.assertTrue(validated.token_ids.is_cuda)
        self.assertEqual(validated.semantic_kind, "target")
        validated.role_ids.fill_(QUERY_INPUT_ROLE_ID)
        with self.assertRaisesRegex(RuntimeError, "validated transfer"):
            GridCMLM().to("cuda").embed(validated)
        data_mutated = cpu.to("cuda")
        model = GridCMLM().to("cuda").eval()
        with torch.inference_mode():
            inference_transferred = cpu.to("cuda")
            self.assertTrue(
                all(
                    not tensor.is_inference()
                    for tensor in inference_transferred._all_tensors()
                )
            )
            self.assertEqual(
                len(
                    tuple(
                        int(tensor._version)
                        for tensor in inference_transferred._all_tensors()
                    )
                ),
                8,
            )
            self.assertTrue(model.embed(inference_transferred).is_cuda)
        with torch.no_grad():
            sealed_output = model.embed(data_mutated).clone()
            data_mutated.role_ids.data.fill_(QUERY_INPUT_ROLE_ID)
            self.assertTrue(torch.equal(sealed_output, model.embed(data_mutated)))
            self.assertFalse(
                hasattr(data_mutated._validation_attestation, "sealed_tensors")
            )
            exposed_snapshot = data_mutated._tensors_for_model_consumption()
            exposed_snapshot[0].data.fill_(3)
            self.assertTrue(torch.equal(sealed_output, model.embed(data_mutated)))

            roundtrip_source = cpu.to("cuda")
            roundtrip_source.token_ids.data.fill_(3)
            self.assertFalse(
                torch.equal(roundtrip_source.token_ids.cpu(), cpu.token_ids)
            )
            roundtripped = roundtrip_source.to("cpu")
            self.assertTrue(torch.equal(roundtripped.token_ids, cpu.token_ids))

            task = _small_task_tokens().to("cuda")
            target = tokenize_target_batch(
                (((MASK_TOKEN_ID,),), ((MASK_TOKEN_ID, 1),)), device="cpu"
            ).to("cuda")
            memory = model.encode_task_memory(task)
            baseline = model.decode_target(target, memory)
            baseline_logits = baseline.logits.clone()
            self.assertTrue(bool(baseline.padding_mask.any().item()))
            baseline.padding_mask.fill_(False)
            self.assertTrue(
                torch.equal(
                    baseline_logits,
                    model.decode_target(target, memory).logits,
                )
            )
            data_exposed = model.decode_target(target, memory)
            data_exposed.padding_mask.data.fill_(False)
            self.assertTrue(
                torch.equal(
                    baseline_logits,
                    model.decode_target(target, memory).logits,
                )
            )

    def test_bf16_cache_math_sdpa_and_maximum_context_forward_backward(self) -> None:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        device = torch.device("cuda")
        grid = tuple(
            tuple((row_index + column_index) % 10 for column_index in range(30))
            for row_index in range(30)
        )
        demonstrations = tuple((grid, grid) for _ in range(10))
        task = tokenize_task_memory(demonstrations, grid, device=device)
        partial = tuple(tuple(MASK_TOKEN_ID for _ in range(30)) for _ in range(30))
        target = tokenize_target_batch((partial,), device=device)
        model = GridCMLM().to(device=device).train()
        torch.cuda.reset_peak_memory_stats(device)

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                decoded = model(task, target)
                loss = decoded.logits.float().mean()
            loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        self.assertTrue(torch.isfinite(loss).item())
        self.assertLess(torch.cuda.max_memory_allocated(device), torch.cuda.get_device_properties(device).total_memory)

        model.zero_grad(set_to_none=True)
        model.eval()
        eight_targets = tokenize_target_batch((partial,) * 8, device=device)
        with torch.no_grad(), sdpa_kernel(backends=[SDPBackend.MATH]):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                memory = model.encode_task_memory(task)
            cache = memory.to_bf16_cuda_cache()
            expanded_memory, expanded_mask = cache.expand_for_batch(8)
            self.assertEqual(cache.memory.dtype, torch.bfloat16)
            self.assertTrue(cache.memory.is_cuda)
            self.assertTrue(cache.memory.is_contiguous())
            self.assertTrue(cache.key_padding_mask.is_contiguous())
            self.assertEqual(expanded_memory.shape[0], 8)
            self.assertEqual(expanded_mask.shape[0], 8)
            self.assertTrue(expanded_memory.is_contiguous())
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                inference = model.decode_target(eight_targets, cache)
        self.assertEqual(inference.logits.shape, (8, 900, 10))
        self.assertTrue(torch.isfinite(inference.logits).all().item())


if __name__ == "__main__":
    unittest.main(verbosity=2)
