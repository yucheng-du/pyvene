import copy
import os
import shutil
import tempfile
import unittest

import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

import pyvene as pv


class InterventionCheckpointLoadTestCase(unittest.TestCase):
    """
    CPU round trips through the public checkpoint APIs:
    save_intervention() + load_intervention() restore artifacts into the
    same (existing) IntervenableModel, while save() + IntervenableModel.load()
    build a new wrapper from saved metadata.
    """

    # tolerances for recomputed forward outputs, losses and gradients;
    # restored parameters and buffers are compared exactly.
    RTOL = 1e-5
    ATOL = 1e-6

    @classmethod
    def setUpClass(cls):
        print("=== Test Suite: InterventionCheckpointLoadTestCase ===")
        cls.vocab_size = 13
        # different, nonzero token ids for base and source
        cls.base = {"input_ids": torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])}
        cls.source = {"input_ids": torch.tensor([[11, 12, 3, 6, 9], [2, 4, 12, 1, 5]])}
        # source position 1 -> base position 3 (example 0),
        # source position 2 -> base position 4 (example 1)
        cls.unit_locations = {"sources->base": ([[[1], [2]]], [[[3], [4]]])}
        cls.targets = torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]])

    def _make_tmp_dir(self):
        tmp_dir = tempfile.mkdtemp(prefix="pyvene_ckpt_")
        self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
        return os.path.join(tmp_dir, "ckpt")

    def _make_gpt2(self, seed=0):
        torch.manual_seed(seed)
        config = GPT2Config(
            n_layer=1,
            n_embd=16,
            n_head=2,
            n_positions=16,
            vocab_size=self.vocab_size,
            attn_pdrop=0.0,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            summary_first_dropout=0.0,
            bos_token_id=0,
            eos_token_id=0,
            attn_implementation="eager",
        )
        return GPT2LMHeadModel(config).to(dtype=torch.float32).eval()

    def _make_low_rank_intervenable(self, gpt2):
        # rank-4 rotated subspace of the 16-dim block output (partial interchange)
        return pv.IntervenableModel(
            {
                "layer": 0,
                "component": "block_output",
                "unit": "pos",
                "low_rank_dimension": 4,
                "intervention_type": pv.LowRankRotatedSpaceIntervention,
            },
            model=gpt2,
        )

    def _forward(self, intervenable):
        _, outputs = intervenable(self.base, [self.source], self.unit_locations)
        logits = outputs.logits
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size), self.targets.reshape(-1)
        )
        return logits, loss

    def _reference(self, intervenable, named_params):
        """Detached forward output, loss and gradients for `named_params`."""
        self._clear_grads(named_params)
        logits, loss = self._forward(intervenable)
        loss.backward()
        grads = {n: p.grad.detach().clone() for n, p in named_params.items()}
        self._clear_grads(named_params)
        return logits.detach().clone(), loss.detach().clone(), grads

    def _clear_grads(self, named_params):
        for p in named_params.values():
            p.grad = None

    def _snapshot(self, module):
        return {k: v.detach().clone() for k, v in module.state_dict().items()}

    def _param_meta(self, named_params):
        return {
            n: (p, p.data_ptr(), p.device, p.dtype, p.requires_grad)
            for n, p in named_params.items()
        }

    def _perturb(self, named_params, seed=1234):
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in named_params.values():
                p.add_(0.5 * torch.randn(p.shape, generator=generator))

    def _assert_meaningful_reference(self, logits, loss, grads):
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(loss))
        for n, g in grads.items():
            self.assertTrue(torch.isfinite(g).all(), n)
            self.assertGreater(g.abs().sum().item(), 0.0, n)

    def _assert_perturbed(self, intervenable, named_params, saved_params, ref_logits):
        for n, p in named_params.items():
            self.assertFalse(torch.equal(p.detach(), saved_params[n]), n)
        with torch.no_grad():
            logits, _ = self._forward(intervenable)
        self.assertGreater((logits - ref_logits).abs().max().item(), 1e-3)

    def _assert_param_meta_unchanged(self, named_params, meta):
        for n, (p_obj, ptr, device, dtype, requires_grad) in meta.items():
            p = named_params[n]
            self.assertIs(p, p_obj, n)
            self.assertEqual(p.data_ptr(), ptr, n)
            self.assertEqual(p.device, device, n)
            self.assertEqual(p.dtype, dtype, n)
            self.assertEqual(p.requires_grad, requires_grad, n)

    def _assert_state_equal(self, module, snapshot):
        state = module.state_dict()
        self.assertEqual(set(state.keys()), set(snapshot.keys()))
        for k, v in snapshot.items():
            self.assertEqual(state[k].device, v.device, k)
            self.assertEqual(state[k].dtype, v.dtype, k)
            self.assertTrue(torch.equal(state[k], v), k)

    def _assert_numerics_restored(self, intervenable, named_params, ref_logits, ref_loss, ref_grads):
        self._clear_grads(named_params)
        logits, loss = self._forward(intervenable)
        loss.backward()
        torch.testing.assert_close(logits.detach(), ref_logits, rtol=self.RTOL, atol=self.ATOL)
        torch.testing.assert_close(loss.detach(), ref_loss, rtol=self.RTOL, atol=self.ATOL)
        for n, p in named_params.items():
            torch.testing.assert_close(p.grad, ref_grads[n], rtol=self.RTOL, atol=self.ATOL)
        self._clear_grads(named_params)

    def test_load_intervention_restores_cpu_checkpoint_into_same_instance(self):
        intervenable = self._make_low_rank_intervenable(self._make_gpt2())
        key, intervention = next(iter(intervenable.interventions.items()))
        self.assertIsInstance(intervention, pv.TrainableIntervention)
        named_params = dict(intervention.named_parameters())
        self.assertGreater(len(named_params), 0)

        ref_logits, ref_loss, ref_grads = self._reference(intervenable, named_params)
        self._assert_meaningful_reference(ref_logits, ref_loss, ref_grads)
        saved_state = self._snapshot(intervention)
        saved_params = {n: p.detach().clone() for n, p in named_params.items()}

        save_dir = self._make_tmp_dir()
        intervenable.save_intervention(save_dir)
        self.assertTrue(os.path.isfile(os.path.join(save_dir, f"intkey_{key}.bin")))

        self._perturb(named_params)
        self._assert_perturbed(intervenable, named_params, saved_params, ref_logits)
        meta = self._param_meta(named_params)

        intervenable.load_intervention(save_dir)

        self._assert_param_meta_unchanged(dict(intervention.named_parameters()), meta)
        self._assert_state_equal(intervention, saved_state)
        self._assert_numerics_restored(intervenable, named_params, ref_logits, ref_loss, ref_grads)

    def test_load_intervention_without_model_artifact(self):
        intervenable = self._make_low_rank_intervenable(self._make_gpt2())
        key, intervention = next(iter(intervenable.interventions.items()))
        named_params = dict(intervention.named_parameters())

        ref_logits, ref_loss, ref_grads = self._reference(intervenable, named_params)
        self._assert_meaningful_reference(ref_logits, ref_loss, ref_grads)
        saved_state = self._snapshot(intervention)
        saved_params = {n: p.detach().clone() for n, p in named_params.items()}

        save_dir = self._make_tmp_dir()
        intervenable.save_intervention(save_dir, include_model=False)
        self.assertEqual(sorted(os.listdir(save_dir)), [f"intkey_{key}.bin"])

        self._perturb(named_params)
        self._assert_perturbed(intervenable, named_params, saved_params, ref_logits)
        meta = self._param_meta(named_params)

        intervenable.load_intervention(save_dir, include_model=False)

        self.assertFalse(os.path.exists(os.path.join(save_dir, "pytorch_model.bin")))
        self._assert_param_meta_unchanged(dict(intervention.named_parameters()), meta)
        self._assert_state_equal(intervention, saved_state)
        self._assert_numerics_restored(intervenable, named_params, ref_logits, ref_loss, ref_grads)

    def test_load_intervention_with_trainable_model_parameter(self):
        gpt2 = self._make_gpt2()
        intervenable = self._make_low_rank_intervenable(gpt2)
        key, intervention = next(iter(intervenable.interventions.items()))

        # IntervenableModel freezes the base model; unfreeze one real weight.
        model_param_name = "transformer.h.0.mlp.c_proj.weight"
        model_param = dict(gpt2.named_parameters())[model_param_name]
        model_param.requires_grad_(True)
        frozen = {
            n: (p, p.detach().clone())
            for n, p in gpt2.named_parameters() if n != model_param_name
        }
        self.assertGreater(len(frozen), 0)
        for n, (p, _) in frozen.items():
            self.assertFalse(p.requires_grad, n)

        named_params = {f"intervention.{n}": p for n, p in intervention.named_parameters()}
        named_params[f"model.{model_param_name}"] = model_param

        ref_logits, ref_loss, ref_grads = self._reference(intervenable, named_params)
        self._assert_meaningful_reference(ref_logits, ref_loss, ref_grads)
        saved_intervention_state = self._snapshot(intervention)
        saved_params = {n: p.detach().clone() for n, p in named_params.items()}

        save_dir = self._make_tmp_dir()
        intervenable.save_intervention(save_dir, include_model=True)
        self.assertEqual(
            sorted(os.listdir(save_dir)), sorted([f"intkey_{key}.bin", "pytorch_model.bin"])
        )
        # only trainable model parameters are saved by design
        saved_model_state = torch.load(
            os.path.join(save_dir, "pytorch_model.bin"), map_location="cpu", weights_only=True
        )
        self.assertEqual(list(saved_model_state.keys()), [model_param_name])

        self._perturb(named_params)
        self._assert_perturbed(intervenable, named_params, saved_params, ref_logits)
        meta = self._param_meta(named_params)

        intervenable.load_intervention(save_dir, include_model=True)

        live_params = {f"intervention.{n}": p for n, p in intervention.named_parameters()}
        live_params[f"model.{model_param_name}"] = dict(gpt2.named_parameters())[model_param_name]
        self._assert_param_meta_unchanged(live_params, meta)
        self._assert_state_equal(intervention, saved_intervention_state)
        self.assertTrue(torch.equal(model_param.detach(), saved_params[f"model.{model_param_name}"]))
        for n, (p, value) in frozen.items():
            self.assertFalse(p.requires_grad, n)
            self.assertTrue(torch.equal(p.detach(), value), n)
        self._assert_numerics_restored(intervenable, named_params, ref_logits, ref_loss, ref_grads)

    def test_non_trainable_intervention_skips_binary_control(self):
        gpt2 = self._make_gpt2()
        intervenable = pv.IntervenableModel(
            {
                "layer": 0,
                "component": "block_output",
                "unit": "pos",
                "intervention_type": pv.VanillaIntervention,
            },
            model=gpt2,
        )
        intervention = next(iter(intervenable.interventions.values()))
        self.assertNotIsInstance(intervention, pv.TrainableIntervention)

        with torch.no_grad():
            base_logits = gpt2(**self.base).logits
            source_logits = gpt2(**self.source).logits
        # block_output of the last block feeds ln_f/lm_head position-wise, so a
        # full positional interchange copies the source logits at that position
        expected = base_logits.clone()
        expected[0, 3] = source_logits[0, 1]
        expected[1, 4] = source_logits[1, 2]
        self.assertGreater((expected - base_logits).abs().max().item(), 1e-3)

        save_dir = self._make_tmp_dir()
        intervenable.save_intervention(save_dir, include_model=False)
        self.assertEqual(os.listdir(save_dir), [])
        intervenable.load_intervention(save_dir, include_model=False)

        with torch.no_grad():
            logits, _ = self._forward(intervenable)
        torch.testing.assert_close(logits, expected, rtol=self.RTOL, atol=self.ATOL)

    def test_static_save_load_new_instance_control(self):
        gpt2 = self._make_gpt2()
        compatible_gpt2 = copy.deepcopy(gpt2)
        intervenable = self._make_low_rank_intervenable(gpt2)
        key, intervention = next(iter(intervenable.interventions.items()))

        with torch.no_grad():
            ref_logits, _ = self._forward(intervenable)
        saved_state = self._snapshot(intervention)

        save_dir = self._make_tmp_dir()
        intervenable.save(save_directory=save_dir, save_to_hf_hub=False)
        self.assertTrue(os.path.isfile(os.path.join(save_dir, "config.json")))
        self.assertTrue(os.path.isfile(os.path.join(save_dir, f"intkey_{key}.bin")))

        # a different seed so the new wrapper's fresh initialization cannot match
        torch.manual_seed(4321)
        loaded = pv.IntervenableModel.load(save_dir, model=compatible_gpt2)

        self.assertIsNot(loaded, intervenable)
        self.assertEqual(list(loaded.interventions.keys()), [key])
        loaded_intervention = loaded.interventions[key]
        self.assertIsInstance(loaded_intervention, pv.LowRankRotatedSpaceIntervention)
        self.assertIsNot(loaded_intervention, intervention)
        self._assert_state_equal(loaded_intervention, saved_state)
        with torch.no_grad():
            loaded_logits, _ = self._forward(loaded)
        torch.testing.assert_close(loaded_logits, ref_logits, rtol=self.RTOL, atol=self.ATOL)


if __name__ == "__main__":
    unittest.main()
