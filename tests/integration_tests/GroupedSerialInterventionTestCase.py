import unittest
from ..utils import *

from transformers import GPT2LMHeadModel


class GroupedSerialInterventionTestCase(unittest.TestCase):
    """Serial interventions with several keys in one group (issue #209).

    A tiny randomly initialized GPT2 is used, so no download is needed.
    Expected outputs are computed with plain PyTorch forward hooks.
    """

    @classmethod
    def setUpClass(self):
        print("=== Test Suite: GroupedSerialInterventionTestCase ===")
        torch.manual_seed(0)
        self.gpt2 = GPT2LMHeadModel(
            GPT2Config(
                n_layer=2,
                n_embd=8,
                n_head=2,
                vocab_size=20,
                n_positions=16,
                attn_pdrop=0.0,
                embd_pdrop=0.0,
                resid_pdrop=0.0,
            )
        ).eval()
        self.base = {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]])}
        self.source_0 = {"input_ids": torch.tensor([[7, 8, 9, 10, 11, 12]])}
        self.source_1 = {"input_ids": torch.tensor([[13, 14, 15, 16, 17, 18]])}

    def _hidden(self, output):
        return output[0] if isinstance(output, tuple) else output

    def _manual_run(self, inputs, patches=None, capture_layer=None):
        """
        Run gpt2 with plain hooks. patches maps a layer to a list of
        (position, vector) written into the block output; capture_layer
        returns the (patched) block output of that layer.
        """
        handles, captured = [], {}
        for layer, swaps in (patches or {}).items():

            def setter(module, args, output, swaps=swaps):
                for position, vector in swaps:
                    self._hidden(output)[:, position] = vector

            handles.append(
                self.gpt2.transformer.h[layer].register_forward_hook(setter)
            )
        if capture_layer is not None:

            def getter(module, args, output):
                captured["hidden"] = self._hidden(output).detach().clone()

            handles.append(
                self.gpt2.transformer.h[capture_layer].register_forward_hook(getter)
            )
        with torch.no_grad():
            logits = self.gpt2(**inputs).logits
        for handle in handles:
            handle.remove()
        return logits, captured.get("hidden")

    def _num_hooks(self):
        return sum(
            len(module._forward_hooks) + len(module._forward_pre_hooks)
            for module in self.gpt2.modules()
        )

    def _grouped_config(self, group_keys, layers, mode="serial", **kwargs):
        return IntervenableConfig(
            [
                {"layer": layer, "component": "block_output", "group_key": group_key}
                for layer, group_key in zip(layers, group_keys)
            ],
            intervention_types=VanillaIntervention,
            mode=mode,
            **kwargs,
        )

    def test_grouped_keys_use_own_positions(self):
        """
        Two keys in one group read source positions 1 and 4 and write
        base positions 2 and 3, over repeated calls.
        """
        pv_gpt2 = IntervenableModel(
            self._grouped_config([0, 0], [0, 0]), model=self.gpt2
        )
        _, source_hidden = self._manual_run(self.source_0, capture_layer=0)
        golden, _ = self._manual_run(
            self.base,
            patches={0: [(2, source_hidden[:, 1]), (3, source_hidden[:, 4])]},
        )
        original, _ = self._manual_run(self.base)
        self.assertFalse(torch.allclose(golden, original))

        for _ in range(2):
            _, outputs = pv_gpt2(
                self.base,
                [self.source_0],
                {"source_0->base": ([[[1]], [[4]]], [[[2]], [[3]]])},
            )
            self.assertTrue(torch.allclose(outputs.logits, golden))
            self.assertEqual(self._num_hooks(), 0)

        # the wrapped model is left untouched afterwards
        self.assertTrue(torch.allclose(self._manual_run(self.base)[0], original))

    def test_multi_group_hook_ownership(self):
        """
        Group 0 holds two keys (layer 0) and patches source_1, group 1 holds
        one key (layer 1) and patches base. Getters of a group must be gone
        once its source forward is done.
        """
        pv_gpt2 = IntervenableModel(
            self._grouped_config([0, 0, 1], [0, 0, 1]), model=self.gpt2
        )
        _, source_0_hidden = self._manual_run(self.source_0, capture_layer=0)
        _, source_1_hidden = self._manual_run(
            self.source_1,
            patches={0: [(2, source_0_hidden[:, 1]), (3, source_0_hidden[:, 4])]},
            capture_layer=1,
        )
        golden, _ = self._manual_run(
            self.base, patches={1: [(5, source_1_hidden[:, 3])]}
        )

        # record the number of hooks on each block at every model call
        hook_counts = []
        layers = self.gpt2.transformer.h
        original_forward = self.gpt2.forward

        def counting_forward(*args, **kwargs):
            hook_counts.append(
                (len(layers[0]._forward_hooks), len(layers[1]._forward_hooks))
            )
            return original_forward(*args, **kwargs)

        self.gpt2.forward = counting_forward
        try:
            for _ in range(2):
                hook_counts.clear()
                _, outputs = pv_gpt2(
                    self.base,
                    [self.source_0, self.source_1],
                    {
                        "source_0->source_1": ([[[1]], [[4]]], [[[2]], [[3]]]),
                        "source_1->base": ([[[3]]], [[[5]]]),
                    },
                )
                self.assertTrue(torch.allclose(outputs.logits, golden))
                # source_0: 2 getters; source_1: 2 setters + 1 getter;
                # base: 1 setter
                self.assertEqual(hook_counts, [(2, 0), (2, 1), (0, 1)])
                self.assertEqual(self._num_hooks(), 0)
        finally:
            del self.gpt2.forward

    def test_grouped_source_representations(self):
        """Supplied source representations are written to each key's position."""
        pv_gpt2 = IntervenableModel(
            self._grouped_config([0, 0], [0, 0]), model=self.gpt2
        )
        torch.manual_seed(1)
        vectors = torch.randn(2, 8)
        golden, _ = self._manual_run(
            self.base, patches={0: [(2, vectors[0]), (3, vectors[1])]}
        )
        _, outputs = pv_gpt2(
            self.base,
            [self.source_0],
            {"source_0->base": ([[[1]], [[4]]], [[[2]], [[3]]])},
            source_representations={
                pv_gpt2.sorted_keys[0]: vectors[0].view(1, 1, 8),
                pv_gpt2.sorted_keys[1]: vectors[1].view(1, 1, 8),
            },
        )
        self.assertTrue(torch.allclose(outputs.logits, golden))
        self.assertEqual(self._num_hooks(), 0)

    def test_grouped_constant_source(self):
        """Constant source interventions are written to each key's position."""
        torch.manual_seed(2)
        vectors = torch.randn(2, 8)
        config = IntervenableConfig(
            [
                {
                    "layer": 0,
                    "component": "block_output",
                    "group_key": 0,
                    "source_representation": vectors[i],
                }
                for i in range(2)
            ],
            intervention_types=VanillaIntervention,
            mode="serial",
        )
        pv_gpt2 = IntervenableModel(config, model=self.gpt2)
        golden, _ = self._manual_run(
            self.base, patches={0: [(2, vectors[0]), (3, vectors[1])]}
        )
        _, outputs = pv_gpt2(
            self.base,
            [self.source_0],
            {"source_0->base": ([[[1]], [[4]]], [[[2]], [[3]]])},
        )
        self.assertTrue(torch.allclose(outputs.logits, golden))
        self.assertEqual(self._num_hooks(), 0)

    def test_grouped_generate_on_prompt(self):
        """generate shares the serial path; intervene on the prompt only."""
        pv_gpt2 = IntervenableModel(
            self._grouped_config([0, 0], [0, 0]), model=self.gpt2
        )
        _, source_hidden = self._manual_run(self.source_0, capture_layer=0)
        swaps = [(2, source_hidden[:, 1]), (3, source_hidden[:, 4])]

        def prompt_setter(module, args, output):
            # the prompt is the only forward with the full sequence
            if self._hidden(output).shape[1] == self.base["input_ids"].shape[1]:
                for position, vector in swaps:
                    self._hidden(output)[:, position] = vector

        generate_kwargs = dict(
            max_new_tokens=3,
            do_sample=False,
            pad_token_id=0,
            return_dict_in_generate=True,
            output_logits=True,
        )
        handle = self.gpt2.transformer.h[0].register_forward_hook(prompt_setter)
        with torch.no_grad():
            golden = self.gpt2.generate(**self.base, **generate_kwargs)
        handle.remove()

        _, outputs = pv_gpt2.generate(
            self.base,
            [self.source_0],
            {"source_0->base": ([[[1]], [[4]]], [[[2]], [[3]]])},
            intervene_on_prompt=True,
            **generate_kwargs,
        )
        self.assertTrue(torch.equal(outputs.sequences, golden.sequences))
        for logits, golden_logits in zip(outputs.logits, golden.logits):
            self.assertTrue(torch.allclose(logits, golden_logits))
        self.assertEqual(self._num_hooks(), 0)

    def test_single_key_groups_serial_control(self):
        """Default grouping (one key per group) is unchanged."""
        pv_gpt2 = IntervenableModel(
            IntervenableConfig(
                [
                    {"layer": 0, "component": "block_output"},
                    {"layer": 1, "component": "block_output"},
                ],
                intervention_types=VanillaIntervention,
                mode="serial",
            ),
            model=self.gpt2,
        )
        _, source_0_hidden = self._manual_run(self.source_0, capture_layer=0)
        _, source_1_hidden = self._manual_run(
            self.source_1, patches={0: [(2, source_0_hidden[:, 1])]}, capture_layer=1
        )
        golden, _ = self._manual_run(
            self.base, patches={1: [(5, source_1_hidden[:, 3])]}
        )
        _, outputs = pv_gpt2(
            self.base,
            [self.source_0, self.source_1],
            {
                "source_0->source_1": ([[[1]]], [[[2]]]),
                "source_1->base": ([[[3]]], [[[5]]]),
            },
        )
        self.assertTrue(torch.allclose(outputs.logits, golden))
        self.assertEqual(self._num_hooks(), 0)

    def test_parallel_control(self):
        """Parallel mode with the same per-key locations gives the same result."""
        pv_gpt2 = IntervenableModel(
            self._grouped_config([0, 1], [0, 0], mode="parallel"), model=self.gpt2
        )
        _, source_hidden = self._manual_run(self.source_0, capture_layer=0)
        golden, _ = self._manual_run(
            self.base,
            patches={0: [(2, source_hidden[:, 1]), (3, source_hidden[:, 4])]},
        )
        _, outputs = pv_gpt2(
            self.base,
            [self.source_0, self.source_0],
            {"sources->base": ([[[1]], [[4]]], [[[2]], [[3]]])},
        )
        self.assertTrue(torch.allclose(outputs.logits, golden))
        self.assertEqual(self._num_hooks(), 0)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(GroupedSerialInterventionTestCase("test_grouped_keys_use_own_positions"))
    suite.addTest(GroupedSerialInterventionTestCase("test_multi_group_hook_ownership"))
    suite.addTest(GroupedSerialInterventionTestCase("test_grouped_source_representations"))
    suite.addTest(GroupedSerialInterventionTestCase("test_grouped_constant_source"))
    suite.addTest(GroupedSerialInterventionTestCase("test_grouped_generate_on_prompt"))
    suite.addTest(GroupedSerialInterventionTestCase("test_single_key_groups_serial_control"))
    suite.addTest(GroupedSerialInterventionTestCase("test_parallel_control"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
