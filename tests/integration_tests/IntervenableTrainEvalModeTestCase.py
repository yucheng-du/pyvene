import copy
import unittest

import torch
import torch.nn.functional as F

from pyvene.models.configuration_intervenable_model import (
    IntervenableConfig,
    RepresentationConfig,
)
from pyvene.models.intervenable_base import IntervenableModel
from pyvene.models.interventions import (
    LocalistRepresentationIntervention,
    SourcelessIntervention,
)
from pyvene.models.mlp.modelings_mlp import MLPConfig, MLPModel


class DropoutIntervention(
    SourcelessIntervention, LocalistRepresentationIntervention
):
    """Sourceless intervention that applies dropout to the base representation."""

    def __init__(self, p, **kwargs):
        super().__init__(**kwargs)
        self.dropout = torch.nn.Dropout(p)

    def forward(self, base, source=None, subspaces=None, **kwargs):
        return self.dropout(base)


def identity_mlp(pdrop=0.0):
    """One-layer MLP whose block output equals its input."""
    mlp = MLPModel(
        MLPConfig(
            n_layer=1,
            h_dim=4,
            activation_function="linear",
            pdrop=pdrop,
            include_bias=False,
            include_emb=False,
        )
    )
    with torch.no_grad():
        mlp.h[0].ff1.weight.copy_(torch.eye(4))
    return mlp


def seeded(seed, fn):
    """Run fn from a fixed CPU RNG state, restoring the global state afterwards."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return fn()


class IntervenableTrainEvalModeTestCase(unittest.TestCase):
    """IntervenableModel.train()/eval() follow torch.nn.Module semantics."""

    maxDiff = None

    def setUp(self):
        self.base = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        self.zeros = torch.zeros_like(self.base)

    def _build(self, p):
        intervention = DropoutIntervention(p)
        pv_model = IntervenableModel(
            IntervenableConfig(
                representations=[
                    RepresentationConfig(
                        layer=0,
                        component="block_output",
                        intervention=intervention,
                    )
                ]
            ),
            model=identity_mlp(),
        )
        return pv_model, intervention

    def _intervened(self, pv_model):
        _, counterfactual_outputs = pv_model(
            base={"inputs_embeds": self.base}, unit_locations={"base": 0}
        )
        return counterfactual_outputs[0]

    def _training_flags(self, pv_model):
        return {name: m.training for name, m in pv_model.named_modules()}

    def assertTensorEqual(self, actual, expected):
        self.assertEqual(actual.tolist(), expected.tolist())

    def test_positive_control_explicit_intervention_eval(self):
        pv_model, intervention = self._build(p=1.0)
        self.assertIs(pv_model.interventions[pv_model.sorted_keys[0]], intervention)
        # a freshly built intervention is in training mode, so p=1 drops all
        self.assertTensorEqual(self._intervened(pv_model), self.zeros)
        intervention.eval()
        self.assertTensorEqual(self._intervened(pv_model), self.base)

    def test_eval_disables_intervention_dropout(self):
        pv_model, _ = self._build(p=1.0)
        pv_model.eval()
        self.assertTensorEqual(self._intervened(pv_model), self.base)

    def test_train_restores_intervention_dropout(self):
        pv_model, intervention = self._build(p=1.0)
        intervention.eval()
        self.assertTensorEqual(self._intervened(pv_model), self.base)
        pv_model.train()
        self.assertTensorEqual(self._intervened(pv_model), self.zeros)

    def test_mode_switch_reaches_all_submodules(self):
        pv_model, intervention = self._build(p=1.0)
        modules = dict(pv_model.named_modules())
        self.assertIs(modules[""], pv_model)
        for submodule in (
            pv_model.interventions,
            intervention,
            intervention.dropout,
            pv_model.model,
            pv_model.model.dropout,
            pv_model.model.h[0].dropout,
        ):
            self.assertTrue(any(m is submodule for m in modules.values()))
        for name, call, expected in (
            ("eval()", lambda: pv_model.eval(), False),
            ("train()", lambda: pv_model.train(), True),
            ("train(False)", lambda: pv_model.train(False), False),
        ):
            call()
            with self.subTest(call=name):
                self.assertEqual(
                    self._training_flags(pv_model), dict.fromkeys(modules, expected)
                )

    def test_train_false_is_equivalent_to_eval(self):
        pv_model, _ = self._build(p=1.0)
        pv_model.train()
        pv_model.eval()
        eval_flags = self._training_flags(pv_model)
        eval_output = self._intervened(pv_model)
        pv_model.train()
        pv_model.train(False)
        self.assertEqual(self._training_flags(pv_model), eval_flags)
        self.assertTensorEqual(self._intervened(pv_model), eval_output)
        self.assertTensorEqual(eval_output, self.base)

    def test_mode_methods_return_self(self):
        pv_model, _ = self._build(p=1.0)
        self.assertIs(pv_model.eval(), pv_model)
        self.assertIs(pv_model.train(), pv_model)
        self.assertIs(pv_model.train(False), pv_model)
        self.assertIs(pv_model.train(mode=True), pv_model)

    def test_mode_switch_preserves_parameters(self):
        for model_gradients in (False, True):
            with self.subTest(model_gradients=model_gradients):
                pv_model, _ = self._build(p=1.0)
                if model_gradients:
                    pv_model.enable_model_gradients()
                before = [
                    (name, p, p.detach().clone(), p.requires_grad)
                    for name, p in pv_model.model.named_parameters()
                ]
                self.assertEqual(len(before), 1)
                self.assertEqual(before[0][3], model_gradients)
                trainable = [id(p) for p in pv_model.get_trainable_parameters()]
                hf_named = [(n, id(p)) for n, p in pv_model.named_parameters()]
                for call in (
                    lambda: pv_model.eval(),
                    lambda: pv_model.train(),
                    lambda: pv_model.train(False),
                    lambda: pv_model.train(),
                ):
                    call()
                    after = list(pv_model.model.named_parameters())
                    self.assertEqual(
                        [name for name, _ in after], [name for name, *_ in before]
                    )
                    for (_, p), (name, p_before, value, requires_grad) in zip(
                        after, before
                    ):
                        self.assertIs(p, p_before, name)
                        self.assertTrue(torch.equal(p, value), name)
                        self.assertEqual(p.requires_grad, requires_grad, name)
                    self.assertEqual(
                        [id(p) for p in pv_model.get_trainable_parameters()],
                        trainable,
                    )
                    self.assertEqual(
                        [(n, id(p)) for n, p in pv_model.named_parameters()],
                        hf_named,
                    )
                    self.assertEqual(pv_model.model_has_grad, model_gradients)

    def test_no_intervention_forward_matches_bare_model_mode(self):
        mlp = identity_mlp(pdrop=0.5)
        bare = copy.deepcopy(mlp)
        pv_model = IntervenableModel(IntervenableConfig(representations=[]), model=mlp)
        base = {"inputs_embeds": self.base}
        seed = 0
        bare.train()
        train_reference = seeded(seed, lambda: bare(**base)[0])
        bare.eval()
        eval_reference = bare(**base)[0]
        self.assertTensorEqual(eval_reference, self.base)
        self.assertNotEqual(train_reference.tolist(), eval_reference.tolist())

        pv_model.train()
        self.assertTrue(mlp.training)
        self.assertTensorEqual(seeded(seed, lambda: pv_model(base)[0][0]), train_reference)
        pv_model.eval()
        self.assertFalse(mlp.training)
        self.assertTensorEqual(seeded(seed, lambda: pv_model(base)[0][0]), eval_reference)

    def test_half_dropout_eval_is_identity_across_seeds(self):
        pv_model, _ = self._build(p=0.5)
        pv_model.eval()
        for seed in range(5):
            with self.subTest(seed=seed):
                self.assertTensorEqual(
                    seeded(seed, lambda: self._intervened(pv_model)), self.base
                )

    def test_half_dropout_train_matches_reference_masks(self):
        pv_model, intervention = self._build(p=0.5)
        seeds = (0, 1)
        references = {
            seed: seeded(
                seed, lambda: F.dropout(self.base, p=0.5, training=True)
            )
            for seed in seeds
        }
        # the reference masks are non-trivial and differ between the seeds
        for reference in references.values():
            self.assertNotEqual(reference.tolist(), self.base.tolist())
            self.assertNotEqual(reference.tolist(), self.zeros.tolist())
        self.assertNotEqual(references[0].tolist(), references[1].tolist())

        intervention.eval()
        self.assertTensorEqual(
            seeded(seeds[0], lambda: self._intervened(pv_model)), self.base
        )
        for switch, call in (
            ("train", lambda: pv_model.train()),
            ("eval->train", lambda: (pv_model.eval(), pv_model.train())),
        ):
            call()
            for seed in seeds:
                with self.subTest(switch=switch, seed=seed):
                    self.assertTensorEqual(
                        seeded(seed, lambda: self._intervened(pv_model)),
                        references[seed],
                    )


if __name__ == "__main__":
    unittest.main()
