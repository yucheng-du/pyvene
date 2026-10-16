"""Native save/load with the RepresentationConfig migration (stanfordnlp/pyvene#157)."""
import contextlib
import copy
import json
import os
import shutil
import tempfile
import unittest

import torch

from pyvene.models.configuration_intervenable_model import (
    IntervenableConfig,
    RepresentationConfig,
)
from pyvene.models.intervenable_base import IntervenableModel
from pyvene.models.interventions import (
    AutoencoderIntervention,
    LowRankRotatedSpaceIntervention,
    VanillaIntervention,
)
from ..representation_config_helpers import (
    HF_METADATA_KEYS,
    LEGACY_ARCHIVES,
    ScaledInterpolationIntervention,
    make_inputs,
    make_mlp,
    unit_locations,
)


def _reference(name):
    return torch.load(os.path.join(LEGACY_ARCHIVES, name, "reference.pt"), weights_only=True)


def _forward(intervenable, base, sources, locations):
    sources = None if sources is None else [{"inputs_embeds": s} for s in sources]
    with torch.no_grad():
        original, intervened = intervenable({"inputs_embeds": base}, sources,
                                            unit_locations=locations,
                                            output_original_output=True)
    return original[0], intervened[0]


def _tensors(module):
    return {name: (id(t), t.detach().clone(), t.device, t.dtype, t.requires_grad)
            for name, t in list(module.named_parameters()) + list(module.named_buffers())}


def _snapshot(intervenable):
    """Values and identities of everything a save() call could touch."""
    config = intervenable.config
    representations = []
    for representation in config.representations:
        fields = {name: (id(value), copy.deepcopy(value) if isinstance(
                         value, (int, float, str, list, dict, type(None))) else value)
                  for name, value in representation._asdict().items()}
        extra = getattr(representation, "intervention_kwargs", None)
        representations.append((id(representation), fields, id(extra), copy.deepcopy(extra)))
    attributes = {name: (id(getattr(config, name)), copy.copy(getattr(config, name)))
                  for name in ("intervention_types", "sorted_keys", "mode", "model_type",
                               "intervention_dimensions", "intervention_constant_sources")}
    return {"representations": representations, "attributes": attributes,
            "model": _tensors(intervenable.model),
            "interventions": {k: (id(v), _tensors(v)) for k, v in intervenable.interventions.items()}}


class RepresentationConfigSaveLoadTestCase(unittest.TestCase):

    def assertSameTensors(self, before, after):
        self.assertEqual(set(before), set(after))
        for name, (ident, value, device, dtype, requires_grad) in before.items():
            self.assertEqual((after[name][0], after[name][2], after[name][3], after[name][4]),
                             (ident, device, dtype, requires_grad), name)
            self.assertTrue(torch.equal(after[name][1], value), name)

    def assertUnchanged(self, before, after):
        self.assertEqual(after["representations"], before["representations"])
        self.assertEqual(after["attributes"], before["attributes"])
        self.assertSameTensors(before["model"], after["model"])
        self.assertEqual(set(after["interventions"]), set(before["interventions"]))
        for key, (ident, tensors) in before["interventions"].items():
            self.assertEqual(after["interventions"][key][0], ident)
            self.assertSameTensors(tensors, after["interventions"][key][1])

    # Archives written by the unmodified base implementation.

    def _load_legacy(self, name, directory=None):
        reference = _reference(name)
        model = make_mlp(state_dict=reference["base_model_state_dict"])
        directory = directory or os.path.join(LEGACY_ARCHIVES, name, "archive")
        return reference, IntervenableModel.load(directory, model)

    def _assert_matches_reference(self, name, reference, loaded, compare_state=True):
        archive = os.path.join(LEGACY_ARCHIVES, name, "archive")
        self.assertEqual(list(loaded.interventions.keys()), reference["intervention_keys"])
        self.assertEqual(list(loaded.sorted_keys), reference["sorted_keys"])
        for key, intervention in loaded.interventions.items():
            if not compare_state:
                break
            state = intervention.state_dict()
            saved = reference["intervention_state_dicts"][key]
            binary = os.path.join(archive, f"intkey_{key}.bin")
            if os.path.exists(binary):
                saved_binary = torch.load(binary, weights_only=True)
                for tensor_name, value in saved_binary.items():
                    torch.testing.assert_close(state[tensor_name], value, rtol=0, atol=0)
            self.assertEqual(set(state), set(saved))
            for tensor_name, value in saved.items():
                self.assertEqual(state[tensor_name].shape, value.shape)
                torch.testing.assert_close(state[tensor_name], value, rtol=0, atol=1e-6)
        inputs = reference["inputs"]
        original, intervened = _forward(loaded, inputs["base"], inputs["sources"],
                                        reference["unit_locations"])
        torch.testing.assert_close(original, reference["outputs"]["original"], rtol=0, atol=1e-5)
        torch.testing.assert_close(intervened, reference["outputs"]["intervened"], rtol=0, atol=1e-5)
        self.assertFalse(torch.allclose(original, intervened))

    def test_legacy_lowrank_archive(self):
        reference, loaded = self._load_legacy("lowrank_two_layers")
        self._assert_matches_reference("lowrank_two_layers", reference, loaded)
        self.assertEqual([(r.layer, r.low_rank_dimension, r.latent_dim)
                          for r in loaded.config.representations], [(0, 2, None), (1, 2, None)])

    def test_legacy_13_field_layout_control(self):
        source = os.path.join(LEGACY_ARCHIVES, "lowrank_two_layers")
        with open(os.path.join(source, "archive", "config.json"), encoding="utf-8") as handle:
            genuine = json.load(handle)
        with open(os.path.join(source, "config_13field_derived.json"), encoding="utf-8") as handle:
            derived = json.load(handle)
        # The derived control only drops the trailing latent_dim=None of each record.
        self.assertTrue(all(len(r) == 14 and r[13] is None for r in genuine["representations"]))
        self.assertEqual(derived["representations"], [r[:13] for r in genuine["representations"]])
        self.assertEqual({k: v for k, v in derived.items() if k != "representations"},
                         {k: v for k, v in genuine.items() if k != "representations"})
        with tempfile.TemporaryDirectory() as directory:
            shutil.copytree(os.path.join(source, "archive"), directory, dirs_exist_ok=True)
            shutil.copyfile(os.path.join(source, "config_13field_derived.json"),
                            os.path.join(directory, "config.json"))
            reference, loaded = self._load_legacy("lowrank_two_layers", directory)
        self._assert_matches_reference("lowrank_two_layers", reference, loaded)

    def test_legacy_constant_source_archive(self):
        reference, loaded = self._load_legacy("constant_source")
        self._assert_matches_reference("constant_source", reference, loaded)
        key = reference["intervention_keys"][0]
        intervention = loaded.interventions[key]
        self.assertTrue(intervention.is_source_constant)
        self.assertTrue(loaded.config.representations[0].hidden_source_representation)
        torch.testing.assert_close(
            intervention.source_representation,
            reference["intervention_state_dicts"][key]["source_representation"], rtol=0, atol=0)

    def test_legacy_linked_archive(self):
        reference, loaded = self._load_legacy("linked_pair")
        self._assert_matches_reference("linked_pair", reference, loaded)
        first, second = loaded.interventions.values()
        self.assertIs(first, second)
        self.assertTrue(reference["sorted_keys"][0].startswith("layer_1_"))
        self.assertEqual(set(loaded._intervention_reverse_link.values()), {"link#0"})

    def test_legacy_autoencoder_archive(self):
        reference, loaded = self._load_legacy("autoencoder_latent")
        self._assert_matches_reference("autoencoder_latent", reference, loaded)
        intervention = list(loaded.interventions.values())[0]
        self.assertEqual(loaded.config.representations[0].latent_dim, 4)
        self.assertEqual(intervention.autoencoder.latent_dim, 4)
        self.assertEqual(intervention.interchange_dim.tolist(), [0, 2])

    def test_legacy_archives_are_rewritten_in_named_form(self):
        for name in ("lowrank_two_layers", "constant_source", "linked_pair", "autoencoder_latent"):
            reference, loaded = self._load_legacy(name)
            with tempfile.TemporaryDirectory() as directory:
                loaded.save(directory)
                with open(os.path.join(directory, "config.json"), encoding="utf-8") as handle:
                    written = json.load(handle)
                reloaded = IntervenableModel.load(
                    directory, make_mlp(state_dict=reference["base_model_state_dict"]))
            self.assertEqual(written["representation_schema_version"], 1, name)
            self.assertTrue(all(isinstance(r, dict) for r in written["representations"]), name)
            legacy = [r._asdict() for r in loaded.config.representations]
            for record, fields in zip(written["representations"], legacy):
                for field in ("layer", "component", "unit", "max_number_of_units",
                              "low_rank_dimension", "intervention_link_key", "latent_dim"):
                    self.assertEqual(record[field], fields[field], (name, field))
                self.assertEqual(record["intervention_kwargs"], {})
            # Re-saving a loaded constant source keeps the existing writer behaviour
            # (the hidden flag follows a tensor held by the configuration), so the
            # tensor comes back through intervention_constant_sources as a plain
            # attribute; compare its value and the outputs instead of state dicts.
            constant = name == "constant_source"
            self._assert_matches_reference(name, reference, reloaded, compare_state=not constant)
            if constant:
                key = reference["intervention_keys"][0]
                torch.testing.assert_close(
                    reloaded.interventions[key].source_representation,
                    reference["intervention_state_dicts"][key]["source_representation"],
                    rtol=0, atol=0)

    # Native trainable save/load.

    def _train_save_load(self, config, seed, prepare=None, allow_mutation=False):
        model = make_mlp(seed)
        intervenable = IntervenableModel(config, model)
        if prepare is not None:
            prepare(intervenable)
        n = len(config.representations)
        base, sources = make_inputs(seed + 1, n)
        locations = unit_locations(n)
        params = list({id(p): p for p in intervenable.get_trainable_parameters()}.values())
        optimizer = torch.optim.SGD(params, lr=0.5)
        # AutoencoderIntervention writes in place into a ReLU output; this public
        # autograd context lets the unchanged intervention math be differentiated.
        context = (torch.autograd.graph.allow_mutation_on_saved_tensors()
                   if allow_mutation else contextlib.nullcontext())
        with context:
            _, out = intervenable({"inputs_embeds": base}, [{"inputs_embeds": s} for s in sources],
                                  unit_locations=locations)
            target = torch.randn(out[0].shape, generator=torch.Generator().manual_seed(seed + 2))
            torch.nn.functional.mse_loss(out[0], target).backward()
        grads = [p.grad.detach().clone() for p in params]
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(max(g.abs().max().item() for g in grads), 0)
        before = [p.detach().clone() for p in params]
        optimizer.step()
        optimizer.zero_grad()
        self.assertGreater(max((p.detach() - b).abs().max().item() for p, b in zip(params, before)), 0)

        snapshot = _snapshot(intervenable)
        with tempfile.TemporaryDirectory() as directory:
            intervenable.save(directory)
            self.assertUnchanged(snapshot, _snapshot(intervenable))
            loaded = IntervenableModel.load(directory, make_mlp(state_dict=model.state_dict()))
        self.assertEqual(list(loaded.interventions.keys()), list(intervenable.interventions.keys()))
        for key, intervention in intervenable.interventions.items():
            restored = loaded.interventions[key]
            self.assertIsNot(restored, intervention)
            self.assertIs(type(restored), type(intervention))
            expected_state = {k: v.detach().clone() for k, v in intervention.state_dict().items()}
            restored_state = restored.state_dict()
            self.assertEqual(set(restored_state), set(expected_state))
            for name, value in expected_state.items():
                self.assertEqual(restored_state[name].shape, value.shape)
                torch.testing.assert_close(restored_state[name], value, rtol=0, atol=0)
            for (name, p), (_, q) in zip(intervention.named_parameters(), restored.named_parameters()):
                self.assertIsNot(p, q)
                self.assertEqual(p.shape, q.shape, name)
        for actual, expected in zip(_forward(loaded, base, sources, locations),
                                    _forward(intervenable, base, sources, locations)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
        return intervenable, loaded

    def test_lowrank_train_save_load(self):
        config = IntervenableConfig(representations=[
            RepresentationConfig(0, "block_output", "pos", 1, low_rank_dimension=2),
            RepresentationConfig(1, "block_output", "pos", 1, low_rank_dimension=2),
        ], intervention_types=LowRankRotatedSpaceIntervention)
        _, loaded = self._train_save_load(config, seed=20)
        for intervention in loaded.interventions.values():
            self.assertEqual(tuple(intervention.rotate_layer.weight.shape), (8, 2))

    def test_autoencoder_train_save_load(self):
        config = IntervenableConfig(representations=[
            RepresentationConfig(0, "block_output", "pos", 1, latent_dim=4)],
            intervention_types=AutoencoderIntervention)

        def valid_latent_indices(intervenable):
            # the default interchange_dim (embed_dim=8) is not a valid latent index
            for intervention in intervenable.interventions.values():
                intervention.set_interchange_dim([1, 3])

        _, loaded = self._train_save_load(config, seed=30, prepare=valid_latent_indices,
                                          allow_mutation=True)
        intervention = list(loaded.interventions.values())[0]
        self.assertEqual(intervention.interchange_dim.tolist(), [1, 3])
        self.assertEqual(tuple(intervention.autoencoder.encoder[0].weight.shape), (4, 8))

    def test_include_model_false_and_true(self):
        model = make_mlp(40)
        config = IntervenableConfig(representations=[
            RepresentationConfig(0, "block_output", "pos", 1, low_rank_dimension=2)],
            intervention_types=LowRankRotatedSpaceIntervention)
        intervenable = IntervenableModel(config, model)
        bias = model.h[1].ff1.bias
        bias.requires_grad_(True)
        with torch.no_grad():
            bias.add_(0.5)
        for include_model in (False, True):
            fresh = make_mlp(40)  # same weights as model before the bias update
            with tempfile.TemporaryDirectory() as directory:
                intervenable.save(directory, include_model=include_model)
                self.assertEqual(
                    os.path.exists(os.path.join(directory, "pytorch_model.bin")), include_model)
                loaded = IntervenableModel.load(directory, fresh, include_model=include_model)
            expected = bias.detach() if include_model else make_mlp(40).h[1].ff1.bias.detach()
            torch.testing.assert_close(loaded.model.h[1].ff1.bias.detach(), expected, rtol=0, atol=0)
            torch.testing.assert_close(loaded.model.h[0].ff1.weight, model.h[0].ff1.weight,
                                       rtol=0, atol=0)

    def test_constant_source_save_load(self):
        model = make_mlp(50)
        source = torch.randn(8, generator=torch.Generator().manual_seed(51))
        config = IntervenableConfig(representations=[
            RepresentationConfig(0, "block_output", "pos", 1, source_representation=source)],
            intervention_types=VanillaIntervention)
        intervenable = IntervenableModel(config, model)
        base, _ = make_inputs(52, 0)
        with tempfile.TemporaryDirectory() as directory:
            intervenable.save(directory)
            loaded = IntervenableModel.load(directory, make_mlp(state_dict=model.state_dict()))
        self.assertIs(config.representations[0].source_representation, source)
        torch.testing.assert_close(
            list(loaded.interventions.values())[0].source_representation, source, rtol=0, atol=0)
        for actual, expected in zip(_forward(loaded, base, None, {"base": 0}),
                                    _forward(intervenable, base, None, {"base": 0})):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_instance_path_golden_output(self):
        model = make_mlp(60)
        instance = LowRankRotatedSpaceIntervention(embed_dim=8, low_rank_dimension=2)
        config = IntervenableConfig(representations=[RepresentationConfig(
            0, "block_output", "pos", 1, low_rank_dimension=2, intervention=instance)])
        intervenable = IntervenableModel(config, model)
        self.assertIs(list(intervenable.interventions.values())[0], instance)
        base, sources = make_inputs(61)
        _, intervened = _forward(intervenable, base, sources, unit_locations(1))
        with torch.no_grad():
            weight = instance.rotate_layer.weight
            h_base, h_source = model.h[0](base), model.h[0](sources[0])
            golden = model.h[1](h_base + (h_source @ weight - h_base @ weight) @ weight.T)
        torch.testing.assert_close(intervened, golden)

    def test_save_load_explicit_intervention_kwargs(self):
        model = make_mlp(70)
        config = IntervenableConfig(representations=[
            RepresentationConfig(0, "block_output", intervention_type=ScaledInterpolationIntervention,
                                 intervention_kwargs={"scale": 0.25, "schedule": {"steps": [1, 2]},
                                                      "dtype": torch.float64}),
            RepresentationConfig(1, "block_output", low_rank_dimension=2,
                                 intervention_type=LowRankRotatedSpaceIntervention),
        ])
        intervenable = IntervenableModel(config, model)
        snapshot = _snapshot(intervenable)
        with tempfile.TemporaryDirectory() as directory:
            intervenable.save(directory)
            self.assertUnchanged(snapshot, _snapshot(intervenable))
            with open(os.path.join(directory, "config.json"), encoding="utf-8") as handle:
                written = json.load(handle)
            loaded = IntervenableModel.load(directory, make_mlp(state_dict=model.state_dict()))
        self.assertEqual(written["representation_schema_version"], 1)
        self.assertEqual(written["intervention_types"], [
            str(ScaledInterpolationIntervention), str(LowRankRotatedSpaceIntervention)])
        first, second = written["representations"]
        self.assertEqual(first["intervention_kwargs"],
                         {"scale": 0.25, "schedule": {"steps": [1, 2]}, "dtype": "float64"})
        self.assertEqual(second["intervention_kwargs"], {})
        scaled = list(loaded.interventions.values())[0]
        self.assertEqual(scaled.received_kwargs["scale"], 0.25)
        self.assertEqual(scaled.received_kwargs["schedule"], {"steps": [1, 2]})
        self.assertIs(scaled.received_kwargs["dtype"], torch.float64)
        for key in HF_METADATA_KEYS:
            if key != "dtype":
                self.assertNotIn(key, scaled.received_kwargs)
        base, sources = make_inputs(71, 2)
        for actual, expected in zip(_forward(loaded, base, sources, unit_locations(2)),
                                    _forward(intervenable, base, sources, unit_locations(2))):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
