"""RepresentationConfig construction and serialization (stanfordnlp/pyvene#157)."""
import copy
import json
import os
import tempfile
import unittest

import torch

from pyvene.models.basic_utils import get_type_from_string
from pyvene.models.configuration_intervenable_model import (
    IntervenableConfig,
    RepresentationConfig,
)
from pyvene.models.intervenable_base import IntervenableModel
from pyvene.models.interventions import (
    LowRankRotatedSpaceIntervention,
    VanillaIntervention,
)
from ..representation_config_helpers import (
    HF_METADATA_KEYS,
    ScaledInterpolationIntervention,
    make_inputs,
    make_mlp,
    unit_locations,
)

# The documented positional order of the declared fields.
FIELDS = (
    "layer", "component", "unit", "max_number_of_units", "low_rank_dimension",
    "intervention_type", "intervention", "subspace_partition", "group_key",
    "intervention_link_key", "moe_key", "source_representation",
    "hidden_source_representation", "latent_dim",
)
SCALED = ScaledInterpolationIntervention
MODEL_KWARGS = {"embed_dim", "use_fast"}


def _run(intervenable, base, source):
    _, out = intervenable({"inputs_embeds": base}, [{"inputs_embeds": source}],
                          unit_locations=unit_locations(1))
    return out[0]


class RepresentationConfigTestCase(unittest.TestCase):

    # Construction paths that already existed.

    def test_default_construction(self):
        self.assertEqual(
            [getattr(RepresentationConfig(), name) for name in FIELDS],
            [0, "block_output", "pos", 1] + [None] * 10)
        config = IntervenableConfig()
        self.assertEqual(len(config.representations), 1)
        self.assertEqual(config.representations[0].component, "block_output")
        self.assertIs(config.intervention_types, VanillaIntervention)

    def test_positional_and_keyword_construction(self):
        values = [3, "mlp_output", "h.pos", 2, 4, None, None, [[0, 2], [2, 4]],
                  1, 7, 0, None, True, 5]
        positional = RepresentationConfig(*values)
        keyword = RepresentationConfig(**dict(zip(FIELDS, values)))
        for name, value in zip(FIELDS, values):
            self.assertEqual(getattr(positional, name), value)
            self.assertEqual(getattr(keyword, name), value)
        self.assertEqual(positional._asdict(), dict(zip(FIELDS, values)))
        self.assertEqual(list(positional._asdict()), list(FIELDS))

    def test_legacy_positional_records(self):
        record_14 = [1, "block_output", "pos", 1, 2, None, None, None, None, 0,
                     None, None, True, None]
        record_13 = record_14[:13]
        autoencoder = [0, "block_output", "pos", 1] + [None] * 9 + [4]
        snapshot = copy.deepcopy([record_13, record_14, autoencoder])
        config = IntervenableConfig(representations=[record_13, record_14, autoencoder,
                                                     [2, "block_input"]])
        thirteen, fourteen, latent, short = config.representations
        for representation in (thirteen, fourteen):
            self.assertEqual(
                (representation.layer, representation.low_rank_dimension,
                 representation.intervention_link_key,
                 representation.hidden_source_representation, representation.latent_dim),
                (1, 2, 0, True, None))
        self.assertEqual(latent.latent_dim, 4)
        self.assertEqual((short.layer, short.component, short.unit, short.max_number_of_units),
                         (2, "block_input", "pos", 1))
        self.assertEqual([record_13, record_14, autoencoder], snapshot)

    def test_dict_records_and_add_intervention(self):
        config = IntervenableConfig(representations=[
            {"layer": 0, "component": "block_output", "intervention_type": VanillaIntervention}])
        self.assertEqual(config.intervention_types, [VanillaIntervention])
        config.add_intervention({"layer": 1, "low_rank_dimension": 2,
                                 "intervention_type": LowRankRotatedSpaceIntervention})
        config.add_intervention(
            RepresentationConfig(0, "block_input", intervention_type=VanillaIntervention))
        config.add_intervention([[1, "block_input", "pos", 1, None, VanillaIntervention]])
        self.assertEqual([r.layer for r in config.representations], [0, 1, 0, 1])
        self.assertEqual(config.representations[1].low_rank_dimension, 2)
        self.assertEqual(config.intervention_types, [
            VanillaIntervention, LowRankRotatedSpaceIntervention,
            VanillaIntervention, VanillaIntervention])
        with self.assertRaises(ValueError):
            config.add_intervention({"layer": 1})  # intervention_type is required here

    def test_type_and_instance_exclusivity(self):
        with self.assertRaises(ValueError):
            IntervenableConfig(representations=[RepresentationConfig(
                intervention_type=VanillaIntervention,
                intervention=VanillaIntervention(embed_dim=8))])
        with self.assertRaises(ValueError):
            IntervenableConfig(representations=[
                RepresentationConfig(intervention_type=VanillaIntervention),
                RepresentationConfig(1)])
        with self.assertRaises(ValueError):
            IntervenableConfig(representations=[(0, "block_output")])

    def test_per_representation_type_selection(self):
        config = IntervenableConfig(representations=[
            RepresentationConfig(0, "block_output", intervention_type=VanillaIntervention),
            RepresentationConfig(1, "block_output", low_rank_dimension=2,
                                 intervention_type=LowRankRotatedSpaceIntervention)])
        self.assertEqual(config.intervention_types,
                         [VanillaIntervention, LowRankRotatedSpaceIntervention])
        intervenable = IntervenableModel(config, make_mlp(1))
        self.assertEqual([type(v) for v in intervenable.interventions.values()],
                         [VanillaIntervention, LowRankRotatedSpaceIntervention])

    # Explicit intervention_kwargs.

    def test_intervention_kwargs_export(self):
        representation = RepresentationConfig(
            1, "block_output", low_rank_dimension=2, intervention_type=SCALED,
            intervention_kwargs={"scale": 0.25, "schedule": {"steps": [1, 2]},
                                 "dtype": torch.float64})
        expected = dict(representation._asdict(), scale=0.25,
                        schedule={"steps": [1, 2]}, dtype=torch.float64)
        self.assertEqual(representation.to_intervention_kwargs(), expected)
        self.assertEqual(list(representation._asdict()), list(FIELDS))
        # PretrainedConfig carries dtype=None and other metadata; none of it is exported.
        plain = RepresentationConfig(0)
        self.assertTrue(hasattr(plain, "dtype") and hasattr(plain, "transformers_version"))
        self.assertEqual(plain.to_intervention_kwargs(), plain._asdict())
        for key in HF_METADATA_KEYS:
            self.assertNotIn(key, plain.to_intervention_kwargs())

    def test_model_passes_only_explicit_kwargs(self):
        config = IntervenableConfig(representations=[
            RepresentationConfig(0, "block_output", intervention_type=SCALED,
                                 intervention_kwargs={"scale": 0.25}),
            RepresentationConfig(1, "block_output", intervention_type=SCALED,
                                 intervention_kwargs={"dtype": None}),
            RepresentationConfig(1, "block_input", intervention_type=SCALED,
                                 intervention_kwargs={"dtype": "float64"}),
        ])
        absent, explicit_none, float64 = IntervenableModel(config, make_mlp(2)).interventions.values()
        self.assertEqual(set(absent.received_kwargs), set(FIELDS) | MODEL_KWARGS | {"scale"})
        self.assertEqual(set(explicit_none.received_kwargs), set(FIELDS) | MODEL_KWARGS | {"dtype"})
        self.assertIsNone(explicit_none.received_kwargs["dtype"])
        self.assertIs(float64.received_kwargs["dtype"], torch.float64)
        self.assertEqual(float64.dtype_probe.dtype, torch.float64)
        self.assertEqual(absent.received_kwargs["embed_dim"], 8)

    def test_dtype_codec(self):
        def stored(value):
            return RepresentationConfig(intervention_kwargs={"dtype": value}).intervention_kwargs["dtype"]
        self.assertIs(stored("bfloat16"), torch.bfloat16)
        self.assertIs(stored(torch.float16), torch.float16)
        self.assertIsNone(stored(None))
        self.assertNotIn("dtype", RepresentationConfig().intervention_kwargs)
        other_strings = {"precision": "float32", "act_fn": "float16"}
        self.assertEqual(RepresentationConfig(intervention_kwargs=other_strings).intervention_kwargs,
                         other_strings)
        for bad in ("fp32", "torch.float32", "Tensor", 32):
            with self.assertRaisesRegex(ValueError, r"intervention_kwargs\.dtype"):
                RepresentationConfig(intervention_kwargs={"dtype": bad})
        records = IntervenableConfig(representations=[
            RepresentationConfig(intervention_kwargs={"dtype": torch.float64}),
            RepresentationConfig(intervention_kwargs={"dtype": None}),
            RepresentationConfig(),
        ]).to_dict()["representations"]
        self.assertEqual([r["intervention_kwargs"] for r in records],
                         [{"dtype": "float64"}, {"dtype": None}, {}])
        with self.assertRaisesRegex(TypeError, "intervention_kwargs"):
            RepresentationConfig(dtype=torch.float32)

    def test_reserved_and_unknown_arguments(self):
        for name in ("layer", "low_rank_dimension", "intervention_link_key", "embed_dim", "use_fast"):
            with self.assertRaisesRegex(ValueError, rf"intervention_kwargs\.{name}"):
                RepresentationConfig(intervention_kwargs={name: 1})
        with self.assertRaisesRegex(ValueError, r"representations\[1\]\.intervention_kwargs\.embed_dim"):
            IntervenableConfig(representations=[
                {"layer": 0}, {"layer": 1, "intervention_kwargs": {"embed_dim": 4}}])
        with self.assertRaisesRegex(TypeError, "intervention_kwargs"):
            RepresentationConfig(0, "block_output", dropout=0.1)
        with self.assertRaisesRegex(ValueError, r"representations\[0\].*dropout"):
            IntervenableConfig(representations=[{"layer": 0, "dropout": 0.1}])
        # A collision introduced after construction is caught when building the model.
        representation = RepresentationConfig(0, "block_output", intervention_type=SCALED)
        representation.intervention_kwargs["use_fast"] = True
        with self.assertRaisesRegex(ValueError, r"representations\[0\]\.intervention_kwargs\.use_fast"):
            IntervenableModel(IntervenableConfig(representations=[representation]), make_mlp(3))

    def test_independent_defaults(self):
        first, second = IntervenableConfig(), IntervenableConfig()
        self.assertIsNot(first.representations, second.representations)
        self.assertIsNot(first.representations[0], second.representations[0])
        first.representations[0].layer = 5
        self.assertEqual(second.representations[0].layer, 0)
        self.assertEqual(IntervenableConfig().representations[0].layer, 0)
        a, b = RepresentationConfig(), RepresentationConfig()
        self.assertIsNot(a.intervention_kwargs, b.intervention_kwargs)
        a.intervention_kwargs["scale"] = 2.0
        self.assertEqual(b.intervention_kwargs, {})
        self.assertEqual(RepresentationConfig().intervention_kwargs, {})

    def test_caller_containers_not_mutated(self):
        nested = {"steps": [1, 2], "inner": {"a": [3]}}
        kwargs = {"scale": 0.5, "schedule": nested}
        record = {"layer": 0, "component": "block_output", "intervention_type": SCALED,
                  "intervention_kwargs": kwargs}
        legacy = [1, "block_output", "pos", 1, None, SCALED]
        snapshot = copy.deepcopy([record, legacy])
        config = IntervenableConfig(representations=[record, legacy])
        with tempfile.TemporaryDirectory() as directory:
            config.to_dict()
            config.save_pretrained(directory)
            IntervenableModel(config, make_mlp(4))
        self.assertEqual([record, legacy], snapshot)
        stored = config.representations[0].intervention_kwargs
        self.assertIsNot(stored, kwargs)
        self.assertIsNot(stored["schedule"]["steps"], nested["steps"])
        nested["steps"].append(99)
        self.assertEqual(stored["schedule"], {"steps": [1, 2], "inner": {"a": [3]}})

    # JSON round trips through the public Hugging Face entry points.

    def test_standalone_representation_round_trip(self):
        kwargs = {"scale": 0.25, "pairs": (1, 2), "dtype": torch.float64,
                  "schedule": {"steps": [1, 2], "decay": 0.5, "name": "cosine",
                               "warmup": True, "floor": None}}
        representation = RepresentationConfig(
            1, "block_output", "pos", 1, low_rank_dimension=2, intervention_type=SCALED,
            subspace_partition=[[0, 2], [2, 8]], intervention_kwargs=kwargs)
        with tempfile.TemporaryDirectory() as directory:
            representation.save_pretrained(directory)
            with open(os.path.join(directory, "config.json"), encoding="utf-8") as handle:
                written = json.load(handle)
            loaded = RepresentationConfig.from_pretrained(directory)
        self.assertEqual(set(written), set(FIELDS) | {"intervention_kwargs", "transformers_version"})
        self.assertEqual(written["intervention_type"], str(SCALED))
        self.assertEqual(written["subspace_partition"], [[0, 2], [2, 8]])
        self.assertEqual(written["intervention_kwargs"], {
            "scale": 0.25, "pairs": [1, 2], "dtype": "float64",
            "schedule": {"steps": [1, 2], "decay": 0.5, "name": "cosine",
                         "warmup": True, "floor": None}})
        for name in FIELDS:
            if name != "intervention_type":
                self.assertEqual(getattr(loaded, name), getattr(representation, name))
        self.assertIs(get_type_from_string(loaded.intervention_type), SCALED)
        self.assertIs(loaded.intervention_kwargs["dtype"], torch.float64)
        self.assertEqual(loaded.intervention_kwargs["pairs"], [1, 2])  # JSON has no tuples
        self.assertEqual(loaded.intervention_kwargs["schedule"], kwargs["schedule"])
        for key in HF_METADATA_KEYS:
            if key != "dtype":
                self.assertNotIn(key, loaded.to_intervention_kwargs())

    def test_nested_config_round_trip(self):
        config = IntervenableConfig(representations=[
            RepresentationConfig(0, "block_output", intervention_type=SCALED,
                                 intervention_kwargs={"scale": 0.25, "dtype": torch.float64}),
            RepresentationConfig(1, "block_output", low_rank_dimension=2,
                                 intervention_type=LowRankRotatedSpaceIntervention),
        ])
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            with open(os.path.join(directory, "config.json"), encoding="utf-8") as handle:
                written = json.load(handle)
            loaded = IntervenableConfig.from_pretrained(directory)
        self.assertEqual(written["representation_schema_version"], 1)
        type_names = [str(SCALED), str(LowRankRotatedSpaceIntervention)]
        self.assertEqual(written["intervention_types"], type_names)
        for record, type_name in zip(written["representations"], type_names):
            self.assertEqual(set(record), set(FIELDS) | {"intervention_kwargs"})
            self.assertEqual(record["intervention_type"], type_name)
        self.assertEqual(written["representations"][0]["intervention_kwargs"],
                         {"scale": 0.25, "dtype": "float64"})
        self.assertEqual(written["representations"][1]["low_rank_dimension"], 2)
        scaled, low_rank = IntervenableModel(loaded, make_mlp(5)).interventions.values()
        self.assertIsInstance(scaled, SCALED)
        self.assertIsInstance(low_rank, LowRankRotatedSpaceIntervention)
        self.assertEqual(set(scaled.received_kwargs),
                         set(FIELDS) | MODEL_KWARGS | {"scale", "dtype"})
        self.assertIs(scaled.received_kwargs["dtype"], torch.float64)
        self.assertEqual(tuple(low_rank.rotate_layer.weight.shape), (8, 2))

    def test_explicit_kwargs_drive_output(self):
        model = make_mlp(6)
        config = IntervenableConfig(representations=[RepresentationConfig(
            0, "block_output", intervention_type=SCALED, intervention_kwargs={"scale": 0.25})])
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            loaded = IntervenableConfig.from_pretrained(directory)
        base, (source,) = make_inputs(7)
        with torch.no_grad():
            h_base, h_source = model.h[0](base), model.h[0](source)
            golden = model.h[1](h_base + 0.25 * (h_source - h_base))
        for candidate in (config, loaded):
            torch.testing.assert_close(_run(IntervenableModel(candidate, model), base, source), golden)

    def test_schema_marker(self):
        legacy = {"representations": [[0, "block_output", "pos", 1]],
                  "intervention_types": [str(VanillaIntervention)]}
        with tempfile.TemporaryDirectory() as directory:
            for name, payload in (("legacy", legacy),
                                  ("future", dict(legacy, representation_schema_version=2))):
                os.makedirs(os.path.join(directory, name))
                with open(os.path.join(directory, name, "config.json"), "w", encoding="utf-8") as handle:
                    json.dump(payload, handle)
            loaded = IntervenableConfig.from_pretrained(os.path.join(directory, "legacy"))
            self.assertEqual(loaded.representations[0].component, "block_output")
            with self.assertRaisesRegex(ValueError, "representation_schema_version"):
                IntervenableConfig.from_pretrained(os.path.join(directory, "future"))
        self.assertEqual(IntervenableConfig().to_dict()["representation_schema_version"], 1)

    # Runtime acceptance versus persistence.

    def test_runtime_callable_works_but_is_not_persisted(self):
        model = make_mlp(8)
        config = IntervenableConfig(representations=[RepresentationConfig(
            0, "block_output", intervention_type=SCALED,
            intervention_kwargs={"scale": 0.5, "activation": torch.tanh})])
        intervenable = IntervenableModel(config, model)
        base, (source,) = make_inputs(9)
        with torch.no_grad():
            h_base, h_source = model.h[0](base), model.h[0](source)
            golden = model.h[1](torch.tanh(h_base + 0.5 * (h_source - h_base)))
        torch.testing.assert_close(_run(intervenable, base, source), golden)
        path = r"representations\[0\]\.intervention_kwargs\.activation"
        with self.assertRaisesRegex(ValueError, path):
            config.to_dict()
        with tempfile.TemporaryDirectory() as directory:
            existing = os.path.join(directory, "config.json")
            with open(existing, "w", encoding="utf-8") as handle:
                handle.write("{}")
            with self.assertRaisesRegex(ValueError, path):
                config.save_pretrained(directory)
            with open(existing, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "{}")
            archive = os.path.join(directory, "archive")
            with self.assertRaisesRegex(ValueError, path):
                intervenable.save(archive)
            self.assertFalse(os.path.exists(archive))
        with self.assertRaisesRegex(ValueError, r"intervention_kwargs\.weights\[0\]"):
            RepresentationConfig(intervention_kwargs={"weights": [torch.ones(2)]}).to_dict()

    def test_source_tensor_function_and_instance_boundaries(self):
        source = RepresentationConfig(0, "block_output", source_representation=torch.ones(8))
        with self.assertRaisesRegex(ValueError, r"representations\[0\]\.source_representation"):
            IntervenableConfig(representations=[source]).to_dict()
        with self.assertRaisesRegex(ValueError, "hidden_source_representation"):
            RepresentationConfig(0, hidden_source_representation=True).to_dict()
        # A runtime function stays usable and is only rejected when persisted.
        model = make_mlp(10)
        function_config = IntervenableConfig(representations=[RepresentationConfig(
            0, "block_output", intervention=lambda b, s: b * 0.5)])
        base, _ = make_inputs(11, 0)
        _, out = IntervenableModel(function_config, model)(
            {"inputs_embeds": base}, unit_locations={"base": 0})
        with torch.no_grad():
            torch.testing.assert_close(out[0], model.h[1](model.h[0](base) * 0.5))
        with self.assertRaisesRegex(ValueError, r"representations\[0\]\.intervention"):
            function_config.to_dict()
        # A supplied instance is recorded by its class only and is never modified.
        instance = LowRankRotatedSpaceIntervention(embed_dim=8, low_rank_dimension=2)
        before = {k: v.clone() for k, v in instance.state_dict().items()}
        config = IntervenableConfig(representations=[RepresentationConfig(
            0, "block_output", low_rank_dimension=2, intervention=instance)])
        record = config.to_dict()["representations"][0]
        self.assertEqual(record["intervention_type"], str(LowRankRotatedSpaceIntervention))
        self.assertIsNone(record["intervention"])
        self.assertIs(config.representations[0].intervention, instance)
        intervenable = IntervenableModel(config, model)
        self.assertIs(list(intervenable.interventions.values())[0], instance)
        for name, value in instance.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_function_model_save_rejects_before_writing(self):
        model = make_mlp(10)
        config = IntervenableConfig(representations=[RepresentationConfig(
            0, "block_output", intervention=lambda b, s: b * 0.5)])
        intervenable = IntervenableModel(config, model)
        base, _ = make_inputs(11, 0)

        def assert_forward():
            with torch.no_grad():
                _, out = intervenable({"inputs_embeds": base}, unit_locations={"base": 0})
                torch.testing.assert_close(out[0], model.h[1](model.h[0](base) * 0.5))

        assert_forward()
        path = r"representations\[0\]\.intervention"
        with tempfile.TemporaryDirectory() as directory:
            fresh = os.path.join(directory, "archive")
            with self.assertRaisesRegex(ValueError, path):
                intervenable.save(fresh)
            self.assertFalse(os.path.exists(fresh))
            existing_files = {"config.json": b"existing config\n",
                              "intkey_sentinel.bin": b"existing weights\n"}
            for name, content in existing_files.items():
                with open(os.path.join(directory, name), "wb") as handle:
                    handle.write(content)
            with self.assertRaisesRegex(ValueError, path):
                intervenable.save(directory)
            self.assertEqual(sorted(os.listdir(directory)), sorted(existing_files))
            for name, content in existing_files.items():
                with open(os.path.join(directory, name), "rb") as handle:
                    self.assertEqual(handle.read(), content)
        assert_forward()

    def test_repr_with_runtime_values(self):
        representation = RepresentationConfig(
            0, intervention_kwargs={"dtype": torch.float16, "activation": torch.tanh})
        self.assertIn("intervention_kwargs", repr(representation))
        self.assertIn("float16", str(IntervenableConfig(representations=[representation])))


if __name__ == "__main__":
    unittest.main()
