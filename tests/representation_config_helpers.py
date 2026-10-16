"""Helpers shared by the RepresentationConfig tests (stanfordnlp/pyvene#157).

The intervention below lives in a normally importable module, so its
"<class 'tests.representation_config_helpers.ScaledInterpolationIntervention'>"
name can be resolved again after a JSON round trip.
"""
import os

import torch

from pyvene.models.interventions import Intervention
from pyvene.models.mlp.modelings_mlp import MLPConfig, MLPModel


LEGACY_ARCHIVES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures", "legacy_native_archives")
MLP_KWARGS = dict(h_dim=8, n_layer=2, pdrop=0.0, include_emb=False)
# Arguments that Hugging Face configs carry but that must never reach an
# intervention constructor.
HF_METADATA_KEYS = (
    "dtype", "torch_dtype", "transformers_version", "model_type", "architectures",
    "_name_or_path", "name_or_path", "return_dict", "output_hidden_states",
    "id2label", "label2id", "_commit_hash", "attn_implementation",
)


def make_mlp(seed=0, state_dict=None):
    """A tiny config-only MLPModel with deterministic, non-trivial weights."""
    torch.manual_seed(seed)
    model = MLPModel(MLPConfig(**MLP_KWARGS))
    if state_dict is not None:
        model.load_state_dict(state_dict)
    else:
        # the default initialisation gives activations around 1e-4
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for _, param in model.named_parameters():
                param.copy_(torch.randn(param.shape, generator=generator) * 0.5)
    return model.eval()


def make_inputs(seed, n_sources=1):
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(2, 1, 8, generator=generator)
    sources = [torch.randn(2, 1, 8, generator=generator) for _ in range(n_sources)]
    return base, sources


def unit_locations(n_interventions):
    """Position 0 of both examples, for every intervention."""
    def locations():
        return [[[0], [0]] for _ in range(n_interventions)]
    return {"sources->base": (locations(), locations())}


class ScaledInterpolationIntervention(Intervention):
    """``base + scale * (source - base)``, optionally passed through ``activation``.

    The constructor keyword arguments are recorded in ``received_kwargs`` so
    that tests can check exactly what the intervenable model passed.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.received_kwargs = dict(kwargs)
        self.scale = float(kwargs.get("scale", 1.0))
        self.activation = kwargs.get("activation")
        if kwargs.get("dtype") is not None:
            self.register_buffer("dtype_probe", torch.zeros(1, dtype=kwargs["dtype"]))

    def forward(self, base, source, subspaces=None, **kwargs):
        output = base + self.scale * (source - base)
        if self.activation is not None:
            output = self.activation(output)
        return output
