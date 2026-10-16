import copy, json, warnings, torch
from collections import OrderedDict
from typing import Any, List, Mapping, Optional

from transformers import PreTrainedTokenizer, TensorType, is_torch_available
from transformers import __version__ as transformers_version
from transformers.configuration_utils import PretrainedConfig

from .interventions import VanillaIntervention


# Declared RepresentationConfig fields in their historical positional order.
# Legacy configuration files store each representation as a positional list in
# this order: 13 entries before ``latent_dim`` was appended, 14 afterwards.
REPRESENTATION_FIELDS = (
    "layer", "component", "unit", "max_number_of_units",
    "low_rank_dimension", "intervention_type", "intervention",
    "subspace_partition", "group_key", "intervention_link_key", "moe_key",
    "source_representation", "hidden_source_representation", "latent_dim",
)
# Keys of a serialized representation record.
REPRESENTATION_RECORD_KEYS = REPRESENTATION_FIELDS + ("intervention_kwargs",)
# Intervention constructor arguments that the intervenable model computes itself.
MODEL_COMPUTED_INTERVENTION_KWARGS = ("embed_dim", "use_fast")
# Marker written by IntervenableConfig.to_dict(). Files without it are legacy
# files (positional records); any other value is rejected.
REPRESENTATION_SCHEMA_KEY = "representation_schema_version"
REPRESENTATION_SCHEMA_VERSION = 1
# Hugging Face bookkeeping that from_pretrained()/from_dict() pass back to
# RepresentationConfig; keys starting with "_" are accepted as well.
_HF_LOADING_KEYS = ("transformers_version", "attn_implementation", "name_or_path")


def _field_path(path, name):
    return f"{path}.{name}" if path else name


def _decode_dtype(value, path):
    """Return ``intervention_kwargs["dtype"]`` as a ``torch.dtype`` (or None)."""
    if value is None or isinstance(value, torch.dtype):
        return value
    if isinstance(value, str) and isinstance(getattr(torch, value, None), torch.dtype):
        return getattr(torch, value)
    raise ValueError(
        f"{path}: expected a torch.dtype, a torch dtype name such as 'float32', "
        f"or None; got {value!r}")


def _encode_dtype(value, path):
    """Return the JSON form of ``intervention_kwargs["dtype"]``, e.g. 'float32'."""
    value = _decode_dtype(value, path)
    return None if value is None else str(value).split(".", 1)[1]


def _to_json(value, path):
    """Return a JSON-compatible copy of ``value``; raise naming ``path`` otherwise."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_json(item, f"{path}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, Mapping):
        encoded = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path}: JSON object keys must be strings, got {key!r}")
            encoded[key] = _to_json(item, f"{path}.{key}")
        return encoded
    raise ValueError(
        f"{path}: a {type(value).__name__} value cannot be stored in a JSON "
        "configuration; only None, bool, int, float, str, lists and dicts "
        "(and a torch.dtype for intervention_kwargs['dtype']) are supported")


def _encode_type(value, path):
    """Encode an intervention class with the importable "<class 'module.Name'>"
    convention read by ``basic_utils.get_type_from_string``."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, type):
        return str(value)
    raise ValueError(f"{path}: expected an importable class or None, got {value!r}")


def _copy_containers(value):
    """Copy plain list/tuple/dict containers; keep every other object as is."""
    if type(value) is list:
        return [_copy_containers(item) for item in value]
    if type(value) is tuple:
        return tuple(_copy_containers(item) for item in value)
    if type(value) is dict:
        return {key: _copy_containers(item) for key, item in value.items()}
    return value


def _check_intervention_kwarg_name(name, path):
    if not isinstance(name, str):
        raise TypeError(f"{path}: argument names must be strings, got {name!r}")
    if name in REPRESENTATION_FIELDS:
        raise ValueError(
            f"{path}.{name}: '{name}' is a RepresentationConfig field; set it on "
            "the representation instead of in intervention_kwargs")
    if name in MODEL_COMPUTED_INTERVENTION_KWARGS:
        raise ValueError(
            f"{path}.{name}: '{name}' is computed by the intervenable model and "
            "cannot be set in intervention_kwargs")


def _normalize_intervention_kwargs(intervention_kwargs, path):
    """Validate and copy an ``intervention_kwargs`` mapping."""
    if intervention_kwargs is None:
        return {}
    if not isinstance(intervention_kwargs, Mapping):
        raise TypeError(
            f"{path}: expected a mapping of intervention constructor arguments, "
            f"got {type(intervention_kwargs).__name__}")
    normalized = {}
    for name, value in intervention_kwargs.items():
        _check_intervention_kwarg_name(name, path)
        normalized[name] = _copy_containers(value)
    if "dtype" in normalized:
        normalized["dtype"] = _decode_dtype(normalized["dtype"], f"{path}.dtype")
    return normalized


def _intervention_constructor_kwargs(representation, path=""):
    """Keyword arguments for constructing the intervention of ``representation``:
    the declared fields (as forwarded by previous releases) plus the explicit
    ``intervention_kwargs``. Hugging Face metadata is never included."""
    kwargs_path = _field_path(path, "intervention_kwargs")
    extra = representation.intervention_kwargs
    if not isinstance(extra, Mapping):
        raise TypeError(f"{kwargs_path}: expected a mapping, got {type(extra).__name__}")
    constructor_kwargs = representation._asdict()
    for name, value in extra.items():
        _check_intervention_kwarg_name(name, kwargs_path)
        if name == "dtype":
            value = _decode_dtype(value, f"{kwargs_path}.dtype")
        constructor_kwargs[name] = value
    return constructor_kwargs


def representation_to_record(representation, path="", model_archive=False):
    """Encode a RepresentationConfig as a named JSON record.

    Values that cannot be stored raise ``ValueError`` naming their path. In a
    standalone configuration a supplied ``intervention`` instance is recorded
    by its class (its own constructor arguments are not inferred), and source
    tensors are rejected. With ``model_archive=True`` (used by
    ``IntervenableModel.save``) intervention types are recorded separately in
    ``IntervenableConfig.intervention_types`` and a source tensor, which is
    stored in the ``intkey_<key>.bin`` file, is replaced by
    ``hidden_source_representation=True``.
    """
    record = {}
    for name in REPRESENTATION_FIELDS:
        if name not in ("intervention_type", "intervention", "source_representation",
                        "hidden_source_representation"):
            record[name] = _to_json(getattr(representation, name), _field_path(path, name))
    intervention_type = representation.intervention_type
    intervention = representation.intervention
    source = representation.source_representation
    if intervention is not None and not isinstance(intervention, torch.nn.Module):
        raise ValueError(
            f"{_field_path(path, 'intervention')}: a {type(intervention).__name__} "
            "is a runtime-only intervention and cannot be stored in a JSON "
            "configuration")
    if model_archive:
        record["intervention_type"] = None
        record["intervention"] = None
        record["source_representation"] = None
        record["hidden_source_representation"] = True if source is not None else None
    else:
        if intervention is not None:
            if intervention_type is not None:
                raise ValueError(
                    f"{_field_path(path, 'intervention')}: only one of the fields "
                    "intervention_type and intervention can be serialized")
            intervention_type = type(intervention)
        if source is not None:
            raise ValueError(
                f"{_field_path(path, 'source_representation')}: source tensors are "
                "only persisted by IntervenableModel.save(), which stores them in "
                "intkey_<key>.bin files; a standalone configuration cannot contain them")
        if representation.hidden_source_representation:
            raise ValueError(
                f"{_field_path(path, 'hidden_source_representation')}: refers to a "
                "source tensor stored in a model archive; a standalone configuration "
                "cannot contain it")
        record["intervention_type"] = _encode_type(
            intervention_type, _field_path(path, "intervention_type"))
        record["intervention"] = None
        record["source_representation"] = None
        record["hidden_source_representation"] = _to_json(
            representation.hidden_source_representation,
            _field_path(path, "hidden_source_representation"))
    kwargs_path = _field_path(path, "intervention_kwargs")
    extra = representation.intervention_kwargs
    if not isinstance(extra, Mapping):
        raise TypeError(f"{kwargs_path}: expected a mapping, got {type(extra).__name__}")
    encoded_kwargs = {}
    for name, value in extra.items():
        _check_intervention_kwarg_name(name, kwargs_path)
        if name == "dtype":
            encoded_kwargs[name] = _encode_dtype(value, f"{kwargs_path}.dtype")
        else:
            encoded_kwargs[name] = _to_json(value, f"{kwargs_path}.{name}")
    record["intervention_kwargs"] = encoded_kwargs
    return {name: record[name] for name in REPRESENTATION_RECORD_KEYS}


def _as_representation_config(representation, path):
    """Normalize one supported representation input exactly once."""
    if isinstance(representation, RepresentationConfig):
        return representation
    if isinstance(representation, list):
        # Legacy positional record; intervention_kwargs cannot be positional.
        return RepresentationConfig(*representation)
    if isinstance(representation, dict):
        unknown = sorted(set(representation) - set(REPRESENTATION_RECORD_KEYS))
        if unknown:
            raise ValueError(
                f"{path}: unknown representation field(s) {unknown}; intervention "
                "constructor arguments belong in 'intervention_kwargs'")
        record = dict(representation)
        record["intervention_kwargs"] = _normalize_intervention_kwargs(
            representation.get("intervention_kwargs"), f"{path}.intervention_kwargs")
        return RepresentationConfig(**record)
    raise ValueError(
        f"{representation} format in our representation list is not supported.")


class RepresentationConfig(PretrainedConfig):
    r"""
    Where and how one intervention is applied.

    The 14 declared fields keep the positional order and defaults of the former
    ``RepresentationConfig`` namedtuple. Extra intervention constructor
    arguments go into the keyword-only ``intervention_kwargs`` mapping, e.g.
    ``intervention_kwargs={"dropout": 0.1, "act_fn": "relu", "dtype": torch.float32}``;
    they are passed to the intervention class together with the declared
    fields. ``intervention_kwargs["dtype"]`` accepts a ``torch.dtype``, a torch
    dtype name or None, and is stored as a ``torch.dtype``.

    This is a Hugging Face ``PretrainedConfig``, not a tuple: positional and
    keyword construction, attribute access and ``_asdict()`` are supported,
    while indexing, unpacking, ``_replace`` and tuple equality/hashing are not.
    Use ``to_intervention_kwargs()`` for intervention constructor arguments,
    never ``vars()`` or ``to_dict()``.
    """

    _fields = REPRESENTATION_FIELDS

    def __init__(
        self,
        layer=0,
        component="block_output",
        unit="pos",
        max_number_of_units=1,
        low_rank_dimension=None,
        intervention_type=None,
        intervention=None,
        subspace_partition=None,
        group_key=None,
        intervention_link_key=None,
        moe_key=None,
        source_representation=None,
        hidden_source_representation=None,
        latent_dim=None,
        *,
        intervention_kwargs=None,
        **kwargs,
    ):
        unexpected = sorted(
            key for key in kwargs
            if key not in _HF_LOADING_KEYS and not key.startswith("_"))
        if unexpected:
            raise TypeError(
                f"RepresentationConfig got unexpected keyword argument(s) {unexpected}; "
                "intervention constructor arguments belong in intervention_kwargs")
        self.layer = layer
        self.component = component
        self.unit = unit
        self.max_number_of_units = max_number_of_units
        self.low_rank_dimension = low_rank_dimension
        self.intervention_type = intervention_type
        self.intervention = intervention
        self.subspace_partition = subspace_partition
        self.group_key = group_key
        self.intervention_link_key = intervention_link_key
        self.moe_key = moe_key
        self.source_representation = source_representation
        self.hidden_source_representation = hidden_source_representation
        self.latent_dim = latent_dim
        self.intervention_kwargs = _normalize_intervention_kwargs(
            intervention_kwargs, "intervention_kwargs")
        super().__init__(**kwargs)

    def _asdict(self):
        """The 14 declared fields, as returned by the former namedtuple.

        ``intervention_kwargs`` is not included; see ``to_intervention_kwargs()``.
        """
        return {name: getattr(self, name) for name in REPRESENTATION_FIELDS}

    def to_intervention_kwargs(self):
        """Keyword arguments for this representation's intervention constructor:
        the declared fields merged with ``intervention_kwargs``. The intervenable
        model adds ``embed_dim`` and ``use_fast``."""
        return _intervention_constructor_kwargs(self)

    def to_dict(self):
        """The named JSON record of this representation plus the transformers
        version. Raises ``ValueError`` for values that cannot be stored."""
        output = representation_to_record(self)
        output["transformers_version"] = transformers_version
        return output

    def to_diff_dict(self):
        # Every record field is always written, and the free-form
        # intervention_kwargs must not be rewritten by recursive dict helpers.
        return self.to_dict()

    def to_json_file(self, json_file_path, use_diff=True):
        # Encode before opening: an unsupported value must not truncate a file.
        json_string = self.to_json_string(use_diff=use_diff)
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(json_string)

    def __repr__(self):
        fields = ", ".join(f"{name}={getattr(self, name)!r}" for name in REPRESENTATION_FIELDS)
        return (f"{self.__class__.__name__}({fields}, "
                f"intervention_kwargs={self.intervention_kwargs!r})")


class IntervenableConfig(PretrainedConfig):
    def __init__(
        self,
        representations=None,
        intervention_types=VanillaIntervention,
        mode="parallel",
        sorted_keys=None,
        model_type=None, # deprecating
        # hidden fields for backlog
        intervention_dimensions=None,
        intervention_constant_sources=None,
        **kwargs,
    ):
        schema_version = kwargs.pop(REPRESENTATION_SCHEMA_KEY, None)
        if schema_version is not None and schema_version != REPRESENTATION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported {REPRESENTATION_SCHEMA_KEY}={schema_version!r}: this "
                f"version of pyvene reads version {REPRESENTATION_SCHEMA_VERSION} and "
                "legacy configurations without the marker.")
        if representations is None:
            representations = [RepresentationConfig()]
        if not isinstance(representations, list):
            representations = [representations]

        self.representations = [
            _as_representation_config(reprs, f"representations[{i}]")
            for i, reprs in enumerate(representations)
        ]
        self.intervention_types = intervention_types
        # the type inside reprs can overwrite
        overwrite = False
        overwrite_intervention_types = []
        for reprs in self.representations:
            if overwrite:
                if reprs.intervention_type is None and reprs.intervention is None:
                    raise ValueError(
                        "intervention_type if used should be specified for all")
            if reprs.intervention_type is not None:
                overwrite = True
                overwrite_intervention_types += [reprs.intervention_type]
            elif reprs.intervention is not None:
                overwrite = True
                overwrite_intervention_types += [type(reprs.intervention)]
            if reprs.intervention_type is not None and reprs.intervention is not None:
                raise ValueError(
                    "Only one of the field should be provided: intervention_type, intervention")
        if None in overwrite_intervention_types:
            raise ValueError(
                "intervention_type if used should be specified for all")
        if overwrite:
            self.intervention_types = overwrite_intervention_types

        self.mode = mode
        self.sorted_keys = sorted_keys
        self.intervention_dimensions = intervention_dimensions
        self.intervention_constant_sources = intervention_constant_sources
        self.model_type = model_type
        super().__init__(**kwargs)

    def add_intervention(self, representations):
        if not isinstance(representations, list):
            representations = [representations]

        for reprs in representations:
            self.representations += [_as_representation_config(
                reprs, f"representations[{len(self.representations)}]")]
            if self.representations[-1].intervention_type is None:
                raise ValueError(
                    "intervention_type should be provided.")

            if self.representations[-1].intervention_type is not None:
                self.intervention_types += [self.representations[-1].intervention_type]
            elif self.representations[-1].intervention is not None:
                self.intervention_types += [self.representations[-1].intervention]

    def to_dict(self):
        """Serialize to a JSON-compatible dict: representations become named
        records, intervention classes "<class 'module.Name'>" strings, and the
        schema marker is added. Raises ``ValueError`` naming the path of any
        value that cannot be stored."""
        records = []
        for i, reprs in enumerate(self.representations):
            path = f"representations[{i}]"
            if isinstance(reprs, RepresentationConfig):
                records.append(representation_to_record(reprs, path))
            elif isinstance(reprs, dict) and set(reprs) <= set(REPRESENTATION_RECORD_KEYS):
                # An already encoded record, as prepared by IntervenableModel.save().
                records.append(_to_json(reprs, path))
            else:
                raise ValueError(f"{path}: {type(reprs).__name__} cannot be serialized")
        if isinstance(self.intervention_types, list):
            intervention_types = [
                _encode_type(t, f"intervention_types[{i}]")
                for i, t in enumerate(self.intervention_types)]
        else:
            intervention_types = _encode_type(self.intervention_types, "intervention_types")
        # Let PretrainedConfig serialize everything else from a shallow copy so
        # that runtime objects of this config are neither copied nor modified.
        serializable = copy.copy(self)
        serializable.representations = records
        serializable.intervention_types = intervention_types
        output = super(IntervenableConfig, serializable).to_dict()
        output[REPRESENTATION_SCHEMA_KEY] = REPRESENTATION_SCHEMA_VERSION
        return output

    def to_json_file(self, json_file_path, use_diff=True):
        # Encode before opening: an unsupported value must not truncate a file.
        json_string = self.to_json_string(use_diff=use_diff)
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(json_string)

    def __repr__(self):
        representations = []
        for reprs in self.representations:
            if isinstance(reprs, list):
                reprs = RepresentationConfig(*reprs)
            new_d = {}
            for k, v in reprs._asdict().items():
                if type(v) not in {str, int, list, tuple, dict} and v is not None and v != [None]:
                    new_d[k] = "PLACEHOLDER"
                else:
                    new_d[k] = v
            new_d["intervention_kwargs"] = reprs.intervention_kwargs
            representations += [new_d]
        _repr = {
            "model_type": str(self.model_type),
            "representations": tuple(representations),
            "intervention_types": str(
                self.intervention_types
            ),
            "mode": self.mode,
            "sorted_keys": tuple(self.sorted_keys) if self.sorted_keys is not None else str(self.sorted_keys),
            "intervention_dimensions": str(self.intervention_dimensions),
        }
        # default=str only affects display of values such as torch dtypes.
        _repr_string = json.dumps(_repr, indent=4, default=str)

        return f"IntervenableConfig\n{_repr_string}"

    def __str__(self):
        return self.__repr__()
