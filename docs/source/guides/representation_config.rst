Representation configs: extending, saving and loading
=====================================================

Each intervention in an :class:`IntervenableConfig
<pyvene.models.configuration_intervenable_model.IntervenableConfig>` is described by a
:class:`RepresentationConfig <pyvene.models.configuration_intervenable_model.RepresentationConfig>`.
Both are Hugging Face ``PretrainedConfig`` subclasses.

Fields and intervention arguments
---------------------------------

``RepresentationConfig`` keeps the 14 fields of the former namedtuple, in the same
positional order and with the same defaults::

   layer=0, component="block_output", unit="pos", max_number_of_units=1,
   low_rank_dimension=None, intervention_type=None, intervention=None,
   subspace_partition=None, group_key=None, intervention_link_key=None,
   moe_key=None, source_representation=None, hidden_source_representation=None,
   latent_dim=None

Any other argument of the intervention class goes into the keyword-only
``intervention_kwargs`` mapping. When the intervenable model builds the intervention,
it passes the 14 fields, the entries of ``intervention_kwargs`` and the values it
computes itself (``embed_dim`` and ``use_fast``). ``RepresentationConfig.to_intervention_kwargs()``
returns these arguments without the computed values. Hugging Face metadata such as the
config's own ``dtype`` or ``transformers_version`` is never passed.

An ``intervention_kwargs`` key that names one of the 14 fields, ``embed_dim`` or
``use_fast`` is rejected, and so is an unknown top-level keyword argument (for
example ``RepresentationConfig(dropout=0.1)``).

``intervention_kwargs["dtype"]`` accepts a ``torch.dtype``, a torch dtype name such as
``"bfloat16"``, or ``None``, and is stored as a ``torch.dtype``. Leaving it out and
passing ``None`` are different: when it is left out, the intervention class picks its
own default.

For example, with `pyreft <https://github.com/stanfordnlp/pyreft>`_:

.. code:: python

   import torch
   import pyreft

   reft_config = pyreft.ReftConfig(representations={
       "layer": 15, "component": "block_output", "low_rank_dimension": 4,
       "intervention_type": pyreft.NoreftIntervention,
       "intervention_kwargs": {"dropout": 0.05, "act_fn": "relu",
                               "add_bias": False, "dtype": torch.bfloat16},
   })
   reft_model = pyreft.get_reft_model(model, reft_config)
   # ... train ...
   reft_model.save("my_reft")
   reft_model = pyreft.ReftModel.load("my_reft", model)  # dropout, act_fn, add_bias, dtype restored

Saved format
------------

``IntervenableModel.save()`` and ``IntervenableConfig.save_pretrained()`` write
``"representation_schema_version": 1`` and every representation as a named record.
The record saved for the example above is (shown in field order; the file sorts keys):

.. code:: json

   {"layer": 15, "component": "block_output", "unit": "pos", "max_number_of_units": 1,
    "low_rank_dimension": 4, "intervention_type": null, "intervention": null,
    "subspace_partition": null, "group_key": null, "intervention_link_key": null,
    "moe_key": null, "source_representation": null, "hidden_source_representation": null,
    "latent_dim": null,
    "intervention_kwargs": {"act_fn": "relu", "add_bias": false, "dropout": 0.05,
                            "dtype": "bfloat16"}}

In a model archive the intervention classes are listed in ``intervention_types``, and
trained weights and constant source tensors are stored in ``intkey_<key>.bin`` files,
as before. In a standalone ``save_pretrained()`` file, ``intervention_type`` holds the
class as ``"<class 'module.ClassName'>"``. The class must be importable under that name
when the file is loaded.

Files without ``representation_schema_version`` are legacy files. Their
representations are positional lists in the field order above, with 13 entries (written
before ``latent_dim`` existed) or 14 entries. pyvene still reads them and always writes
the named form, which older pyvene releases cannot read. Any other schema version is
rejected.

Runtime objects and persistence
-------------------------------

At runtime, a representation can still hold an intervention instance or function, a
source tensor, or non-JSON values in ``intervention_kwargs`` (for example a callable).
Only JSON values can be saved: ``None``, booleans, numbers, strings, lists and dicts
with string keys, plus a ``torch.dtype`` in ``intervention_kwargs["dtype"]``. Tuples
are saved as lists. Saving a configuration that holds any other value raises a
``ValueError`` that names the value, for example
``representations[0].intervention_kwargs.activation``, before any file is written.

Source tensors are saved only by ``IntervenableModel.save()``, which stores them in the
intervention's binary file. A standalone configuration cannot store them.

A supplied intervention instance is saved as its class together with the
representation's fields and ``intervention_kwargs``. pyvene does not infer the
instance's own constructor arguments. A config that passed settings only through an
instance, such as ``LoreftIntervention(..., dropout=0.1)``, therefore cannot be
reconstructed with those settings. Use ``intervention_type`` and
``intervention_kwargs`` instead.

Differences from the former namedtuple
--------------------------------------

Positional and keyword construction, attribute access and ``_asdict()`` still work.
``_asdict()`` returns the 14 fields and leaves out ``intervention_kwargs``. A
``RepresentationConfig`` is no longer a tuple, so indexing, unpacking, ``_replace``,
tuple equality, hashing and immutability are not available. As with any
``PretrainedConfig``, iterating over it yields attribute names. Pickles of old
namedtuple objects cannot be loaded.
