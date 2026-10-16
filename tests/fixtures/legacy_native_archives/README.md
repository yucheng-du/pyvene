# Legacy native archives (compatibility fixtures)

These tiny `IntervenableModel.save()` archives were produced by the **unmodified**
pyvene commit `9e333904dcf9e597ca76170010d17f4d4580de8d`, whose
`RepresentationConfig` was a namedtuple stored as a positional JSON list.
`tests/integration_tests/RepresentationConfigSaveLoadTestCase.py` loads them
with the current code to check backward compatibility. Do not regenerate them
with a newer pyvene: they are only meaningful as artifacts of the old writer.

| Directory | Content |
|---|---|
| `lowrank_two_layers/` | Two trained `LowRankRotatedSpaceIntervention` (14-field records) |
| `constant_source/` | Constant `source_representation` (hidden flag + tensor in `intkey_*.bin`) |
| `linked_pair/` | Two representations sharing one intervention via `intervention_link_key`, declared layer 1 first |
| `autoencoder_latent/` | Trained `AutoencoderIntervention` with `latent_dim=4` (14th positional field) |

Each directory holds `archive/` (the exact `save()` output) and `reference.pt`
(base-model weights, inputs, unit locations, intervention states and the
reference outputs computed by the old code).

`lowrank_two_layers/config_13field_derived.json` is a **derived** historical-layout
control: the genuine 14-field `config.json` with only the trailing
`latent_dim=None` removed from every record (the 13-field layout written before
`latent_dim` existed). It is not an independently produced historical archive.

`generate_legacy_archives.py` is the generator that was run (CPU, PyTorch 2.6.0,
transformers 4.57.1); `provenance.json` records the command, environment and the
SHA-256 of every file.
