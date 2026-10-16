#!/usr/bin/env python
"""Generate tiny legacy native ``IntervenableModel`` archives with an UNMODIFIED PyVene.

These archives are compatibility fixtures for the RepresentationConfig migration
(stanfordnlp/pyvene#157).  They must be produced by the *old* implementation,
which stores ``RepresentationConfig`` namedtuples as positional JSON lists, and
they must never be regenerated with a newer implementation.

The script refuses to run unless ``--pyvene-root`` is a git checkout whose HEAD
is exactly ``--expected-sha`` with no modified tracked files, and unless the
imported ``pyvene`` comes from that checkout and still exposes the namedtuple.

Example::

    git worktree add --detach ../pyvene-base 9e333904dcf9e597ca76170010d17f4d4580de8d
    python generate_legacy_archives.py --pyvene-root ../pyvene-base \
        --expected-sha 9e333904dcf9e597ca76170010d17f4d4580de8d --out OUT_DIR

Everything is CPU-only and tiny (an 8-dimensional, 2-layer MLP).
"""
import argparse
import contextlib
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys

MLP_KWARGS = dict(h_dim=8, n_layer=2, pdrop=0.0, include_emb=False)


def _git(root, *args):
    return subprocess.check_output(["git", "-C", root, *args], text=True).strip()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pyvene-root", required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = os.path.abspath(args.pyvene_root)
    head = _git(root, "rev-parse", "HEAD")
    if head != args.expected_sha:
        sys.exit(f"refusing to run: {root} is at {head}, expected {args.expected_sha}")
    modified = _git(root, "status", "--porcelain", "--untracked-files=no")
    if modified:
        sys.exit(f"refusing to run: tracked files are modified in {root}:\n{modified}")
    out_root = os.path.abspath(args.out)
    if os.path.exists(out_root):
        sys.exit(f"refusing to overwrite existing output directory {out_root}")

    sys.path.insert(0, root)
    import torch
    torch.set_num_threads(2)
    import transformers
    import numpy
    import pyvene as pv
    from pyvene.models.mlp.modelings_mlp import MLPConfig, MLPModel

    pyvene_dir = os.path.dirname(os.path.abspath(pv.__file__))
    if pyvene_dir != os.path.join(root, "pyvene"):
        sys.exit(f"refusing to run: imported pyvene from {pyvene_dir}, not {root}")
    if not isinstance(pv.RepresentationConfig(), tuple):
        sys.exit("refusing to run: this PyVene no longer uses the legacy namedtuple")

    def make_base_model(seed):
        torch.manual_seed(seed)
        model = MLPModel(MLPConfig(**MLP_KWARGS))
        # The default HF initialisation of this toy model produces activations
        # around 1e-4; re-initialise deterministically so that numeric
        # comparisons of intervened outputs are meaningful.
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for _, param in model.named_parameters():
                param.copy_(torch.randn(param.shape, generator=generator) * 0.5)
        model.eval()
        return model

    def make_inputs(seed, n_sources):
        generator = torch.Generator().manual_seed(seed)
        base = torch.randn(2, 1, 8, generator=generator)
        sources = [torch.randn(2, 1, 8, generator=generator) for _ in range(n_sources)]
        return base, sources

    def wrap(tensor):
        return {"inputs_embeds": tensor}

    def train_one_step(intervenable, base, sources, unit_locations, seed,
                       allow_mutation=False):
        # Linked representations share one intervention; de-duplicate its parameters.
        params = list({id(p): p for p in intervenable.get_trainable_parameters()}.values())
        optimizer = torch.optim.SGD(params, lr=0.5)
        # AutoencoderIntervention.forward writes in place into a ReLU output, so a
        # plain backward() raises; the public autograd context below lets the
        # unchanged intervention math be differentiated.
        context = (torch.autograd.graph.allow_mutation_on_saved_tensors()
                   if allow_mutation else contextlib.nullcontext())
        with context:
            _, out = intervenable(wrap(base), [wrap(s) for s in sources],
                                  unit_locations=unit_locations)
            target = torch.randn(out[0].shape, generator=torch.Generator().manual_seed(seed))
            loss = torch.nn.functional.mse_loss(out[0], target)
            loss.backward()
        grads = [p.grad.detach().clone() for p in params]
        before = [p.detach().clone() for p in params]
        optimizer.step()
        optimizer.zero_grad()
        change = max((p.detach() - b).abs().max().item() for p, b in zip(params, before))
        grad_max = max(g.abs().max().item() for g in grads)
        if not (all(torch.isfinite(g).all() for g in grads) and grad_max > 0 and change > 0):
            raise RuntimeError("training step did not produce finite non-zero updates")
        return {"loss": loss.item(), "grad_abs_max": grad_max,
                "param_abs_change_max": change, "optimizer": "SGD(lr=0.5)",
                "autograd_context": ("allow_mutation_on_saved_tensors"
                                     if allow_mutation else "default")}

    def record(name, description, intervenable, model, base, sources,
               unit_locations, training):
        archive_dir = os.path.join(out_root, name, "archive")
        intervenable.save(save_directory=archive_dir)
        with torch.no_grad():
            original, intervened = intervenable(
                wrap(base), None if sources is None else [wrap(s) for s in sources],
                unit_locations=unit_locations, output_original_output=True)
        reference = {
            "mlp_kwargs": dict(MLP_KWARGS),
            "base_model_state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
            "inputs": {"base": base, "sources": sources},
            "unit_locations": unit_locations,
            "outputs": {"original": original[0].detach().clone(),
                        "intervened": intervened[0].detach().clone()},
            "intervention_keys": list(intervenable.interventions.keys()),
            "sorted_keys": list(intervenable.sorted_keys),
            "intervention_state_dicts": {
                k: {n: t.detach().clone() for n, t in v.state_dict().items()}
                for k, v in intervenable.interventions.items()},
            "training": training,
        }
        if torch.equal(reference["outputs"]["original"], reference["outputs"]["intervened"]):
            raise RuntimeError(f"{name}: intervention does not change the output")
        torch.save(reference, os.path.join(out_root, name, "reference.pt"))
        return {"description": description}

    os.makedirs(out_root)
    archives = {}

    # A. Two trainable low-rank rotated-space interventions (primary 14-field archive).
    model = make_base_model(1001)
    base, sources = make_inputs(2001, 2)
    config = pv.IntervenableConfig(
        representations=[
            pv.RepresentationConfig(0, "block_output", "pos", 1, low_rank_dimension=2),
            pv.RepresentationConfig(1, "block_output", "pos", 1, low_rank_dimension=2),
        ],
        intervention_types=pv.LowRankRotatedSpaceIntervention,
    )
    intervenable = pv.IntervenableModel(config, model)
    unit_locations = {"sources->base": ([[[0], [0]], [[0], [0]]], [[[0], [0]], [[0], [0]]])}
    training = train_one_step(intervenable, base, sources, unit_locations, 3001)
    archives["lowrank_two_layers"] = record(
        "lowrank_two_layers",
        "Two LowRankRotatedSpaceIntervention (rank 2) at block_output of layers 0 and 1, "
        "after one SGD step; latent_dim is None in every record.",
        intervenable, model, base, sources, unit_locations, training)

    # B. Constant source tensor stored in the intervention binary.
    model = make_base_model(1002)
    base, _ = make_inputs(2002, 0)
    source_vector = torch.randn(8, generator=torch.Generator().manual_seed(4002))
    config = pv.IntervenableConfig(
        representations=[pv.RepresentationConfig(
            0, "block_output", "pos", 1, source_representation=source_vector)],
        intervention_types=pv.VanillaIntervention,
    )
    intervenable = pv.IntervenableModel(config, model)
    archives["constant_source"] = record(
        "constant_source",
        "VanillaIntervention with a constant source_representation tensor at block_output "
        "of layer 0 (hidden_source_representation flag + tensor in intkey binary).",
        intervenable, model, base, None, {"base": 0}, None)

    # C. Linked pair sharing one intervention, declared in non-ascending layer order.
    model = make_base_model(1003)
    base, sources = make_inputs(2003, 2)
    config = pv.IntervenableConfig(
        representations=[
            pv.RepresentationConfig(1, "block_output", "pos", 1, low_rank_dimension=2,
                                    intervention_link_key=0),
            pv.RepresentationConfig(0, "block_output", "pos", 1, low_rank_dimension=2,
                                    intervention_link_key=0),
        ],
        intervention_types=pv.LowRankRotatedSpaceIntervention,
    )
    intervenable = pv.IntervenableModel(config, model)
    keys = list(intervenable.interventions.keys())
    if intervenable.interventions[keys[0]] is not intervenable.interventions[keys[1]]:
        raise RuntimeError("linked interventions are not shared")
    unit_locations = {"sources->base": ([[[0], [0]], [[0], [0]]], [[[0], [0]], [[0], [0]]])}
    training = train_one_step(intervenable, base, sources, unit_locations, 3003)
    archives["linked_pair"] = record(
        "linked_pair",
        "Two representations (layer 1 first, then layer 0) sharing one "
        "LowRankRotatedSpaceIntervention through intervention_link_key=0, after one SGD step.",
        intervenable, model, base, sources, unit_locations, training)

    # D. Autoencoder intervention with an explicit latent_dim (14th positional field).
    model = make_base_model(1004)
    base, sources = make_inputs(2004, 1)
    config = pv.IntervenableConfig(
        representations=[pv.RepresentationConfig(0, "block_output", "pos", 1, latent_dim=4)],
        intervention_types=pv.AutoencoderIntervention,
    )
    intervenable = pv.IntervenableModel(config, model)
    key = list(intervenable.interventions.keys())[0]
    # The default interchange_dim is embed_dim (8), which is not a valid latent index.
    intervenable.interventions[key].set_interchange_dim([0, 2])
    unit_locations = {"sources->base": ([[[0], [0]]], [[[0], [0]]])}
    training = train_one_step(intervenable, base, sources, unit_locations, 3004,
                              allow_mutation=True)
    archives["autoencoder_latent"] = record(
        "autoencoder_latent",
        "AutoencoderIntervention with latent_dim=4 at block_output of layer 0, latent "
        "interchange indices [0, 2], after one SGD step.",
        intervenable, model, base, sources, unit_locations, training)

    # 13-field historical-layout control derived from the genuine 14-field config.
    genuine_path = os.path.join(out_root, "lowrank_two_layers", "archive", "config.json")
    derived_path = os.path.join(out_root, "lowrank_two_layers", "config_13field_derived.json")
    with open(genuine_path, encoding="utf-8") as handle:
        derived = json.load(handle)
    for record_ in derived["representations"]:
        if len(record_) != 14 or record_[13] is not None:
            raise RuntimeError("expected 14-field records with latent_dim=None")
        del record_[13]
    with open(derived_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(derived, indent=2, sort_keys=True) + "\n")

    files = {}
    for dirpath, _, filenames in os.walk(out_root):
        for filename in sorted(filenames):
            path = os.path.join(dirpath, filename)
            files[os.path.relpath(path, out_root)] = _sha256(path)
    provenance = {
        "purpose": "Legacy native IntervenableModel archives produced by the unmodified base "
                   "implementation (positional RepresentationConfig records).",
        "generator": os.path.basename(__file__),
        "generator_sha256": _sha256(os.path.abspath(__file__)),
        "command": [os.path.basename(sys.executable)] + sys.argv,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pyvene_commit": head,
        "pyvene_module": os.path.relpath(pv.__file__, root),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": numpy.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "base_model": {"class": "pyvene.models.mlp.modelings_mlp.MLPModel",
                       "config": dict(MLP_KWARGS),
                       "reinitialisation": "every parameter <- 0.5 * randn(seeded generator)"},
        "archives": archives,
        "derived_13field_fixture": {
            "path": "lowrank_two_layers/config_13field_derived.json",
            "source": "lowrank_two_layers/archive/config.json",
            "transformation": "json.load(source); for each record in 'representations': assert "
                              "len(record) == 14 and record[13] is None (latent_dim); delete "
                              "record[13]; json.dumps(indent=2, sort_keys=True) + newline. "
                              "No other key or value is changed.",
            "note": "Historical-layout control derived from the genuine base-produced 14-field "
                    "archive; it is NOT an independently produced historical archive.",
        },
        "files_sha256": files,
    }
    with open(os.path.join(out_root, "provenance.json"), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": out_root, "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
