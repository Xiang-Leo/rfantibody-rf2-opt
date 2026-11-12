"""Fingerprint RF2 model inputs, recycle state and selected output per candidate."""

import argparse
import hashlib
import json
import math
from functools import wraps
from pathlib import Path
import random
import runpy
import sys


RECYCLE_FIELDS = ("msa_prev", "pair_prev", "state_prev", "xyz_prev", "mask_recycle")


def scalar(value):
    value = float(value)
    return value if math.isfinite(value) else repr(value)


def install(torch, numpy, runner):
    def fingerprint(tensor):
        if tensor is None:
            return None
        binary = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        return {"sha256": hashlib.sha256(binary).hexdigest(),
                "shape": list(tensor.shape), "dtype": str(tensor.dtype)}

    def rng_state():
        numpy_state = numpy.random.get_state()
        numpy_bytes = numpy_state[1].tobytes() + repr((numpy_state[0], *numpy_state[2:])).encode()
        return {
            "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
            "numpy": hashlib.sha256(numpy_bytes).hexdigest(),
            "torch_cpu": fingerprint(torch.get_rng_state()),
            "torch_cuda": ([fingerprint(state) for state in torch.cuda.get_rng_state_all()]
                           if torch.cuda.is_initialized() else []),
        }

    original_predict = runner.AbPredictor.__call__
    original_convert = runner.AbPredictor._output_dictionary
    original_write = runner.write_output
    candidates = []
    current = None

    @wraps(original_predict)
    def predict(predictor, pose, tag):
        nonlocal current
        current = {"tag": tag, "cycles": []}

        def first_input(module, positional, keyword):
            if "initial_inputs" not in current:
                current["initial_inputs"] = {
                    name: fingerprint(value) for name, value in keyword.items()
                    if value is None or isinstance(value, torch.Tensor)
                }
                current["initial_rng"] = rng_state()

        hook = predictor.model.register_forward_pre_hook(first_input, with_kwargs=True)
        try:
            result = original_predict(predictor, pose, tag)
            if "selected" not in current or len(current["cycles"]) != predictor.conf.inference.num_recycles + 1:
                raise RuntimeError(f"Incomplete RF2 trace for {tag}")
            candidates.append(current)
            return result
        finally:
            hook.remove()
            current = None

    @wraps(original_convert)
    def convert(predictor, raw, inputs):
        raw_backbone = fingerprint(raw[5][-1])
        raw_torsions = fingerprint(raw[6][-1])
        output = original_convert(predictor, raw, inputs)
        bins = predictor.lddt_bins
        logits = output["pred_lddt"].to(bins.device)
        plddt = (torch.nn.Softmax(dim=1)(logits) * bins[None, :, None]).sum(dim=1)
        current["cycles"].append({
            "cycle": len(current["cycles"]) + 1,
            "backbone": raw_backbone,
            "torsions": raw_torsions,
            "recycle": {key: fingerprint(output[key]) for key in RECYCLE_FIELDS},
            "pose_xyz": fingerprint(output["xyz_prev"][0]),
            "pae_logits": fingerprint(output["logits_pae"]),
            "plddt_logits": fingerprint(output["pred_lddt"]),
            "plddt": fingerprint(plddt),
            "mean_plddt": scalar(plddt.mean()),
            "rng": rng_state(),
        })
        return output

    @wraps(original_write)
    def write(outputs, tag, conf):
        best = outputs.get("best")
        expected = None
        best_score = -1.0
        for cycle in current["cycles"]:
            mean = float(cycle["mean_plddt"])
            if mean > best_score:
                best_score = mean
                expected = cycle

        current["output_keys"] = list(outputs)
        current["selected"] = None
        if best is not None:
            pose = best["pose"]
            metrics = {key: fingerprint(value) if isinstance(value, torch.Tensor) else scalar(value)
                       for key, value in best["metrics"].items()}
            xyz = fingerprint(pose.xyz)
            if expected is None or xyz != expected["pose_xyz"] or metrics["pred_lddt"] != expected["plddt"]:
                raise RuntimeError(f"RF2 output for {tag} differs from the earliest maximum-pLDDT cycle")
            current["selected"] = {"cycle": expected["cycle"], "xyz": xyz, "metrics": metrics}
        elif expected is not None:
            raise RuntimeError(f"RF2 produced no output for {tag} despite a winning cycle")
        current["final_rng"] = rng_state()
        return original_write(outputs, tag, conf)

    runner.AbPredictor.__call__ = predict
    runner.AbPredictor._output_dictionary = convert
    runner.write_output = write
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True, help="model_probe.py or the original RF2 entry")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    trace = args.trace.resolve()
    probe = args.probe.resolve(strict=True)
    if trace.exists():
        parser.error("choose a fresh trace path")
    probe_args = args.args[1:] if args.args[:1] == ["--"] else args.args
    if not probe_args:
        parser.error("pass RF2 launch arguments after --")

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(probe.parent))
    import numpy
    import torch
    from rfantibody.rf2.modules import model_runner

    candidates = install(torch, numpy, model_runner)
    sys.argv = [str(probe), *probe_args]
    runpy.run_path(str(probe), run_name="__main__")
    if not candidates:
        raise RuntimeError("RF2 did not process any candidates")
    with trace.open("x") as output:
        json.dump({"schema_version": 1, "candidates": candidates}, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
