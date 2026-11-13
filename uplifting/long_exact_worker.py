"""Run one seeded RF2 candidate with the packaged FP32 kernels."""

import argparse
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import sys

from long_exact import install


CHECKPOINT_SHA256 = "cb38916458f5a9ecad0cf33182d816a836204b58896c59bd5a4ef475f63950e7"
SCREENED_LENGTHS = (397, 1123, 1124)


def checkpoint_matches(path):
    if path is None:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest() == CHECKPOINT_SHA256


def input_length(conf):
    """Count the single candidate's Cα records as the RF2 HLT reader does."""
    if conf.input.pdb is not None:
        source = Path(conf.input.pdb)
        is_quiver = False
    elif conf.input.quiver is not None:
        source = Path(conf.input.quiver)
        is_quiver = True
    else:
        return None

    tags = length = 0
    previous = None
    with source.open() as stream:
        for line in stream:
            if is_quiver and line.startswith("QV_TAG"):
                tags += 1
                if tags > 1:
                    return None
            if (tags or not is_quiver) and line[:4] == "ATOM" and line[12:16].strip() == "CA":
                residue = (line[21:22].strip(), line[22:26].strip())
                if residue != previous:
                    length += 1
                previous = residue
    return length


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path)
    parser.add_argument("--weights", type=Path, default=os.environ.get("UPLIFTING_RF2_WEIGHTS"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--recycles", type=int)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or (args.recycles is not None and args.recycles < 0):
        parser.error("supply the RF2 entry command and a nonnegative recycle count")
    if args.report is not None and os.path.lexists(args.report):
        parser.error("choose a fresh report path")
    entry = Path(command[0]).resolve(strict=True)
    import torch
    from rfantibody.rf2.modules.model_runner import AbPredictor
    from rfantibody.rf2.network.RoseTTAFoldModel import RoseTTAFoldModule

    metadata = {"plugin_calls": {"enabled": False, "fallback_reason": "no RF2 model constructed"},
                "forward_count": 0, "complex_length": None}
    initialize = AbPredictor.__init__
    original = RoseTTAFoldModule.forward
    cache = cache_rng = None
    configured = False

    @wraps(initialize)
    def configure(module, conf, *positional, **keyword):
        nonlocal cache, cache_rng, configured
        if configured:
            raise RuntimeError("Use one RF2 predictor per long-exact worker")
        configured = True
        length = input_length(conf)
        metadata["complex_length"] = length
        if length not in SCREENED_LENGTHS:
            counts = {"enabled": False, "fallback_reason": "complex length outside the screened shapes"}
        else:
            path = Path(conf.model.model_weights)
            if (args.weights is not None and args.weights.resolve() != path.resolve()) or not checkpoint_matches(path):
                counts = {"enabled": False, "fallback_reason": "requires the verified public RF2_ab.pt checkpoint"}
            else:
                counts = install(torch, "audit" if args.audit else "fused", args.native_dir)
        metadata["plugin_calls"] = counts
        if counts["enabled"]:
            from triton.runtime import cache as triton_cache

            cache = triton_cache
            cache_rng = cache.random
            cache.random = random.SystemRandom()
        else:
            print(f"RF2 long-exact fallback: {counts['fallback_reason']}", file=sys.stderr, flush=True)
        return initialize(module, conf, *positional, **keyword)

    @wraps(original)
    def forward(module, *positional, **keyword):
        result = original(module, *positional, **keyword)
        metadata["forward_count"] += 1
        return result

    AbPredictor.__init__ = configure
    RoseTTAFoldModule.forward = forward
    try:
        sys.argv = [str(entry), *command[1:]]
        runpy.run_path(str(entry), run_name="__main__")
    finally:
        AbPredictor.__init__ = initialize
        RoseTTAFoldModule.forward = original
        if cache is not None:
            cache.random = cache_rng
    if args.recycles is not None and metadata["forward_count"] != args.recycles + 1:
        raise ValueError(f"RF2 completed {metadata['forward_count']} forwards, expected {args.recycles + 1}")
    if args.report is not None:
        with args.report.open("x") as report:
            json.dump(metadata, report, sort_keys=True)
            report.write("\n")
    print(f"RF2 long-exact {json.dumps(metadata, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
