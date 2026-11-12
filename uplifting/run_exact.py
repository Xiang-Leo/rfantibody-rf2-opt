"""Run the RF2 exact mode, with an independent seeded process per candidate."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def quiver_records(path):
    # Match the pinned reader's line and tag rules while retaining the original
    # line endings, scores, REMARKs, and any preamble before the first record.
    with path.open("r", newline="") as source:
        tag, lines = None, []
        for line in source:
            if line.startswith("QV_TAG"):
                if tag is not None:
                    yield tag, "".join(lines)
                    lines = []
                fields = line.split()
                if len(fields) != 2:
                    raise ValueError(f"Invalid Quiver tag line in {path}: {line.rstrip()!r}")
                tag = fields[1]
            lines.append(line)
        if tag is not None:
            yield tag, "".join(lines)


def candidate_inputs(quiver, pdb, single_quiver):
    if pdb is not None:
        # Upstream derives the tag from the input basename. Quiver tags need
        # one token, so stage a name without whitespace when necessary.
        tag = "_".join(pdb.stem.split()) or "candidate"
        if tag != pdb.stem:
            staged_pdb = single_quiver.parent / f"{tag}.pdb"
            staged_pdb.symlink_to(pdb)
            pdb = staged_pdb
        yield "pdb", pdb, tag
        return

    seen = set()
    for tag, record in quiver_records(quiver):
        if tag in seen:
            raise ValueError(f"Duplicate Quiver tag in {quiver}: {tag!r}")
        seen.add(tag)
        single_quiver.write_text(record, newline="")
        yield "quiver", single_quiver, tag
    if not seen:
        raise ValueError(f"No Quiver records in {quiver}")


def environment(gpu):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("NABLOOP_")}
    # Retain the pinned PyTorch precision defaults even in a shell previously
    # used for accelerated runs. cuDNN keeps its original defaults.
    env.pop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", None)
    env.pop("NVIDIA_TF32_OVERRIDE", None)
    pythonpath = [str(ROOT / "src"), str(ROOT / "include/SE3Transformer")]
    if env.get("PYTHONPATH"):
        pythonpath.extend(os.path.abspath(path) for path in env["PYTHONPATH"].split(os.pathsep))
    env.update({
        "DGLBACKEND": "pytorch",
        "PYTHONPATH": os.pathsep.join(pythonpath),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "NABLOOP_RF2_DEFER_OUTPUT": "1",
        "NABLOOP_RF2_SKIP_AUX": "1",
    })
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def predict(work, ordinal, input_kind, input_path, tag, weights, seed, env):
    result = work / "prediction.qv"
    command = [sys.executable]
    if env.get("NABLOOP_RF2_SE3_EXACT") == "1":
        command.append(str(ROOT / "uplifting/rf2_se3_exact.py"))
    command.extend([
        str(ROOT / "scripts/rf2_predict.py"),
        f"input.{input_kind}=${{oc.env:UPLIFTING_RF2_INPUT}}",
        "model.model_weights=${oc.env:UPLIFTING_RF2_WEIGHTS}",
        "output.quiver=prediction.qv", "output.output_intermediates=False",
        "inference.num_recycles=10", "inference.hotspot_show_proportion=0.0",
        "inference.cautious=True", f"+inference.seed={seed}",
        "hydra.job.chdir=False", f"hydra.run.dir=hydra-{ordinal:06d}",
    ])
    # Pass external paths as literal resolver values. This keeps their spaces,
    # commas, or quotes out of Hydra's override grammar.
    subprocess.run(command, cwd=work, check=True, env=env | {
        "UPLIFTING_RF2_INPUT": str(input_path),
        "UPLIFTING_RF2_WEIGHTS": str(weights),
    })
    if not result.is_file():
        raise ValueError(f"RF2 produced no Quiver output for {tag!r}")
    tags = [output_tag for output_tag, _ in quiver_records(result)]
    if tags != [f"{tag}_best"]:
        raise ValueError(f"Unexpected RF2 output tags for {tag!r}: {tags!r}")
    return result


def run(args):
    if not (ROOT / "scripts/rf2_predict.py").is_file():
        raise ValueError("Run this launcher from its RF2 checkout")
    for path in (args.quiver or args.pdb, args.weights):
        if not path.is_file():
            raise ValueError(f"File does not exist: {path}")
    if os.path.lexists(args.output):
        raise ValueError(f"Choose a fresh output path: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = environment(args.gpu)
    if args.se3_exact:
        env["NABLOOP_RF2_SE3_EXACT"] = "1"
    with tempfile.TemporaryDirectory(prefix=".uplifting-rf2-", dir=args.output.parent) as directory:
        work = Path(directory)
        combined = work / "combined.qv"
        inputs = candidate_inputs(args.quiver, args.pdb, work / "input.qv")
        with combined.open("wb") as destination:
            for ordinal, (input_kind, input_path, tag) in enumerate(inputs, 1):
                result = predict(work, ordinal, input_kind, input_path, tag, args.weights, args.seed, env)
                with result.open("rb") as record:
                    shutil.copyfileobj(record, destination)
                result.unlink()
        # The staging directory shares the destination filesystem. Publish in
        # one operation, with no replacement even if another run finished first.
        os.link(combined, args.output)
    print(f"Wrote {ordinal} RF2 prediction(s) to {args.output}")


def absolute_path(value):
    # Keep the supplied basename for the tag of a symlinked PDB.
    return Path(value).expanduser().absolute()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--quiver", type=absolute_path, help="input Quiver of candidates")
    inputs.add_argument("--pdb", type=absolute_path, help="one RFantibody HLT REMARK PDB")
    parser.add_argument("--weights", required=True, type=absolute_path, help="path to RF2_ab.pt")
    parser.add_argument("--output", required=True, type=absolute_path, help="new output Quiver path")
    parser.add_argument("--seed", required=True, type=int, help="seed used independently for each candidate")
    parser.add_argument("--gpu", help="physical CUDA device index or UUID (default uses CUDA visibility)")
    parser.add_argument("--se3-exact", action="store_true", help="use the experimental FP32 SE(3) radial contractions")
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in the NumPy seed range [0, 2**32)")
    try:
        run(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"RF2 failed: {error}\n")


if __name__ == "__main__":
    main()
