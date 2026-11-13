"""Build the RF2 SM89 native kernels using the pinned cuBLASLt runtime."""

import argparse
import ctypes
from importlib.metadata import PackageNotFoundError, distribution
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from long_exact import CUBLASLT_VERSION, CUDA_SOURCES, DEFAULT_NATIVE_DIR


SOURCE = Path(__file__).resolve().parent / "long_exact/native"


def cublas_runtime():
    try:
        package = distribution("nvidia-cublas-cu11")
    except PackageNotFoundError as error:
        raise ValueError("Supply --cublas-lib or install the locked nvidia-cublas-cu11 wheel") from error
    if package.version != "11.11.3.6":
        raise ValueError(f"Expected nvidia-cublas-cu11 11.11.3.6, found {package.version}")
    return package.locate_file("nvidia/cublas/lib/libcublasLt.so.11")


def build(args):
    cuda = args.cuda_home
    if cuda is None:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            raise ValueError("Supply --cuda-home or add nvcc to PATH")
        cuda = Path(nvcc).resolve().parents[1]
    cuda = cuda.expanduser().resolve()
    include = args.cuda_include.expanduser().resolve() if args.cuda_include else cuda / "include"
    if not (cuda / "bin/nvcc").is_file() or not (include / "cublasLt.h").is_file():
        raise ValueError("The CUDA toolkit must provide nvcc and cublasLt.h")
    runtime = (args.cublaslt_lib or cublas_runtime()).expanduser().absolute()
    if runtime.name != "libcublasLt.so.11" or not runtime.is_file():
        raise ValueError("Choose the CUDA 11 libcublasLt.so.11 file")
    version = ctypes.CDLL(str(runtime)).cublasLtGetVersion
    version.argtypes = []
    version.restype = ctypes.c_size_t
    actual = version()
    if actual != CUBLASLT_VERSION:
        raise ValueError(f"Expected cuBLASLt {CUBLASLT_VERSION}, loaded {actual}")
    nvcc = cuda / "bin/nvcc"
    reported = subprocess.check_output([nvcc, "--version"], text=True)
    match = re.search(r"release (\d+)\.(\d+)", reported)
    if match is None or tuple(map(int, match.groups())) < (11, 8):
        raise ValueError("nvcc 11.8 or newer is required to build SM89 kernels")

    args.native_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".build-", dir=args.native_dir) as temporary:
        stage = Path(temporary)
        for name in CUDA_SOURCES:
            subprocess.run([nvcc, "-O3", "-arch=sm_89", "--shared", "-Xcompiler=-fPIC",
                            "-cudart=static", "--fmad=false", "-o", stage / f"{name}_kernel.so",
                            SOURCE / f"{name}.cu"], check=True)
        cublas = stage / "linear_cublaslt.so"
        subprocess.run([args.cxx, "-O3", "-std=c++17", "-shared", "-fPIC", "-I", include,
                        SOURCE / "axial_cublaslt.cpp", runtime, "-o", cublas], check=True)
        for artifact in stage.iterdir():
            artifact.replace(args.native_dir / artifact.name)
    print(f"Built RF2 exact kernels in {args.native_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", "--out", type=Path, default=DEFAULT_NATIVE_DIR)
    parser.add_argument("--cuda-home", type=Path, default=os.environ.get("CUDA_HOME"))
    parser.add_argument("--cuda-include", type=Path, help="separate directory containing cublasLt.h")
    parser.add_argument("--cublas-lib", "--cublaslt-lib", dest="cublaslt_lib", type=Path,
                        help="path to the pinned libcublasLt.so.11")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    args = parser.parse_args()
    try:
        build(args)
    except (AttributeError, OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"RF2 native build failed: {error}\n")


if __name__ == "__main__":
    main()
