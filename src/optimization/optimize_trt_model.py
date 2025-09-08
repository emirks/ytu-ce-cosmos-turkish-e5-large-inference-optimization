import argparse
import glob
import os
import sys
import subprocess
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import onnxruntime as ort


def run_cmd(cmd: List[str]) -> None:
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if proc.returncode != 0:
        print(proc.stdout)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    else:
        print(proc.stdout)


def export_onnx(model_id: str, onnx_dir: str, opset: int) -> str:
    Path(onnx_dir).mkdir(parents=True, exist_ok=True)
    # Prefer optimum-cli if available; otherwise raise
    cmd = [
        sys.executable,
        "-m",
        "optimum.exporters.onnx",
        "--model",
        model_id,
        "--task",
        "feature-extraction",
        "--opset",
        str(opset),
        onnx_dir,
    ]
    print(f"[info] Exporting ONNX via: {' '.join(cmd)}")
    run_cmd(cmd)

    onnx_files = glob.glob(str(Path(onnx_dir) / "*.onnx"))
    if not onnx_files:
        raise FileNotFoundError("No ONNX file found after export")
    onnx_path = sorted(onnx_files)[0]
    print(f"[info] ONNX path: {onnx_path}")
    return onnx_path


def get_input_names(onnx_path: str) -> List[str]:
    cpu_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return [i.name for i in cpu_sess.get_inputs()]


def make_profile_strings(
    names: List[str],
    batch_min: int,
    seq_min: int,
    batch_opt: int,
    seq_opt: int,
    batch_max: int,
    seq_max: int,
) -> Tuple[str, str, str]:
    parts = []
    if any("input_ids" in n for n in names):
        parts.append(
            f"input_ids:{batch_min}x{seq_min},{batch_opt}x{seq_opt},{batch_max}x{seq_max}"
        )
    if any("attention_mask" in n for n in names):
        parts.append(
            f"attention_mask:{batch_min}x{seq_min},{batch_opt}x{seq_opt},{batch_max}x{seq_max}"
        )
    if any("token_type_ids" in n for n in names):
        parts.append(
            f"token_type_ids:{batch_min}x{seq_min},{batch_opt}x{seq_opt},{batch_max}x{seq_max}"
        )

    def reattach(parts_src: List[str], which: int) -> str:
        out = []
        for p in parts_src:
            tname = p.split(":")[0]
            dims = p.split(",")[which].split(":")[-1]
            out.append(f"{tname}:{dims}")
        return ",".join(out)

    mins = reattach(parts, 0)
    opts = reattach(parts, 1)
    maxs = reattach(parts, 2)
    return mins, opts, maxs


def build_trt_engine_cache(
    onnx_path: str,
    engine_cache_dir: str,
    timing_cache_path: str,
    trt_min: str,
    trt_opt: str,
    trt_max: str,
    fp16: bool = True,
    opt_level: int = 3,
) -> bool:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    available = ort.get_available_providers()
    trt_opts = {
        "trt_fp16_enable": bool(fp16),
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": engine_cache_dir,
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": timing_cache_path,
        "trt_builder_optimization_level": int(opt_level),
        "trt_profile_min_shapes": trt_min,
        "trt_profile_opt_shapes": trt_opt,
        "trt_profile_max_shapes": trt_max,
    }

    if "TensorrtExecutionProvider" in available:
        providers = [("TensorrtExecutionProvider", trt_opts)]
        print("[info] TensorRT EP available; attempting to build engine cache…")
    elif "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print(
            "[warn] TensorRT EP not available. Falling back to CUDAExecutionProvider for ONNX validation only."
        )
    else:
        providers = ["CPUExecutionProvider"]
        print(
            "[warn] Neither TensorRT nor CUDA EPs available. Falling back to CPU for ONNX validation only."
        )

    print("[info] Providers requested:", providers)
    print(
        "[info] Building session (first run may be slow if TensorRT is compiling or CUDA is initializing)…"
    )
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    active_providers = sess.get_providers()
    print("[info] Providers active:", active_providers)
    trt_active = "TensorrtExecutionProvider" in active_providers

    # Prepare a single run at opt shape to trigger build/validate
    ins = {i.name: i for i in sess.get_inputs()}

    def shape_for(name: str, shape_str: str) -> Tuple[int, int]:
        # e.g. "input_ids:128x256" -> (128, 256)
        part = [p for p in shape_str.split(",") if p.startswith(f"{name}:")]
        if not part:
            # fallback: first entry
            part = [shape_str.split(",")[0]]
        dims = part[0].split(":")[1]
        b, s = dims.split("x")
        return int(b), int(s)

    feed: Dict[str, np.ndarray] = {}
    for in_name in ins.keys():
        if "input_ids" in in_name:
            b, s = shape_for("input_ids", trt_opt)
        elif "attention_mask" in in_name:
            b, s = shape_for("attention_mask", trt_opt)
        elif "token_type_ids" in in_name:
            b, s = shape_for("token_type_ids", trt_opt)
        else:
            # unknown name: use the first spec
            spec = trt_opt.split(",")[0]
            _, dims = spec.split(":")
            b, s = map(int, dims.split("x"))
        dtype = np.int64
        feed[in_name] = np.zeros((b, s), dtype=dtype)

    _ = sess.run(None, feed)

    # Validate cache artifacts only if TRT was actually used
    if trt_active:
        cache_dir = Path(engine_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        artifacts = list(cache_dir.glob("*"))
        if not artifacts:
            print(f"[warn] No artifacts found in engine cache dir: {engine_cache_dir}")
        else:
            print(
                f"[info] Engine cache populated: {len(artifacts)} files in {engine_cache_dir}"
            )
    else:
        print(
            "[info] Skipped TensorRT engine cache validation because TRT EP was not active."
        )

    return trt_active


def main():
    MODEL_ID = "ytu-ce-cosmos/turkish-e5-large"
    ONNX_DIR = "onnx_e5"
    OPSET = 17

    BATCH_MIN, SEQ_MIN = 1, 64
    BATCH_OPT, SEQ_OPT = 128, 256
    BATCH_MAX, SEQ_MAX = 1024, 512

    ENGINE_CACHE_DIR = "trt_cache"
    TIMING_CACHE_PATH = "trt_cache/timing.cache"

    FP16 = True
    OPT_LEVEL = 3

    onnx_path = export_onnx(MODEL_ID, ONNX_DIR, OPSET)
    names = get_input_names(onnx_path)
    trt_min, trt_opt, trt_max = make_profile_strings(
        names,
        BATCH_MIN,
        SEQ_MIN,
        BATCH_OPT,
        SEQ_OPT,
        BATCH_MAX,
        SEQ_MAX,
    )

    trt_used = build_trt_engine_cache(
        onnx_path=onnx_path,
        engine_cache_dir=ENGINE_CACHE_DIR,
        timing_cache_path=TIMING_CACHE_PATH,
        trt_min=trt_min,
        trt_opt=trt_opt,
        trt_max=trt_max,
        fp16=FP16,
        opt_level=OPT_LEVEL,
    )

    print("[done] ONNX exported.")
    if trt_used:
        print("[done] TensorRT engine cache prepared.")
        print(
            f"[hint] Use ONNX_PATH={onnx_path} and TRT cache at {ENGINE_CACHE_DIR} in your embed script."
        )
    else:
        print(
            "[hint] TensorRT EP was not active. You can still use the ONNX model. To enable TRT later, install TRT libs and an ORT build with TRT EP."
        )


if __name__ == "__main__":
    main()
