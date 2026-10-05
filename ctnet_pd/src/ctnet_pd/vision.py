"""ViT and Swin baselines on log-Mel segments (Experiment II; continuity with the preliminary study).

The preliminary study compared CTNet with a Vision Transformer and a Swin
Transformer. Here both are fine-tuned from ImageNet checkpoints (Hugging Face,
PyTorch) on exactly the same band-standardized log-Mel segments, subject-
disjoint folds, subject-balanced loss and inner-validation early stopping as
CTNet. Each 128 x 229 segment is replicated to three channels and resized
(bilinear) to the backbone's pretraining resolution.

PyTorch cannot share a process with TensorFlow in this environment, so
training and prediction run in a separate interpreter (``isolation``); this
module imports neither TensorFlow nor, at import time, PyTorch.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .isolation import run_isolated

TORCH_ARCHITECTURES = ("vit", "swin")


@dataclass
class TorchModelHandle:
    """A fine-tuned PyTorch model living on disk (weights) and run in a separate process."""

    weights_path: str
    model_name: str
    image_size: int

    def save(self, path) -> None:
        shutil.copyfile(self.weights_path, Path(path).with_suffix(".pt"))

    def cleanup(self) -> None:
        """Delete the temporary working directory that holds the weights."""
        shutil.rmtree(Path(self.weights_path).parent, ignore_errors=True)


def _vcfg(cfg: Config) -> dict:
    v = cfg.vision_transformers
    t = cfg.training
    return {
        "image_size": int(v.image_size), "weight_decay": float(v.weight_decay),
        "mixed_precision": bool(v.mixed_precision), "batch_size": int(v.batch_size or t.batch_size),
        "max_epochs": int(v.max_epochs or t.max_epochs), "patience": int(t.early_stopping_patience),
        "reduce_lr_factor": float(t.reduce_lr_factor), "reduce_lr_patience": int(t.reduce_lr_patience),
        "min_lr": float(t.min_lr),
    }


def _save_arrays(directory: Path, **arrays) -> None:
    for name, arr in arrays.items():
        if arr is not None:
            np.save(directory / f"{name}.npy", np.asarray(arr, dtype=np.float16 if name.startswith("X") else np.float32))


def train(cfg: Config, X_tr, y_tr, w_tr, X_val=None, y_val=None, w_val=None, seed: int = 0,
          work_dir: str | Path | None = None, learning_rates=None, epochs: int | None = None):
    """Fine-tune ``cfg.model.architecture`` (vit | swin).

    With validation data, each learning rate of the search space is trained with
    early stopping on the inner-validation (subject-balanced) loss and the best
    one is kept. Without validation data, a single learning rate is trained for
    ``epochs`` epochs (external-validation retraining).

    Returns ``(handle, hp, best_epoch)``.
    """
    arch = cfg.model.architecture
    if arch not in TORCH_ARCHITECTURES:
        raise ValueError(f"{arch} is not a PyTorch architecture")
    v = cfg.vision_transformers
    name = v.checkpoints[arch]
    if learning_rates is None:
        learning_rates = list(v.learning_rates) if cfg.tuning.enabled else [float(v.learning_rate)]
    work = Path(tempfile.mkdtemp(dir=work_dir))
    _save_arrays(work, X_tr=X_tr, y_tr=y_tr, w_tr=w_tr, X_val=X_val, y_val=y_val, w_val=w_val)
    weights = work / "best.pt"
    res = run_isolated(_train_child, str(work), name, _vcfg(cfg), [float(lr) for lr in learning_rates],
                       int(seed), epochs, str(weights), stream_output=True)
    for f in work.glob("*.npy"):
        f.unlink()
    handle = TorchModelHandle(str(weights), name, int(v.image_size))
    hp = {"learning_rate": res["learning_rate"], "val_loss_by_lr": res["val_loss_by_lr"]}
    return handle, hp, int(res["best_epoch"])


def predict_proba(handle: TorchModelHandle, X, batch_size: int = 64) -> np.ndarray:
    """P(PD) for each segment."""
    with tempfile.TemporaryDirectory(dir=Path(handle.weights_path).parent) as tmp:
        _save_arrays(Path(tmp), X_te=X)
        return run_isolated(_predict_child, str(Path(tmp) / "X_te.npy"), handle.model_name,
                            handle.weights_path, handle.image_size, batch_size)


# ------------------------------------------------------------------ child-process code

def _device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(name: str):
    from transformers import AutoModelForImageClassification

    return AutoModelForImageClassification.from_pretrained(name, num_labels=2, ignore_mismatched_sizes=True)


def _to_pixels(xb, size: int):
    """(B, H, W) log-Mel -> (B, 3, size, size) pixel values."""
    import torch.nn.functional as F

    x = xb.unsqueeze(1)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x.repeat(1, 3, 1, 1)


def _weighted_loss(model, X, y, w, idx, size, device, amp):
    import torch
    import torch.nn.functional as F

    xb = torch.as_tensor(np.asarray(X[idx], dtype=np.float32), device=device)
    yb = torch.as_tensor(np.asarray(y[idx], dtype=np.int64), device=device)
    wb = torch.as_tensor(np.asarray(w[idx], dtype=np.float32), device=device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        logits = model(pixel_values=_to_pixels(xb, size)).logits
    loss = F.cross_entropy(logits.float(), yb, reduction="none")
    return (loss * wb).sum(), wb.sum()


def _train_child(work: str, name: str, v: dict, learning_rates, seed: int, epochs, weights_out: str) -> dict:
    import copy

    import torch

    work = Path(work)
    load = lambda k: np.load(work / f"{k}.npy", mmap_mode="r") if (work / f"{k}.npy").exists() else None
    X_tr, y_tr, w_tr = load("X_tr"), load("y_tr"), load("w_tr")
    X_val, y_val, w_val = load("X_val"), load("y_val"), load("w_val")
    has_val = X_val is not None
    device = _device()
    amp = v["mixed_precision"] and device == "cuda"
    bs, size = v["batch_size"], v["image_size"]
    n_epochs = epochs or v["max_epochs"]

    best, val_loss_by_lr = None, {}
    for lr in learning_rates:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = _load_model(name).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=v["weight_decay"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=v["reduce_lr_factor"],
                                                           patience=v["reduce_lr_patience"], min_lr=v["min_lr"])
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        best_loss, best_epoch, best_state, wait = np.inf, n_epochs, None, 0
        for epoch in range(1, n_epochs + 1):
            model.train()
            perm = rng.permutation(len(y_tr))
            for start in range(0, len(perm), bs):
                idx = np.sort(perm[start:start + bs])
                total, wsum = _weighted_loss(model, X_tr, y_tr, w_tr, idx, size, device, amp)
                opt.zero_grad(set_to_none=True)
                scaler.scale(total / wsum).backward()
                scaler.step(opt)
                scaler.update()
            if not has_val:
                continue
            model.eval()
            with torch.no_grad():
                parts = [_weighted_loss(model, X_val, y_val, w_val, np.arange(s, min(s + bs, len(y_val))),
                                        size, device, amp) for s in range(0, len(y_val), bs)]
            vl = float(sum(p[0] for p in parts) / sum(p[1] for p in parts))
            sched.step(vl)
            print(f"[{name.split('/')[-1]}] lr={lr:g} epoch {epoch}: val_loss={vl:.4f}", flush=True)
            if vl < best_loss - 1e-6:
                best_loss, best_epoch, wait = vl, epoch, 0
                best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            else:
                wait += 1
                if wait >= v["patience"]:
                    break
        if not has_val:
            best_loss, best_epoch = float("nan"), n_epochs
            best_state = copy.deepcopy({k: t.detach().cpu() for k, t in model.state_dict().items()})
        val_loss_by_lr[lr] = best_loss
        if best is None or (has_val and best_loss < best[0]):
            best = (best_loss, lr, best_epoch, best_state)
        del model, opt
        if device == "cuda":
            torch.cuda.empty_cache()
    torch.save(best[3], weights_out)
    return {"learning_rate": best[1], "best_epoch": best[2], "val_loss": best[0], "val_loss_by_lr": val_loss_by_lr}


def _predict_child(x_path: str, name: str, weights: str, size: int, batch_size: int) -> np.ndarray:
    import torch

    device = _device()
    model = _load_model(name)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.to(device).eval()
    X = np.load(x_path, mmap_mode="r")
    out = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.as_tensor(np.asarray(X[start:start + batch_size], dtype=np.float32), device=device)
            logits = model(pixel_values=_to_pixels(xb, size)).logits.float()
            out.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out)
