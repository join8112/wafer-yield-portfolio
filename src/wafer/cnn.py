"""Small CNN for wafer-map defect patterns, plus training loop and Grad-CAM.

Designed to train in a few minutes on a free Colab T4 GPU.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from wafer.data import FAILURE_TYPES

N_CLASSES = len(FAILURE_TYPES)


def to_channels(X: np.ndarray) -> np.ndarray:
    """(n, H, W) maps with 0/1/2 -> (n, 2, H, W) float: [is_die, is_fail].

    Two binary channels avoid implying that "fail" (2) is numerically "more" than "pass" (1).
    """
    X = np.asarray(X)
    return np.stack([(X > 0), (X == 2)], axis=1).astype(np.float32)


class WaferDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = torch.from_numpy(to_channels(X))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment:
            # Defect patterns keep their meaning under rotation and mirroring,
            # so these augmentations are physically valid for wafer maps.
            x = torch.rot90(x, int(torch.randint(4, (1,))), dims=(1, 2))
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=(2,))
        return x, self.y[i]


def _block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class WaferCNN(nn.Module):
    """3 conv blocks -> global average pooling -> linear. ~0.3M parameters."""

    def __init__(self, n_classes: int = N_CLASSES, width: int = 32, dropout: float = 0.3):
        super().__init__()
        self.b1 = _block(2, width)
        self.b2 = _block(width, width * 2)
        self.b3 = _block(width * 2, width * 4)  # Grad-CAM target layer
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width * 4, n_classes))

    def features(self, x):
        x = self.pool(self.b1(x))
        x = self.pool(self.b2(x))
        return self.b3(x)

    def forward(self, x):
        f = self.features(x)
        return self.head(f.mean(dim=(2, 3)))


def class_weights(y: np.ndarray, power: float = 0.5) -> torch.Tensor:
    """Inverse-frequency weights, softened by `power` (0.5 = inverse square root).

    Full inverse frequency would weight Near-full ~1000x more than none and
    make training unstable; the square root keeps rare classes visible.
    """
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    w = (counts.max() / np.maximum(counts, 1)) ** power
    return torch.tensor(w / w.mean(), dtype=torch.float32)


@torch.no_grad()
def predict(model, X: np.ndarray, batch_size: int = 1024, device=None) -> np.ndarray:
    """Returns class probabilities, shape (n, N_CLASSES)."""
    device = device or next(model.parameters()).device
    model.eval()
    loader = DataLoader(WaferDataset(X, np.zeros(len(X))), batch_size=batch_size)
    out = [F.softmax(model(xb.to(device)), 1).cpu().numpy() for xb, _ in loader]
    return np.concatenate(out)


def train(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 2e-3,
    weight_power: float = 0.5,
    device: str | None = None,
    seed: int = 42,
    verbose: bool = True,
):
    """Train with class-weighted loss and keep the epoch with the best validation macro-F1."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = WaferCNN().to(device)
    loader = DataLoader(
        WaferDataset(X_train, y_train, augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if device == "cuda" else 0,
        pin_memory=device == "cuda",
    )
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights(np.asarray(y_train), weight_power).to(device)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(loader))

    history, best_f1, best_state = [], -1.0, None
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(yb)

        val_pred = predict(model, X_val, device=device).argmax(1)
        val_f1 = f1_score(y_val, val_pred, average="macro")
        history.append(
            {"epoch": epoch, "train_loss": total / len(loader.dataset), "val_macro_f1": val_f1}
        )
        if val_f1 > best_f1:
            best_f1, best_state = val_f1, copy.deepcopy(model.state_dict())
        if verbose:
            print(
                f"epoch {epoch:2d}  loss {history[-1]['train_loss']:.4f}  "
                f"val macro-F1 {val_f1:.4f}  ({time.time() - t0:.0f}s)"
            )

    model.load_state_dict(best_state)
    return model, history


def grad_cam(model, x: np.ndarray, target: int | None = None) -> tuple[np.ndarray, int]:
    """Grad-CAM heat-map for one wafer map (H, W) -> ((H, W) in [0, 1], predicted class).

    Shows which dies drove the prediction; for a correct 'Edge-Ring' call the
    heat should sit on the wafer edge, not on random interior dies.
    """
    device = next(model.parameters()).device
    model.eval()
    inp = torch.from_numpy(to_channels(x[None])).to(device)
    feats = model.features(inp)
    feats.retain_grad()
    logits = model.head(feats.mean(dim=(2, 3)))
    cls = int(logits.argmax(1)) if target is None else target
    model.zero_grad()
    logits[0, cls].backward()
    w = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * feats).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape, mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    return cam / (cam.max() + 1e-8), cls
