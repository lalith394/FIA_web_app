"""
Lightweight, modular inference helpers for segmentation / autoencoder models.

This file intentionally focuses only on reusable inference functions used by the
backend API: finding model metadata, loading the model, reading images from disk,
processing them into tensors, running inference, and saving outputs (masks or
reconstructions). The old monolithic evaluation helpers and visualization code
were removed to keep this module small and focused.

Public helpers:
- find_model_metadata(model_name) -> (model_type, model_dir, metadata)
- load_model(model_dir, model_name, model_type, num_channels) -> torch.nn.Module
- read_image(path, size=(H,W)) -> torch.Tensor (C,H,W), float in [0,1]
- save_image(np_array, path, as_rgb=False) -> None
- infer_images(model_name, image_paths, threshold=0.5, out_dir=None, save_features=False, batch_size=1, rgb=False)

The infer_images function returns a list of file paths written on disk.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision import transforms as T

from model import UNet, AutoEncoder, AutoEncoder_RFMiD


def find_model_metadata(models_root: str, model_name: str) -> Tuple[str, str, dict]:
    """Search models/<type>/<model_name>/metadata.json then models/<model_name>/metadata.json.

    Returns: (model_type, model_dir, metadata)
    Raises FileNotFoundError if metadata not found.
    """
    if not os.path.isdir(models_root):
        raise FileNotFoundError(f"models root not found: {models_root}")

    # search under models/<type>/<model_name>
    for t in os.listdir(models_root):
        cand = os.path.join(models_root, t, model_name)
        meta_path = os.path.join(cand, 'metadata.json')
        if os.path.isdir(cand) and os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf8') as f:
                meta = json.load(f)
            model_type = meta.get('type') or t
            return model_type, cand, meta

    # fallback models/<model_name>/metadata.json
    cand2 = os.path.join(models_root, model_name)
    meta_path2 = os.path.join(cand2, 'metadata.json')
    if os.path.exists(meta_path2):
        with open(meta_path2, 'r', encoding='utf8') as f:
            meta = json.load(f)
        model_type = meta.get('type') or 'segmentation'
        return model_type, cand2, meta

    # final try segmentation/<model_name>
    seg_cand = os.path.join(models_root, 'segmentation', model_name)
    if os.path.isdir(seg_cand):
        meta_path = os.path.join(seg_cand, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf8') as f:
                meta = json.load(f)
        else:
            meta = {}
        return 'segmentation', seg_cand, meta

    raise FileNotFoundError(f"metadata not found for model {model_name}")


def load_model(model_dir: str, model_name: str, model_type: str, num_channels: Optional[List[int]] = None, device: Optional[torch.device] = None) -> torch.nn.Module:
    """Load a model given a directory and metadata info. Picks type-specific class.

    model_dir should be the folder containing <model_name>.pth. If multiple .pth present,
    the first one is used.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # resolve weight path
    pth = os.path.join(model_dir, f"{model_name}.pth")
    if not os.path.exists(pth):
        candidates = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        if not candidates:
            raise FileNotFoundError(f"no .pth weights found in {model_dir}")
        pth = os.path.join(model_dir, candidates[0])

    # pick model type
    if model_type == 'segmentation':
        arch = UNet(num_channels=(num_channels or [64, 128, 256, 512, 1024, 512, 256, 128, 64])).to(device)
    elif model_type == 'autoencoder':
        arch = AutoEncoder_RFMiD(num_channels=(num_channels or [64, 128, 256, 512, 1024, 512, 256, 128, 64])).to(device)
    else:
        # default to UNet for unknown types
        arch = UNet(num_channels=(num_channels or [64, 128, 256, 512, 1024, 512, 256, 128, 64])).to(device)

    checkpoint = torch.load(pth, map_location=device)
    state = checkpoint.get('model_state_dict', checkpoint)
    arch.load_state_dict(state)
    arch.eval()
    return arch


def read_image(path: str, size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """Read an image from disk and return a float32 tensor in range [0,1] shaped (C,H,W).

    If size is provided it should be (H, W).
    """
    img = Image.open(path).convert('RGB')
    if size is not None:
        img = img.resize((int(size[1]), int(size[0])), Image.BILINEAR) # type: ignore
    to_tensor = T.ToTensor()
    return to_tensor(img).float()


def save_image(arr: np.ndarray, path: str, binary: bool = False, threshold: float = 128) -> None:
    """
    Save a numpy array as a single-channel (grayscale) image using matplotlib.

    Behavior:
    - Float arrays are assumed to be in [0,1] and are clipped to that range.
    - Integer arrays (e.g., uint8 in 0..255) are converted to float in [0,1] by dividing by 255.
    - Multi-channel arrays are converted to grayscale via luminance.
    - `binary` parameter is honored by converting values to 0 or 1 (NOT multiplied by 255).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    img = np.asarray(arr)

    # If CHW (C,H,W) convert to HWC
    if img.ndim == 3 and img.shape[0] <= 4 and img.shape[0] != img.shape[2]:
        img = np.transpose(img, (1, 2, 0))

    # If multi-channel, convert to grayscale luminance
    if img.ndim == 3:
        if img.shape[2] == 3:
            img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        else:
            img = img.mean(axis=2)

    # Convert to float in [0,1]
    if img.dtype in [np.uint8, np.int32, np.int64]:
        img = (img.astype(np.float32) / 255.0)
    else:
        img = img.astype(np.float32)
        if img.max() > 1.0:
            # If values appear to be in 0..255, scale to 0..1
            img = img / 255.0

    # Apply binary threshold if requested (produce 0 or 1)
    if binary:
        img = (img > (threshold / 255.0)).astype(np.float32)

    # Use matplotlib to write grayscale image. vmin/vmax ensures consistent mapping.
    plt.imsave(path, img, cmap='gray', vmin=0.0, vmax=1.0)


def process_batch(model: torch.nn.Module, batch: torch.Tensor, model_type: str, threshold: float = 0.5) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run a model forward and return processed outputs and optional features.

    For segmentation models this should return probabilities (sigmoid applied) or logits depending on model.
    """
    with torch.no_grad():
        outputs = None
        features = None
        try:
            out = model(batch, return_features=True)
            # some models return (outputs, features)
            if isinstance(out, tuple) and len(out) == 2:
                outputs, features = out
            else:
                outputs = out
        except TypeError:
            # model doesn't accept return_features
            outputs = model(batch)

        if model_type == 'segmentation':
            outputs = torch.sigmoid(outputs) # type: ignore
        return outputs, features # type: ignore


def infer_images(model_name: str, image_paths: List[str], threshold: float = 0.5, out_dir: Optional[str] = None, save_features: bool = False, batch_size: int = 1, rgb: bool = False) -> List[str]:
    """Metadata-driven inference entrypoint.

    - locates model metadata, loads the model, reads images, runs inference in batches
    - writes masks/reconstructions to out_dir (absolute or relative to ./output)
    - returns list of written file paths
    """
    try:
        # try to use user's set_seed helper if present in utils
        from utils import set_seed as _set_seed
        _set_seed(42)
    except Exception:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models_root = os.path.join(os.getcwd(), 'models')
    model_type, model_dir, meta = find_model_metadata(models_root, model_name)
    print(model_type, model_dir)

    params = meta.get('parameters', {}) if isinstance(meta, dict) else {}
    num_channels = None
    if 'num_channels' in params:
        v = params['num_channels']
        if isinstance(v, dict):
            num_channels = v.get('default')
        else:
            num_channels = v

    res = None
    if 'resolution' in params:
        v = params['resolution']
        if isinstance(v, dict):
            res = v.get('default')
        else:
            res = v
    # defaults
    if model_type == 'segmentation' and (not res or len(res) != 2):
        res = [384, 576]

    # resolve out_dir
    if out_dir is None:
        out_dir = model_name
    if isinstance(out_dir, str) and os.path.isabs(out_dir):
        results_dir = out_dir
    else:
        results_dir = os.path.join(os.getcwd(), 'output', str(out_dir))
    os.makedirs(results_dir, exist_ok=True)

    model = load_model(model_dir, model_name, model_type, num_channels=num_channels, device=device)

    # build transforms
    size = tuple(map(int, res)) if res else None
    def read_and_stack(paths: List[str]):
        tensors = []
        valid_paths = []
        for p in paths:
            try:
                t = read_image(p, size=size) # type: ignore
                tensors.append(t)
                valid_paths.append(p)
            except Exception:
                # skip unreadable file
                continue
        if not tensors:
            return torch.empty(0), []
        batch = torch.stack(tensors, dim=0).to(device)
        return batch, valid_paths

    outputs_written: List[str] = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch, valid_paths = read_and_stack(batch_paths)
        if batch.numel() == 0:
            continue

        outputs, features = process_batch(model, batch, model_type, threshold=threshold)

        # each output item -> save
        for j in range(outputs.shape[0]):
            out_single = outputs[j].cpu()
            original_path = valid_paths[j]

            # build save path preserving uploads structure if possible
            base = os.path.splitext(os.path.basename(original_path))[0]
            uploads_root = os.path.join(os.getcwd(), 'uploads')
            try:
                common = os.path.commonpath([os.path.abspath(original_path), uploads_root])
            except Exception:
                common = ''
            if common == os.path.abspath(uploads_root):
                rel = os.path.relpath(original_path, start=uploads_root)
                rel_dir = os.path.dirname(rel)
                save_dir = os.path.join(results_dir, rel_dir)
            else:
                save_dir = results_dir
            os.makedirs(save_dir, exist_ok=True)

            out_file = os.path.join(save_dir, f"{base}_mask.png" if model_type == 'segmentation' else f"{base}.png")

            if model_type == 'segmentation':
                # Save raw model output (probabilities) as a grayscale image (one file per input)
                mask = out_single.squeeze().cpu().numpy()
                save_image(mask, out_file)
                outputs_written.append(os.path.abspath(out_file))
            else:
                # reconstructions: assume C,H,W
                arr = out_single.cpu().numpy()
                # Autoencoder reconstructions: normalize to single-channel grayscale
                if arr.ndim == 3:
                    # assume (C,H,W) or (H,W,C)
                    if arr.shape[0] <= 4 and arr.shape[0] != arr.shape[1]:
                        img_np = arr.squeeze(0) if arr.shape[0] == 1 else np.transpose(arr, (1, 2, 0))
                    else:
                        img_np = np.transpose(arr, (1, 2, 0))
                else:
                    img_np = arr
                # If multi-channel, convert to grayscale inside save_image
                save_image(img_np, out_file)
                outputs_written.append(os.path.abspath(out_file))

            # features
            if save_features and features is not None:
                feat = features[j].cpu().numpy()
                feat_dir = os.path.join(save_dir, f"{base}_d4_layer")
                os.makedirs(feat_dir, exist_ok=True)
                for ch in range(feat.shape[0]):
                    fmap = feat[ch]
                    fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
                    fmap = (fmap * 255).astype('uint8')
                    save_image(fmap, os.path.join(feat_dir, f"channel_{ch}.png"))

    return outputs_written
