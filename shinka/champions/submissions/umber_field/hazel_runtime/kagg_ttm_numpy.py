"""Pure-numpy inference for the oracle's TinyTimeMixer checkpoint.

Why this exists: Kaggle's sandbox runs agents on the CPU with 1 s per step and a
one-off ~60 s budget for the first request, which is when the agent file is
exec'd. Importing torch + transformers there took the validation episode of the
2026-09-12 "Orchard Tide" submission past that budget (TIMEOUT at step 1 on both
seats). numpy imports in ~50 ms and the network is ~1 M parameters, so the
forward pass is reimplemented here from `tsfm_public` 0.3.9's
`modeling_tinytimemixer.py`, restricted to what the checkpoint actually
instantiates (see `check_oracle.py numpy` / `compare_numpy_backend` below):

    TinyTimeMixerForPrediction, loss=mse (no distribution head), scaling=std,
    context 512 / patch 64 / stride 64 -> 8 patches, d_model 192,
    encoder: Linear(64->192), no positional encoding, no register/FFT tokens,
             adaptive patching with 3 levels (factors 4, 2, 1), 2 mixer layers
             per level in common_channel mode (patch mixer + feature mixer),
    decoder: Linear(192->128), 2 layers in mix_channel mode (channel mixer,
             patch mixer, feature mixer), no raw residual,
    head:    flatten(8 x 128) -> Linear(1024 -> 96) per channel, channels
             prediction_channel_indices, then y * scale + loc of those channels.

Every block: LayerNorm (eps 1e-5) on the last axis, MLP fc1 -> exact GELU ->
fc2 (expansion 2), softmax gated attention on the mixed axis (gate AFTER the MLP
in the patch/feature mixers, BEFORE it in the channel mixer), residual add.
Dropout is inference-inactive. Against the torch model in float64 the port
agrees to ~1e-7 (the erf fit); in float32 the two differ by ~1e-5 on random
windows and up to ~1e-3 on windows with exactly constant channels, where the
difference is torch's own float32 reduction error (numpy-f32 sits closer to the
f64 truth). Both are far below anything the policy thresholds on.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_SAFETENSOR_DTYPES = {
    "F32": np.float32, "F16": np.float16, "F64": np.float64,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_,
}


def load_safetensors(path):
    """Read a .safetensors file without the `safetensors` package (float32 out)."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        data = fh.read()
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype = _SAFETENSOR_DTYPES.get(meta["dtype"])
        if dtype is None:
            raise ValueError(f"unsupported safetensors dtype {meta['dtype']} for {name}")
        start, end = meta["data_offsets"]
        out[name] = np.frombuffer(data[start:end], dtype=dtype).reshape(meta["shape"]).astype(np.float32)
    return out


# ----------------------------------------------------------------- primitives
# All elementwise work stays in float32 (the arrays are ~0.5 M elements per
# block and there are ~18 GELUs per forecast); only the scaler statistics are
# accumulated in float64 -- numpy's float32 reduction over the strided time axis
# is a naive sum whose ~3e-5 relative error, divided by the sqrt(1e-5) floor of
# a constant channel, would put a ~0.02 offset on every constant channel.
_SQRT_HALF = np.float32(np.sqrt(0.5))
_ERF_P = np.float32(0.3275911)
_ERF_A = [np.float32(v) for v in (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)]
def _erf_poly(z):
    """erf(z) for z >= 0, Abramowitz & Stegun 7.1.26 (|err| <= 1.5e-7), in place."""
    t = _ERF_P * z
    t += np.float32(1.0)
    np.reciprocal(t, out=t)
    poly = t * _ERF_A[4]
    for a in (_ERF_A[3], _ERF_A[2], _ERF_A[1], _ERF_A[0]):
        poly += a
        poly *= t
    np.multiply(z, z, out=z)
    np.negative(z, out=z)
    np.exp(z, out=z)
    z *= poly
    np.subtract(np.float32(1.0), z, out=z)
    return z


def _gelu(x):
    """torch.nn.functional.gelu (exact, erf-based): the polynomial erf lands on
    the same float32 values as scipy's erf (max 5e-7 off the float64 result on a
    typical block) at a quarter of its cost, and needs no scipy."""
    e = _erf_poly(np.abs(x) * _SQRT_HALF)
    np.copysign(e, x, out=e)
    e += np.float32(1.0)
    e *= x
    e *= np.float32(0.5)
    return e


def _layer_norm(x, w, b, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    xc = x - mean
    var = np.square(xc).mean(axis=-1, keepdims=True)
    var += np.float32(eps)
    np.sqrt(var, out=var)
    xc /= var
    xc *= w
    xc += b
    return xc


def _softmax(x):
    """softmax over the last axis; reductions run on a 2-D view (numpy's
    keepdims reductions over the short last axis of a 3-D array are slower)."""
    n = x.shape[-1]
    x2 = np.ascontiguousarray(x).reshape(-1, n)
    e = x2 - x2.max(axis=1)[:, None]
    np.exp(e, out=e)
    e /= e.sum(axis=1)[:, None]
    return e.reshape(x.shape)


def _linear(x, w, b):
    """x[..., in] @ w.T + b as ONE 2-D GEMM (numpy's stacked matmul would issue
    one tiny GEMM per leading index -- 150 channels x 57 layers per forecast)."""
    x2 = np.ascontiguousarray(x).reshape(-1, x.shape[-1])
    out = np.matmul(x2, w.T)
    out += b
    return out.reshape(x.shape[:-1] + (w.shape[0],))


# ----------------------------------------------------------------- the network
class TTMNumpy:
    """`predict(block)` for one context window; mirrors the torch model's
    `prediction_outputs[0]` for the checkpoint described in the module docstring."""

    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        cfg = json.loads((model_dir / "config.json").read_text())
        self.cfg = cfg
        self.context = int(cfg["context_length"])
        self.patch_len = int(cfg["patch_length"])
        self.patch_stride = int(cfg["patch_stride"])
        self.num_patches = int(cfg["num_patches"])
        self.d_model = int(cfg["d_model"])
        self.decoder_d_model = int(cfg["decoder_d_model"])
        self.n_channels = int(cfg["num_input_channels"])
        self.horizon = int(cfg["prediction_length"])
        self.pred_idx = sorted(int(i) for i in cfg["prediction_channel_indices"])
        self.num_layers = int(cfg["num_layers"])
        self.decoder_layers = int(cfg["decoder_num_layers"])
        self.levels = int(cfg["adaptive_patching_levels"])
        self.norm_eps = float(cfg.get("norm_eps", 1e-5))
        self._check_supported(cfg)
        self.w = load_safetensors(model_dir / "model.safetensors")
        # adaptive patching: mixers[i] is level (levels-1-i) -> factor 2**level
        self.factors = []
        for i in range(self.levels):
            level = self.levels - 1 - i
            factor = 2 ** level
            if self.d_model // factor <= 4:  # the reference disables the level then
                factor = 1
            if self.d_model % factor:
                raise ValueError("d_model must be divisible by 2**level")
            self.factors.append(factor)
        seq = self.context
        self.n_patches_seq = (seq - self.patch_len) // self.patch_stride + 1
        self.seq_start = seq - (self.patch_len + self.patch_stride * (self.n_patches_seq - 1))
        if self.n_patches_seq != self.num_patches:
            raise ValueError(f"config num_patches {self.num_patches} != derived {self.n_patches_seq}")
        self._expect_weights()

    @staticmethod
    def _check_supported(cfg):
        unsupported = {
            "loss": ("mse", "mae", "huber", "pinball"),  # any point loss: no distribution head
            "scaling": ("std",),
            "mode": ("common_channel",),
            "decoder_mode": ("mix_channel", "common_channel"),
        }
        for key, allowed in unsupported.items():
            if cfg.get(key) not in allowed:
                raise ValueError(f"unsupported config {key}={cfg.get(key)!r}")
        for key in ("use_positional_encoding", "resolution_prefix_tuning", "multi_scale", "decompose",
                    "enable_forecast_channel_mixing", "decoder_raw_residual", "multi_quantile_head",
                    "enable_base_norm_always", "light_mode", "self_attn"):
            if cfg.get(key):
                raise ValueError(f"unsupported config {key}=True")
        for key in ("register_tokens", "fft_length", "decoder_adaptive_patching_levels"):
            if cfg.get(key):
                raise ValueError(f"unsupported config {key}={cfg.get(key)}")
        if cfg.get("categorical_vocab_size_list") or cfg.get("exogenous_channel_indices"):
            raise ValueError("unsupported: categorical / exogenous channels")
        if cfg.get("masked_context_length") or cfg.get("prediction_filter_length"):
            raise ValueError("unsupported: masked_context_length / prediction_filter_length")
        if not cfg.get("use_decoder", False):
            raise ValueError("unsupported: use_decoder=False")
        if cfg.get("gate_mode", "softmax") != "softmax" or cfg.get("use_register_context_gating"):
            raise ValueError("unsupported gate mode")
        if cfg.get("norm_mlp", "LayerNorm").lower() != "layernorm":
            raise ValueError("unsupported norm_mlp")

    def _expect_weights(self):
        needed = ["backbone.encoder.patcher.weight", "decoder.adapter.weight", "head.base_forecast_block.weight"]
        for i in range(self.levels):
            for j in range(self.num_layers):
                needed.append(f"backbone.encoder.mlp_mixer_encoder.mixers.{i}.mixer_layers.{j}.feature_mixer.mlp.fc1.weight")
        for j in range(self.decoder_layers):
            needed.append(f"decoder.decoder_block.mixers.{j}.feature_mixer.mlp.fc1.weight")
        missing = [k for k in needed if k not in self.w]
        if missing:
            raise KeyError(f"checkpoint lacks {missing[:3]}...")
        if self.w["head.base_forecast_block.weight"].shape != (self.horizon, self.num_patches * self.decoder_d_model):
            raise ValueError("head shape does not match config")

    # -- blocks -----------------------------------------------------------------
    def _mlp(self, x, p):
        h = _gelu(_linear(x, self.w[p + "fc1.weight"], self.w[p + "fc1.bias"]))
        return _linear(h, self.w[p + "fc2.weight"], self.w[p + "fc2.bias"])

    def _gate(self, x, p):
        return x * _softmax(_linear(x, self.w[p + "attn_layer.weight"], self.w[p + "attn_layer.bias"]))

    def _norm(self, x, p):
        return _layer_norm(x, self.w[p + "norm.weight"], self.w[p + "norm.bias"], self.norm_eps)

    def _patch_mixer(self, h, p):  # h: (B, C, P, D); mixes P
        res = h
        h = self._norm(h, p + "norm.")
        h = np.swapaxes(h, 2, 3)  # (B, C, D, P)
        h = self._mlp(h, p + "mlp.")
        h = self._gate(h, p + "gating_block.")
        return np.swapaxes(h, 2, 3) + res

    def _feature_mixer(self, h, p):  # mixes D
        res = h
        h = self._norm(h, p + "norm.")
        h = self._mlp(h, p + "mlp.")
        h = self._gate(h, p + "gating_block.")
        return h + res

    def _channel_mixer(self, h, p):  # mixes C; gate BEFORE the MLP (reference order)
        res = h
        h = self._norm(h, p + "norm.")
        h = np.transpose(h, (0, 3, 2, 1))  # (B, D, P, C)
        h = self._gate(h, p + "gating_block.")
        h = self._mlp(h, p + "mlp.")
        return np.transpose(h, (0, 3, 2, 1)) + res

    def _layer(self, h, p, mix_channel):
        if mix_channel:
            h = self._channel_mixer(h, p + "channel_feature_mixer.")
        if h.shape[2] > 1:
            h = self._patch_mixer(h, p + "patch_mixer.")
        return self._feature_mixer(h, p + "feature_mixer.")

    # -- forward ------------------------------------------------------------------
    def forward(self, x):
        """x: (B, context, C) float32 -> (B, horizon, len(pred_idx)) in data units."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        B, T, C = x.shape
        if T != self.context or C != self.n_channels:
            raise ValueError(f"expected ({self.context}, {self.n_channels}) per sample, got ({T}, {C})")
        # std scaler over the context (observed mask = ones); statistics in float64
        x64 = x.astype(np.float64)
        loc64 = x64.mean(axis=1, keepdims=True)
        var64 = np.square(x64 - loc64).mean(axis=1, keepdims=True)
        loc = loc64.astype(np.float32)
        scale = np.sqrt(var64 + 1e-5).astype(np.float32)
        xs = ((x64 - loc64) / np.sqrt(var64 + 1e-5)).astype(np.float32)
        # patchify -> (B, C, P, L)
        xs = xs[:, self.seq_start:, :]
        idx = np.arange(self.patch_len)[None, :] + self.patch_stride * np.arange(self.num_patches)[:, None]
        patched = np.transpose(xs[:, idx, :], (0, 3, 1, 2))  # (B, C, P, L)
        h = _linear(patched, self.w["backbone.encoder.patcher.weight"], self.w["backbone.encoder.patcher.bias"])
        # encoder: adaptive patching levels, common_channel
        for i, factor in enumerate(self.factors):
            h = h.reshape(B, C, self.num_patches * factor, self.d_model // factor)
            for j in range(self.num_layers):
                h = self._layer(h, f"backbone.encoder.mlp_mixer_encoder.mixers.{i}.mixer_layers.{j}.", False)
            h = h.reshape(B, C, self.num_patches, self.d_model)
        # decoder
        h = _linear(h, self.w["decoder.adapter.weight"], self.w["decoder.adapter.bias"])
        mix = self.cfg["decoder_mode"] == "mix_channel"
        for j in range(self.decoder_layers):
            h = self._layer(h, f"decoder.decoder_block.mixers.{j}.", mix)
        # head
        flat = h.reshape(B, C, self.num_patches * self.decoder_d_model)
        y = _linear(flat, self.w["head.base_forecast_block.weight"], self.w["head.base_forecast_block.bias"])  # (B, C, H)
        y = np.transpose(y, (0, 2, 1))[:, :, self.pred_idx]  # (B, H, n_pred)
        return (y * scale[:, :, self.pred_idx] + loc[:, :, self.pred_idx]).astype(np.float32)

    def predict(self, block):
        """block: (context, C) -> (horizon, n_pred), like `_Model.predict`."""
        return self.forward(np.asarray(block, dtype=np.float32)[None])[0]


def compare_numpy_backend(model_dir, n=16, seed=0, blocks=None):
    """Max |torch - numpy| over random windows (and optional real `blocks`).
    Needs torch + tsfm_public; used by check_oracle.py."""
    import torch
    from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

    ref = TinyTimeMixerForPrediction.from_pretrained(str(model_dir)).eval()
    npm = TTMNumpy(model_dir)
    rng = np.random.default_rng(seed)
    windows = []
    for k in range(n):
        x = rng.normal(size=(npm.context, npm.n_channels)).astype(np.float32) * rng.uniform(0.1, 5.0)
        if k % 4 == 1:  # constant channels (zero variance -> scale = sqrt(1e-5))
            x[:, : npm.n_channels // 3] = rng.uniform(-2, 2)
        if k % 4 == 2:  # log1p-like sparse positives
            x = np.log1p(np.abs(x)) * (rng.uniform(size=x.shape) < 0.2)
        windows.append(x)
    windows += [np.asarray(b, dtype=np.float32) for b in (blocks or [])]
    worst = 0.0
    worst_rel = 0.0
    for x in windows:
        with torch.no_grad():
            t = ref(past_values=torch.from_numpy(x)[None]).prediction_outputs[0].numpy()
        m = npm.predict(x)
        diff = np.abs(t - m).max()
        worst = max(worst, float(diff))
        worst_rel = max(worst_rel, float(diff / (np.abs(t).max() + 1e-6)))
    return worst, worst_rel, len(windows)
