"""Convert a Microsoft VibeVoice voice cache (.pt) into the safetensors format
that `audio vibevoice ... --voice-cache <file>` consumes.

Microsoft ships pre-built voice caches with the upstream demo:
https://github.com/microsoft/VibeVoice/tree/main/demo/voices/streaming_model

Each .pt is a dict of `BaseModelOutputWithPast` for `lm`, `tts_lm`, `neg_lm`,
and `neg_tts_lm`. We flatten it into a tensor dict the speech-swift loader
expects.

The Realtime-0.5B checkpoint itself does not ship the acoustic encoder, so
voice caches cannot be minted from raw audio against that checkpoint. Either
use one of Microsoft's pre-built .pt voices via this script, or encode against
VibeVoice-1.5B by passing --long-form to `audio vibevoice-encode-voice`.

Requires: torch (any 2.x), safetensors. Install in a venv:

    python3 -m venv /tmp/.vv && source /tmp/.vv/bin/activate
    pip install --quiet torch safetensors

Usage:

    python convert_vibevoice_voice.py <input.pt> <output.safetensors>
"""
from __future__ import annotations
import sys

if len(sys.argv) != 3:
    sys.exit("usage: convert_vibevoice_voice.py <input.pt> <output.safetensors>")

src, dst = sys.argv[1], sys.argv[2]

import torch
from safetensors.torch import save_file

raw = torch.load(src, weights_only=False, map_location="cpu")

out: dict[str, torch.Tensor] = {}

# Last hidden states (the loader needs three of the four).
out["lm_hidden"] = raw["lm"].last_hidden_state.contiguous().to(torch.float32)
out["tts_lm_hidden"] = raw["tts_lm"].last_hidden_state.contiguous().to(torch.float32)
out["neg_tts_lm_hidden"] = raw["neg_tts_lm"].last_hidden_state.contiguous().to(torch.float32)


def kv_layers(pkv):
    """Return [(K, V), ...] for the populated layers in a DynamicCache.

    Supports both the legacy layout (`pkv.key_cache[i]`/`pkv.value_cache[i]`,
    transformers <=4.56) and the new layout (`pkv.layers[i].keys/.values`,
    transformers >=4.57). Stops at the first None layer (DynamicCache holds a
    slot per logical layer and only fills the ones the forward actually used).
    """
    if hasattr(pkv, "key_cache"):
        return [
            (pkv.key_cache[i], pkv.value_cache[i])
            for i in range(len(pkv.key_cache))
        ]
    layers = []
    for layer in pkv.layers:
        try:
            k, v = layer.keys, layer.values
        except AttributeError:
            break
        if k is None:
            break
        layers.append((k, v))
    return layers


def dump(name: str) -> None:
    layers = kv_layers(raw[name].past_key_values)
    for i, (k, v) in enumerate(layers):
        out[f"{name}_key_{i}"] = k.contiguous().to(torch.float32)
        out[f"{name}_value_{i}"] = v.contiguous().to(torch.float32)
    last = layers[-1][0].shape if layers else "empty"
    print(f"  {name:>11s}: {len(layers):>2d} layers, last K shape {last}")


for name in ("lm", "tts_lm", "neg_lm", "neg_tts_lm"):
    dump(name)

save_file(out, dst)
print(f"\nWrote {dst} ({len(out)} tensors).")
