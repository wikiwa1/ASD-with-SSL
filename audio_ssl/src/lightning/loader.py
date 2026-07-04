from __future__ import annotations

# Pick the right SSL LightningModule for a run based on its config `arch:` field, so the
# shared embedding eval / monitor can load either a JEPA (I-JEPA) or a LeJEPA checkpoint.
# Both expose target_encoder / context_encoder / normalize, so downstream code is identical.


def load_ssl_module(config: dict, checkpoint_path, device):
    arch = str(config.get("arch", "jepa")).lower()
    if arch == "lejepa":
        from audio_ssl.src.lightning.lejepa_module import LitLeJEPA
        return LitLeJEPA.load_from_checkpoint(str(checkpoint_path), map_location=device)
    if arch == "beats":
        from audio_ssl.src.lightning.beats_jepa_module import LitBEATsJEPA
        # beats_checkpoint=None: the Lightning ckpt carries all weights; the original
        # BEATs .pt is only needed at training time.
        return LitBEATsJEPA.load_from_checkpoint(str(checkpoint_path), map_location=device,
                                                 beats_checkpoint=None)
    from audio_ssl.src.lightning.jepa_module import LitJEPA
    return LitJEPA.load_from_checkpoint(str(checkpoint_path), map_location=device)
