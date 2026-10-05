"""Tiny random Hugging Face models for the tests.

These builders run in a separate interpreter (``isolation.run_isolated``), which
imports this module; it must therefore never import TensorFlow (directly or
through ctnet_pd.pipeline / training / model).
"""


def save_tiny_wavlm(path: str) -> None:
    import transformers

    config = transformers.WavLMConfig(hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                                      intermediate_size=32, conv_dim=(8, 8), conv_stride=(5, 4),
                                      conv_kernel=(10, 8), num_conv_pos_embeddings=16,
                                      num_conv_pos_embedding_groups=2)
    transformers.WavLMModel(config).save_pretrained(path)
    transformers.Wav2Vec2FeatureExtractor(sampling_rate=16000).save_pretrained(path)


def save_tiny_image_classifier(path: str, arch: str) -> None:
    import transformers

    if arch == "vit":
        cfg = transformers.ViTConfig(image_size=32, patch_size=8, hidden_size=16, num_hidden_layers=1,
                                     num_attention_heads=2, intermediate_size=32, num_labels=2)
        transformers.ViTForImageClassification(cfg).save_pretrained(path)
    elif arch == "swin":
        cfg = transformers.SwinConfig(image_size=32, patch_size=4, embed_dim=8, depths=[1, 1],
                                      num_heads=[1, 2], window_size=4, num_labels=2)
        transformers.SwinForImageClassification(cfg).save_pretrained(path)
    else:
        raise ValueError(arch)
