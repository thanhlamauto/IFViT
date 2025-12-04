import os
import sys
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze, unfreeze
from flax.serialization import to_state_dict


FINGERNET_SRC_DIR = "/kaggle/working/IFViT/FingerNet/src"
KERAS_WEIGHTS_PATH = "/kaggle/working/IFViT/IFViT/FingerNet/models/released_version/Model.model"
OUTPUT_NPZ_PATH = "/kaggle/working/IFViT/IFViT/fingernet_jax/fingernet_flax_params.npz"


def load_keras_model():
    """
    Import Keras architecture from the original FingerNet repo and load
    pretrained weights from Model.model.
    """
    # Ensure we can import train_test_deploy and its local utils
    if FINGERNET_SRC_DIR not in sys.path:
        sys.path.append(FINGERNET_SRC_DIR)

    # Kaggle typically has the standalone "keras" package installed (Keras 3),
    # while the original FingerNet code expects TF1-style "keras" that matches
    # tf.keras.* APIs. We monkey-patch sys.modules so that any
    # "import keras" inside train_test_deploy.py actually resolves to tf.keras.
    import tensorflow as tf  # type: ignore
    if "keras" in sys.modules:
        # Drop pre-existing standalone keras if present.
        del sys.modules["keras"]
    sys.modules["keras"] = tf.keras

    # train_test_deploy.py calls argparse.parse_args() at import time.
    # Provide it with a minimal, valid argv to avoid parsing our script args.
    saved_argv = sys.argv[:]
    sys.argv = ["train_test_deploy.py", "0", "deploy"]
    try:
        import train_test_deploy as ttd  # type: ignore
    finally:
        sys.argv = saved_argv

    # Build the Keras model and load weights explicitly from the absolute path.
    # Input shape doesn't matter as long as it's compatible with the network.
    model = ttd.get_main_net((512, 512, 1), KERAS_WEIGHTS_PATH)
    return model


def init_flax_variables():
    """
    Initialise Flax FingerNet variables (params + batch_stats) on a dummy input.
    """
    from fingernet_flax import FingerNet

    rng = jax.random.PRNGKey(0)
    dummy = jnp.ones((1, 512, 512, 1), dtype=jnp.float32)
    variables = FingerNet().init(rng, dummy, train=False)
    return variables


def _assign_param(tree: Dict[str, Any], module_name: str, param_name: str, value: np.ndarray) -> bool:
    """
    Recursively search the nested Flax param/batch_stats dict and assign
    value to tree[...][module_name][param_name] when found.
    """
    if not isinstance(tree, dict):
        return False

    if module_name in tree and isinstance(tree[module_name], dict) and param_name in tree[module_name]:
        tree[module_name][param_name] = jnp.array(value)
        return True

    for v in tree.values():
        if isinstance(v, dict) and _assign_param(v, module_name, param_name, value):
            return True

    return False


def convert_weights():
    keras_model = load_keras_model()
    variables = init_flax_variables()

    params = unfreeze(variables["params"])
    batch_stats = unfreeze(variables.get("batch_stats", {}))

    for layer in keras_model.layers:
        name = layer.name
        cls = layer.__class__.__name__

        # Conv2D layers
        if cls == "Conv2D":
            # The two Gabor convs in the original model are implemented as
            # fixed convolutions in JAX (no trainable params), so we skip them.
            if name in ("enh_img_real_1", "enh_img_imag_1"):
                continue

            weights = layer.get_weights()
            if len(weights) == 2:
                w, b = weights
            elif len(weights) == 1:
                w, = weights
                b = np.zeros((w.shape[-1],), dtype=w.dtype)
            else:
                continue

            ok_w = _assign_param(params, name, "kernel", w)
            ok_b = _assign_param(params, name, "bias", b)
            if not (ok_w and ok_b):
                print(f"[WARN] Conv2D weights for layer '{name}' not mapped into Flax params.")

        # BatchNormalization layers
        elif cls == "BatchNormalization":
            weights = layer.get_weights()
            if len(weights) != 4:
                continue
            gamma, beta, moving_mean, moving_var = weights

            ok_scale = _assign_param(params, name, "scale", gamma)
            ok_bias = _assign_param(params, name, "bias", beta)
            ok_mean = _assign_param(batch_stats, name, "mean", moving_mean)
            ok_var = _assign_param(batch_stats, name, "var", moving_var)

            if not (ok_scale and ok_bias and ok_mean and ok_var):
                print(f"[WARN] BatchNorm weights for layer '{name}' not fully mapped.")

        # PReLU layers
        elif cls == "PReLU":
            weights = layer.get_weights()
            if len(weights) != 1:
                continue
            alpha, = weights  # often shape (1, 1, C) with shared_axes=[1,2]
            alpha = np.squeeze(alpha)  # -> (C,)
            ok_alpha = _assign_param(params, name, "alpha", alpha)
            if not ok_alpha:
                print(f"[WARN] PReLU weights for layer '{name}' not mapped.")

    # Save converted params
    os.makedirs(os.path.dirname(OUTPUT_NPZ_PATH), exist_ok=True)
    np.savez(
        OUTPUT_NPZ_PATH,
        params=to_state_dict(freeze(params)),
        batch_stats=to_state_dict(freeze(batch_stats)),
    )
    print(f"[INFO] Saved Flax parameters to: {OUTPUT_NPZ_PATH}")


if __name__ == "__main__":
    convert_weights()


