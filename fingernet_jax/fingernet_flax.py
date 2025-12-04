# fingernet_flax.py
from typing import Any, Dict

import jax
import jax.numpy as jnp
from flax import linen as nn

from utils_jax import gabor_bank


class PReLU(nn.Module):
    init_alpha: float = 0.0

    @nn.compact
    def __call__(self, x):
        c = x.shape[-1]
        alpha = self.param(
            'alpha',
            nn.initializers.constant(self.init_alpha),
            (c,)
        )
        alpha = alpha.reshape((1, 1, 1, c))
        return jnp.where(x >= 0, x, alpha * x)


class ConvBN(nn.Module):
    features: int
    kernel_size: tuple
    strides: tuple = (1, 1)
    dilation: tuple = (1, 1)
    name_suffix: str = ""

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='SAME',
            kernel_dilation=self.dilation,
            name=f"conv-{self.name_suffix}",
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            name=f"bn-{self.name_suffix}",
        )(x)
        return x


class ConvBNPReLU(nn.Module):
    features: int
    kernel_size: tuple
    strides: tuple = (1, 1)
    dilation: tuple = (1, 1)
    name_suffix: str = ""

    @nn.compact
    def __call__(self, x, train: bool):
        conv_name = f"conv{self.name_suffix}" if self.dilation == (1, 1) else f"atrousconv{self.name_suffix}"
        x = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='SAME',
            kernel_dilation=self.dilation,
            name=conv_name,
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            name=f"bn-{self.name_suffix}",
        )(x)
        x = PReLU(init_alpha=0.0, name=f"prelu-{self.name_suffix}")(x)
        return x


def img_normalization_jax(img_input, m0=0.0, var0=1.0):
    m = jnp.mean(img_input, axis=(1, 2, 3), keepdims=True)
    var = jnp.var(img_input, axis=(1, 2, 3), keepdims=True)
    after = jnp.sqrt(var0 * jnp.square(img_input - m) / (var + 1e-8))
    image_n = jnp.where(img_input > m, m0 + after, m0 - after)
    return image_n


def softsign(x):
    return x / (1.0 + jnp.abs(x))


def select_max(x):
    x = x / (jnp.max(x, axis=-1, keepdims=True) + 1e-8)
    x = jnp.where(x > 0.999, x, 0.0)
    x = x / (jnp.sum(x, axis=-1, keepdims=True) + 1e-8)
    return x


def upsample_nn(x, scale=(8, 8)):
    x = jnp.repeat(x, scale[0], axis=1)
    x = jnp.repeat(x, scale[1], axis=2)
    return x


def ori_highest_peak_jax(y_pred, length=180):
    from utils_jax import gausslabel
    glabel = gausslabel(length=length, stride=2).astype(jnp.float32)  # (1,1,90,1)
    glabel = jnp.transpose(glabel, (0, 1, 3, 2))  # -> (1,1,1,90) but conv_general_dilated expects HWIO
    glabel = glabel.reshape((1, 1, 1, glabel.shape[-1]))  # H=1,W=1
    # conv along orientation channel
    # Trick: dùng conv depthwise 1x1
    return jax.lax.conv_general_dilated(
        y_pred,
        glabel,  # (1,1,1,90)
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )


class FingerNet(nn.Module):
    """
    JAX/Flax version of get_main_net() for inference (deploy mode).
    Input shape: (B, H, W, 1), any H,W multiples of 8.
    """

    @nn.compact
    def __call__(self, x, train: bool = False) -> Dict[str, Any]:
        # img_norm
        x = img_normalization_jax(x)

        # VGG-like feature extractor
        conv = ConvBNPReLU(64, (3, 3), name_suffix="1_1")(x, train)
        conv = ConvBNPReLU(64, (3, 3), name_suffix="1_2")(conv, train)
        conv = nn.max_pool(conv, window_shape=(2, 2), strides=(2, 2), padding='SAME')

        conv = ConvBNPReLU(128, (3, 3), name_suffix="2_1")(conv, train)
        conv = ConvBNPReLU(128, (3, 3), name_suffix="2_2")(conv, train)
        conv = nn.max_pool(conv, window_shape=(2, 2), strides=(2, 2), padding='SAME')

        conv = ConvBNPReLU(256, (3, 3), name_suffix="3_1")(conv, train)
        conv = ConvBNPReLU(256, (3, 3), name_suffix="3_2")(conv, train)
        conv = ConvBNPReLU(256, (3, 3), name_suffix="3_3")(conv, train)
        conv = nn.max_pool(conv, window_shape=(2, 2), strides=(2, 2), padding='SAME')

        # multi-scale ASPP
        # scale_1
        scale_1 = ConvBNPReLU(256, (3, 3), dilation=(1, 1), name_suffix="4_1")(conv, train)
        ori_1 = ConvBNPReLU(128, (1, 1), name_suffix="ori_1_1")(scale_1, train)
        ori_1 = nn.Conv(90, (1, 1), padding='SAME', name='ori_1_2')(ori_1)

        seg_1 = ConvBNPReLU(128, (1, 1), name_suffix="seg_1_1")(scale_1, train)
        seg_1 = nn.Conv(1, (1, 1), padding='SAME', name='seg_1_2')(seg_1)

        # scale_2
        scale_2 = ConvBNPReLU(256, (3, 3), dilation=(4, 4), name_suffix="4_2")(conv, train)
        ori_2 = ConvBNPReLU(128, (1, 1), name_suffix="ori_2_1")(scale_2, train)
        ori_2 = nn.Conv(90, (1, 1), padding='SAME', name='ori_2_2')(ori_2)

        seg_2 = ConvBNPReLU(128, (1, 1), name_suffix="seg_2_1")(scale_2, train)
        seg_2 = nn.Conv(1, (1, 1), padding='SAME', name='seg_2_2')(seg_2)

        # scale_3
        scale_3 = ConvBNPReLU(256, (3, 3), dilation=(8, 8), name_suffix="4_3")(conv, train)
        ori_3 = ConvBNPReLU(128, (1, 1), name_suffix="ori_3_1")(scale_3, train)
        ori_3 = nn.Conv(90, (1, 1), padding='SAME', name='ori_3_2')(ori_3)

        seg_3 = ConvBNPReLU(128, (1, 1), name_suffix="seg_3_1")(scale_3, train)
        # NOTE: Keras uses name='seg_3_2' here – keep exactly the same for weight mapping
        seg_3 = nn.Conv(1, (1, 1), padding='SAME', name='seg_3_2')(seg_3)

        # ori fusion
        ori_out = ori_1 + ori_2 + ori_3
        ori_out_1 = nn.sigmoid(ori_out)
        # ori_out_2 trong code gốc cũng là sigmoid(ori_out), dùng cho loss khác; infer có thể giống nhau
        ori_out_2 = nn.sigmoid(ori_out)

        # seg fusion
        seg_out = seg_1 + seg_2 + seg_3
        seg_out = nn.sigmoid(seg_out)

        # enhance part: gabor filtering
        filters_cos_np, filters_sin_np = gabor_bank(stride=2, Lambda=8)
        filters_cos = jnp.array(filters_cos_np)
        filters_sin = jnp.array(filters_sin_np)

        # conv with fixed gabor kernels, no bias
        filter_img_real = jax.lax.conv_general_dilated(
            x,
            filters_cos,
            window_strides=(1, 1),
            padding='SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        filter_img_imag = jax.lax.conv_general_dilated(
            x,
            filters_sin,
            window_strides=(1, 1),
            padding='SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )

        ori_peak = ori_highest_peak_jax(ori_out_1)
        ori_peak = select_max(ori_peak)
        upsample_ori = upsample_nn(ori_peak, (8, 8))

        seg_round = softsign(seg_out)
        upsample_seg = upsample_nn(seg_round, (8, 8))

        mul_mask_real = filter_img_real * upsample_ori
        enh_img_real = jnp.sum(mul_mask_real, axis=-1, keepdims=True)

        mul_mask_imag = filter_img_imag * upsample_ori
        enh_img_imag = jnp.sum(mul_mask_imag, axis=-1, keepdims=True)

        enh_img = jnp.arctan2(enh_img_imag, enh_img_real)
        enh_seg_img = jnp.concatenate([enh_img, upsample_seg], axis=-1)

        # mnt part
        mnt_conv = ConvBNPReLU(64, (9, 9), name_suffix="mnt_1_1")(enh_seg_img, train)
        mnt_conv = nn.max_pool(mnt_conv, window_shape=(2, 2), strides=(2, 2), padding='SAME')

        mnt_conv = ConvBNPReLU(128, (5, 5), name_suffix="mnt_2_1")(mnt_conv, train)
        mnt_conv = nn.max_pool(mnt_conv, window_shape=(2, 2), strides=(2, 2), padding='SAME')

        mnt_conv = ConvBNPReLU(256, (3, 3), name_suffix="mnt_3_1")(mnt_conv, train)
        mnt_conv = nn.max_pool(mnt_conv, window_shape=(2, 2), strides=(2, 2), padding='SAME')

        # mnt orientation
        mnt_o_1 = jnp.concatenate([mnt_conv, ori_out_1], axis=-1)
        mnt_o_2 = ConvBNPReLU(256, (1, 1), name_suffix="mnt_o_1_1")(mnt_o_1, train)
        mnt_o_3 = nn.Conv(180, (1, 1), padding='SAME', name='mnt_o_1_2')(mnt_o_2)
        mnt_o_out = nn.sigmoid(mnt_o_3)

        # mnt w
        mnt_w_1 = ConvBNPReLU(256, (1, 1), name_suffix="mnt_w_1_1")(mnt_conv, train)
        mnt_w_2 = nn.Conv(8, (1, 1), padding='SAME', name='mnt_w_1_2')(mnt_w_1)
        mnt_w_out = nn.sigmoid(mnt_w_2)

        # mnt h
        mnt_h_1 = ConvBNPReLU(256, (1, 1), name_suffix="mnt_h_1_1")(mnt_conv, train)
        mnt_h_2 = nn.Conv(8, (1, 1), padding='SAME', name='mnt_h_1_2')(mnt_h_1)
        mnt_h_out = nn.sigmoid(mnt_h_2)

        # mnt score
        mnt_s_1 = ConvBNPReLU(256, (1, 1), name_suffix="mnt_s_1_1")(mnt_conv, train)
        mnt_s_2 = nn.Conv(1, (1, 1), padding='SAME', name='mnt_s_1_2')(mnt_s_1)
        mnt_s_out = nn.sigmoid(mnt_s_2)

        return {
            "enh_img_real": enh_img_real,
            "ori_out_1": ori_out_1,
            "ori_out_2": ori_out_2,
            "seg_out": seg_out,
            "mnt_o_out": mnt_o_out,
            "mnt_w_out": mnt_w_out,
            "mnt_h_out": mnt_h_out,
            "mnt_s_out": mnt_s_out,
        }
