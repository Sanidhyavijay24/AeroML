# -*- coding: utf-8 -*-
"""
@file models.py
@description Model architecture and training initialization functions
@module aeroml
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def set_all_seeds(seed: int) -> None:
    """Set random seeds for reproducibility across NumPy and TensorFlow."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def dense_block(x, units: int, dropout: float):
    """A standard dense block with normalization, Swish activation, and optional dropout."""
    x = layers.Dense(units, kernel_initializer="he_normal")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("swish")(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    return x


def build_forward_model(profile_dim: int, scalar_dim: int, flow_dim: int) -> keras.Model:
    """Build the forward MLP prediction model ensemble component."""
    profile_in = layers.Input(shape=(profile_dim,), name="profile")
    scalar_in = layers.Input(shape=(scalar_dim,), name="scalar")
    flow_in = layers.Input(shape=(flow_dim,), name="flow")

    p = layers.GaussianNoise(0.01)(profile_in)
    p = dense_block(p, 512, 0.10)
    p = dense_block(p, 256, 0.10)
    p = dense_block(p, 128, 0.05)

    s = dense_block(scalar_in, 64, 0.05)
    s = dense_block(s, 32, 0.00)

    f = dense_block(flow_in, 64, 0.05)
    f = dense_block(f, 32, 0.00)

    x = layers.Concatenate()([p, s, f])
    x = dense_block(x, 256, 0.10)
    x = dense_block(x, 128, 0.05)
    shared = dense_block(x, 64, 0.00)

    ld_head = dense_block(shared, 32, 0.00)
    cl_head = dense_block(shared, 32, 0.00)
    cd_head = dense_block(shared, 32, 0.00)

    outputs = {
        "ldmax": layers.Dense(1, name="ldmax")(ld_head),
        "clmax": layers.Dense(1, name="clmax")(cl_head),
        "cdmin_log": layers.Dense(1, name="cdmin_log")(cd_head),
    }

    return keras.Model(
        inputs=[profile_in, scalar_in, flow_in],
        outputs=outputs,
        name="AeroML_XFOIL_Forward_MLP",
    )
