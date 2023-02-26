TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
END_OPT_STRING = "\n" + "=" * 60 + "\n"
import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # suppress warnings
import h5py

import bingo.optimizer.tf_quantum as tfq
from bingo.gate_sets.gate_set import GateSet
import qutip as qt
import datetime
import time
from .operators import DisplacementOperator


class ECDGateSet(GateSet):
    def __init__(self, N_blocks=20, name="ECD_control", **kwargs):
        super().__init__(name)
        # combine all keyword arguments
        self.parameters = {
            **self.parameters,
            **kwargs,
        }  # python 3.9: self.parameters | kwargs
        self.disp_op = DisplacementOperator(self.parameters["N_cav"])

    @property
    def parameter_names(self):
        return [
            "betas_rho",
            "betas_angle",
            "phis",
            "thetas",
        ]

    def randomization_ranges(self):
        return {
            "betas_rho": (
                -1 * self.parameters["beta_scale"],
                1 * self.parameters["beta_scale"],
            ),
            "betas_angle": (-np.pi, np.pi),
            "phis": (-np.pi, np.pi),
            "thetas": (-np.pi, np.pi),
        }

    @tf.function
    def batch_construct_block_operators(self, opt_vars):
        betas_rho = opt_vars["betas_rho"]
        betas_angle = opt_vars["betas_angle"]
        phis = opt_vars["phis"]
        thetas = opt_vars["thetas"]

        # conditional displacements
        Bs = (
            tf.cast(betas_rho, dtype=tf.complex64)
            / tf.constant(2, dtype=tf.complex64)
            * tf.math.exp(
                tf.constant(1j, dtype=tf.complex64)
                * tf.cast(betas_angle, dtype=tf.complex64)
            )
        )

        ds_g = self.disp_op(Bs)
        ds_e = tf.linalg.adjoint(ds_g)

        Phis = phis - tf.constant(np.pi, dtype=tf.float32) / tf.constant(
            2, dtype=tf.float32
        )
        Thetas = thetas / tf.constant(2, dtype=tf.float32)
        Phis = tf.cast(
            tf.reshape(Phis, [Phis.shape[0], Phis.shape[1], 1, 1]), dtype=tf.complex64
        )
        Thetas = tf.cast(
            tf.reshape(Thetas, [Thetas.shape[0], Thetas.shape[1], 1, 1]),
            dtype=tf.complex64,
        )

        exp = tf.math.exp(tf.constant(1j, dtype=tf.complex64) * Phis)
        exp_dag = tf.linalg.adjoint(exp)
        cos = tf.math.cos(Thetas)
        sin = tf.math.sin(Thetas)

        # constructing the blocks of the matrix
        ul = cos * ds_g
        ll = exp * sin * ds_e
        ur = tf.constant(-1, dtype=tf.complex64) * exp_dag * sin * ds_g
        lr = cos * ds_e

        # without pi pulse, block matrix is:
        # (ul, ur)
        # (ll, lr)
        # however, with pi pulse included:
        # (ll, lr)
        # (ul, ur)
        # pi pulse also adds -i phase, however don't need to trck it unless using multiple oscillators.a
        blocks =  tf.concat([tf.concat([ll, lr], 3), tf.concat([ul, ur], 3)], 2)
                
        return blocks

    def preprocess_params_before_saving(self, opt_params, *args):
        processed_params = {}
        processed_params["betas"] = tf.Variable(
            tf.cast(opt_params["betas_rho"], dtype=tf.complex64)
            * tf.math.exp(1j * tf.cast(opt_params["betas_angle"], dtype=tf.complex64)),
            name="betas",
            dtype=tf.complex64,
        )
        processed_params["phis"] = (opt_params["phis"] + np.pi) % (2 * np.pi) - np.pi
        processed_params["thetas"] = (opt_params["thetas"] + np.pi) % (
            2 * np.pi
        ) - np.pi

        return processed_params
