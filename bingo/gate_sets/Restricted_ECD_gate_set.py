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
from .operators import CD_q_sx, CD_p_sy


class RestrictedECDGateSet(GateSet):
    def __init__(self, N_blocks=20, name="ECD_control", **kwargs):
        super().__init__(name)
        # combine all keyword arguments
        self.parameters = {
            **self.parameters,
            **kwargs,
        }  # python 3.9: self.parameters | kwargs
        N_cav = self.parameters["N_cav"]
        self.CD_q_sx_op = lambda betas_q: CD_q_sx(N_cav)(betas_q)
        self.CD_p_sy_op = lambda betas_p: CD_p_sy(N_cav)(betas_p)

    @property
    def parameter_names(self):
        return [
            "betas_q",
            "betas_p"
        ]

    def randomization_ranges(self):
        return {
            "betas_q": (
                -1 * self.parameters["beta_scale"],
                1 * self.parameters["beta_scale"],
            ),
            "betas_p": (
                -1 * self.parameters["beta_scale"],
                1 * self.parameters["beta_scale"],
            )
        }

    @tf.function
    def batch_construct_block_operators(self, opt_vars):
        betas_q = opt_vars["betas_q"]
        betas_p = opt_vars["betas_p"]

        blocks = self.CD_p_sy_op(betas_p)@self.CD_q_sx_op(betas_q)
        return blocks
