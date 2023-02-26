TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
END_OPT_STRING = "\n" + "=" * 60 + "\n"
import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # supress warnings
import h5py

import bingo.optimizer.tf_quantum as tfq
from bingo.gate_sets.gate_set import GateSet
import qutip as qt
import datetime
import time
from .operators import DisplacementOperator, SQROperator

# Qubit coupled to cavity
# gate set is: SQR + Disp
#circuit Block is: [1.) Disp(alpha)...2.) SQR(phi,theta)]
class SQRDispGateSet(GateSet):
    def __init__(self,  name="ECD_control", **kwargs):
        super().__init__(name)
        # combine all keyword arguments
        self.parameters = {
            **self.parameters,
            **kwargs,
        }  # python 3.9: self.parameters | kwargs
        self.SQR_op = lambda phis, thetas: SQROperator(self.parameters["N_cav"])(phis, thetas)
        self.disp_op = lambda alpha: tfq.two_mode_op(DisplacementOperator(self.parameters["N_cav"])(alpha),1,other_dim=2)

    @property
    def parameter_names(self):
        return [
            "alphas_rho",
            "alphas_angle",
            "phis",
            "thetas"
        ]

    def randomization_ranges(self):
        return {
            "alphas_rho": (
                -1 * self.parameters["alpha_scale"],
                1 * self.parameters["alpha_scale"],
            ),
            "alphas_angle": (-np.pi, np.pi),
            "thetas": (-np.pi, np.pi, self.parameters["N_SQR"]),
            "phis": (-np.pi, np.pi, self.parameters["N_SQR"])
        }

    @tf.function
    def batch_construct_block_operators(self, opt_vars):
        alphas_rho = opt_vars["alphas_rho"]
        alphas_angle = opt_vars["alphas_angle"]
        thetas = opt_vars["thetas"]
        phis = opt_vars["phis"]

        As = tf.cast(alphas_rho, dtype=tf.complex64) * tf.math.exp(
                tf.constant(1j, dtype=tf.complex64)
                * tf.cast(alphas_angle, dtype=tf.complex64)
            )
        
        Ds_a = self.disp_op(As)
        SQRs = self.SQR_op(phis, thetas)
        blocks = SQRs @ Ds_a 
        return blocks

    def preprocess_params_before_saving(self, opt_params, *args):
        processed_params = {}
        processed_params["alphas"] = tf.Variable(
            tf.cast(opt_params["alphas_rho"], dtype=tf.complex64)
            * tf.math.exp(1j * tf.cast(opt_params["alphas_angle"], dtype=tf.complex64)),
            name="betas a",
            dtype=tf.complex64,
        )
        processed_params["thetas"] = (opt_params["thetas"] + np.pi) % (2 * np.pi) - np.pi
        processed_params["phis"] = (opt_params["phis"] + np.pi) % (2 * np.pi) - np.pi
        return processed_params
