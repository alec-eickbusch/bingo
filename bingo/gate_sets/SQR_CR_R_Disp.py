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
from .operators import DisplacementOperator, ConditionalRotation, SQROperator, QubitRotationXY

# Qubit coupled to cavity
# gate set is: Selective qubit rotations (SQR) + Controlled rotations (CR) + qubit rotations (R) + oscillator displacement (D)
#block is: 1.) D(alpha) R(phi, theta) 2.) SQR  3.) CR
class SQRCRRDispGateSet(GateSet):
    def __init__(self,  name="SQR_CR_R_Disp", **kwargs):
        super().__init__(name)
        # combine all keyword arguments
        self.parameters = {
            **self.parameters,
            **kwargs,
        }  # python 3.9: self.parameters | kwargs
        self.CR_op = lambda CR_theta: ConditionalRotation(self.parameters["N_cav"])(CR_theta)
        self.SQR_op = lambda SQR_phis, SQR_thetas: SQROperator(self.parameters["N_cav"])(SQR_phis, SQR_thetas)
        self.disp_op = lambda alpha: tfq.two_mode_op(DisplacementOperator(self.parameters["N_cav"])(alpha),1,other_dim=2)
        self.R_op = lambda phi, theta: tfq.two_mode_op(QubitRotationXY()(theta, phi), 0, other_dim = self.parameters["N_cav"])

    @property
    def parameter_names(self):
        return [
            "alphas_rho",
            "alphas_angle",
            "phis",
            "thetas",
            "SQR_phis",
            "SQR_thetas",
            "CR_thetas"
        ]

    def randomization_ranges(self):
        return {
            "alphas_rho": (
                -1 * self.parameters["alpha_scale"],
                1 * self.parameters["alpha_scale"],
            ),
            "alphas_angle": (-np.pi, np.pi),
            "phis":(-np.pi,np.pi),
            "thetas":(-np.pi,np.pi),
            "SQR_thetas": (-np.pi, np.pi, self.parameters["N_SQR"]),
            "SQR_phis": (-np.pi, np.pi, self.parameters["N_SQR"]),
            "CR_thetas":(-np.pi,np.pi)
        }

    @tf.function
    def batch_construct_block_operators(self, opt_vars):
        alphas_rho = opt_vars["alphas_rho"]
        alphas_angle = opt_vars["alphas_angle"]
        thetas = opt_vars["thetas"]
        phis = opt_vars["phis"]
        SQR_thetas = opt_vars["SQR_thetas"]
        SQR_phis = opt_vars["SQR_phis"]
        CR_thetas = opt_vars["CR_thetas"]

        As = tf.cast(alphas_rho, dtype=tf.complex64) * tf.math.exp(
                tf.constant(1j, dtype=tf.complex64)
                * tf.cast(alphas_angle, dtype=tf.complex64)
            )
        
        Rs = self.R_op(phis, thetas)
        Ds = self.disp_op(As)
        SQRs = self.SQR_op(SQR_phis, SQR_thetas)
        CRs = self.CR_op(CR_thetas)
        blocks = CRs@SQRs@Ds@Rs
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
        processed_params["SQR_phis"] = (opt_params["SQR_phis"] + np.pi) % (2 * np.pi) - np.pi
        processed_params["SQR_thetas"] = (opt_params["SQR_thetas"] + np.pi) % (2 * np.pi) - np.pi
        processed_params["CR_thetas"] = (opt_params["CR_thetas"] + np.pi) % (2 * np.pi) - np.pi
        return processed_params
