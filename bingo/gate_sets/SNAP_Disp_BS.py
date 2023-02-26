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
from .operators import DisplacementOperator, SNAPOperator, BeamSplitterOperator

# Three modes Alice, Bob, and transmon.
#However, only Alice is coupled to the transmon.
#Alice and Bob are coupled via a beam splitter.
# Alice and Bob also have displacements.

#circuit Block is: [Disp(a)Disp(b)...SNAP(a)...BS(a&b)]
class SNAPDispBSGateSet(GateSet):
    def __init__(self,  name="ECD_control", **kwargs):
        super().__init__(name)
        # combine all keyword arguments
        self.parameters = {
            **self.parameters,
            **kwargs,
        }  # python 3.9: self.parameters | kwargs
        self.snap_op = lambda thetas: tfq.two_mode_op(
        SNAPOperator(self.parameters["N_cav"])(thetas),0)
        self.disp_op_a = lambda alpha: tfq.two_mode_op(DisplacementOperator(self.parameters["N_cav"])(alpha),0)
        self.disp_op_b = lambda beta: tfq.two_mode_op(DisplacementOperator(self.parameters["N_cav"])(beta),1)
        self.BS_op = BeamSplitterOperator(self.parameters["N_cav"])

    @property
    def parameter_names(self):
        return [
            "alphas_rho",
            "alphas_angle",
            "betas_rho",
            "betas_angle",
            "SNAP_thetas",
            "BS_phis",
            "BS_thetas",
        ]

    def randomization_ranges(self):
        return {
            "alphas_rho": (
                -1 * self.parameters["beta_scale"],
                1 * self.parameters["beta_scale"],
            ),
            "alphas_angle": (-np.pi, np.pi),
            "betas_rho": (
                -1 * self.parameters["beta_scale"],
                1 * self.parameters["beta_scale"],
            ),
            "betas_angle": (-np.pi, np.pi),
            "SNAP_thetas": (-np.pi, np.pi, self.parameters["N_snap"]),
            "BS_thetas": (
                -np.pi/2.0,np.pi/2.0
            ),
            "BS_phis": (
                -np.pi,np.pi
            ),
        }

    @tf.function
    def batch_construct_block_operators(self, opt_vars):
        alphas_rho = opt_vars["alphas_rho"]
        alphas_angle = opt_vars["alphas_angle"]
        betas_rho = opt_vars["betas_rho"]
        betas_angle = opt_vars["betas_angle"]
        SNAP_thetas = opt_vars["SNAP_thetas"]
        BS_thetas = opt_vars["BS_thetas"]
        BS_phis = opt_vars["BS_phis"]

        # conditional displacements

        As = tf.cast(alphas_rho, dtype=tf.complex64) * tf.math.exp(
                tf.constant(1j, dtype=tf.complex64)
                * tf.cast(alphas_angle, dtype=tf.complex64)
            )
        Bs = tf.cast(betas_rho, dtype=tf.complex64) * tf.math.exp(
                tf.constant(1j, dtype=tf.complex64)
                * tf.cast(betas_angle, dtype=tf.complex64)
            )
        
        Ds_a = self.disp_op_a(As)
        Ds_b = self.disp_op_b(Bs)
        SNAPs = self.snap_op(SNAP_thetas)
        BS = self.BS_op(BS_phis, BS_thetas)
        blocks = BS @ SNAPs @ Ds_a @ Ds_b
        return blocks

    def preprocess_params_before_saving(self, opt_params, *args):
        processed_params = {}
        processed_params["alphas"] = tf.Variable(
            tf.cast(opt_params["alphas_rho"], dtype=tf.complex64)
            * tf.math.exp(1j * tf.cast(opt_params["alphas_angle"], dtype=tf.complex64)),
            name="betas a",
            dtype=tf.complex64,
        )
        processed_params["betas"] = tf.Variable(
            tf.cast(opt_params["betas_rho"], dtype=tf.complex64)
            * tf.math.exp(1j * tf.cast(opt_params["betas_angle"], dtype=tf.complex64)),
            name="betas b",
            dtype=tf.complex64,
        )
        processed_params["SNAP_thetas"] = (opt_params["SNAP_thetas"] + np.pi) % (2 * np.pi) - np.pi
        processed_params["BS_phis"] = (opt_params["BS_phis"] + np.pi) % (2 * np.pi) - np.pi
        processed_params["BS_thetas"] = (opt_params["BS_thetas"] + np.pi) % (
            2 * np.pi
        ) - np.pi

        return processed_params
