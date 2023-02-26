TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
END_OPT_STRING = "\n" + "=" * 60 + "\n"
import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # suppress warnings
import h5py

import bingo.optimizer.tf_quantum as tfq
from bingo.optimizer.GateSynthesizer import GateSynthesizer
from bingo.gate_sets.gate_set import GateSet
import qutip as qt
import datetime
import time
from .operators import HamiltonianEvolutionOperator, HamiltonianEvolutionOperatorInPlace
from typing import List, Dict
import numpy as np


class GRAPE(GateSet, GateSynthesizer):
    def __init__(
        self,
        H_static,
        H_control: List,
        DAC_delta_t=2,
        bandwidth=0.1, # this number is the bandwidth of the pulse as a fraction of half the sampling frequency f_s / 2 = 1 / 2 / DAC_delta_t. it is rounded down appropriately below
        ringup=10,
        inplace=False,
        name="GRAPE_control",
        gatesynthargs=None,
        **kwargs
    ):
        GateSet.__init__(self, name)
        self.N_drives = int(len(H_control))
        assert self.N_drives % 2 == 0

        self.bandwidth = bandwidth
        if isinstance(self.bandwidth, list):
            self.N_cutoff = [int(k * gatesynthargs["N_blocks"] / 2) for k in self.bandwidth]
            assert(len(self.N_cutoff) == int(self.N_drives // 2))
        else:
            self.N_cutoff = [int(self.bandwidth * gatesynthargs["N_blocks"] / 2)] * int(self.N_drives // 2) # this is the maximum positive frequency component index we use
        
        self.H_static = tfq.qt2tf(H_static)
        self.N = self.H_static.shape[0]
        self.DAC_delta_t = DAC_delta_t
        self.ringup = ringup

        # combine all keyword arguments
        self.parameters = {
            "bandwidth" : self.bandwidth,
            "N_cutoff" : self.N_cutoff,
            "DAC_delta_t" : self.DAC_delta_t,
            "ringup" : self.ringup,
            **self.parameters,
            **kwargs,
        }  # python 3.9: self.parameters | kwargs
        
        self.H_control = []
        for k in H_control:
            self.H_control.append(tfq.qt2tf(k))

        if inplace:
            self.U = HamiltonianEvolutionOperatorInPlace(
                N=self.H_static.shape[-1], H_static=self.H_static, delta_t=DAC_delta_t
            )
        else:
            self.U = HamiltonianEvolutionOperator(
                N=self.H_static.shape[-1], H_static=self.H_static, delta_t=DAC_delta_t
            )
        
        if gatesynthargs:
            GateSynthesizer.__init__(self, gateset=self, **gatesynthargs)
        else:
            raise ValueError("GateSynthesizer settings must be specified at this time for GRAPE with the kwarg 'gatesynthargs'")


    @property
    def parameter_names(self):

        params = []
        for k in range(int(self.N_drives // 2)):
            params.append("I_DC" + str(k))
            params.append("I_real" + str(k))
            params.append("I_imag" + str(k))
            params.append("Q_DC" + str(k))
            params.append("Q_real" + str(k))
            params.append("Q_imag" + str(k))

        return params

    def randomization_ranges(self):
        ranges = {}
        scales = []
        if isinstance(self.parameters["scale"], list):
            scales = self.parameters["scale"]
        else:
            scales = [self.parameters["scale"]] * int(self.N_drives // 2)

        for k in range(int(self.N_drives // 2)):
            ranges["I_DC" + str(k)] = (
                -1 * scales[k],
                1 * scales[k],
            )
            ranges["I_real" + str(k)] = (
                -1 * scales[k],
                1 * scales[k],
            )
            ranges["I_imag" + str(k)] = (
                -1 * scales[k],
                1 * scales[k],
            )
            ranges["Q_DC" + str(k)] = (
                -1 * scales[k],
                1 * scales[k],
            )
            ranges["Q_real" + str(k)] = (
                -1 * scales[k],
                1 * scales[k],
            )
            ranges["Q_imag" + str(k)] = (
                -1 * scales[k],
                1 * scales[k],
            )

        return ranges

    def randomize_and_set_vars(self):
        """
        This function creates the tf variables over which we will optimize and randomizes their initial values.
        We redefine it here to allow for a separation between the number of blocks (time steps/pulse length) and optimization variables.


        Returns
        -----------
        dict of tf.Variable (no tf.constants are allowed) of dimension (N_blocks, parallel) with initialized values.
        Randomization ranges are pulled from those specified in gateset.randomization_ranges().
        Note that the variables in this dict that will be optimized must have ``trainable=True``
        """

        init_vars = {}

        init_scales = self.gateset.randomization_ranges()
        init_vars = {}
        for var_name in self.gateset.parameter_names:
            scale = init_scales[var_name]
            if "DC" in var_name:
                var_np = np.random.uniform(
                    scale[0],
                    scale[1],
                    size=(1, self.parameters["N_multistart"]),
                )
            elif "imag" in var_name or "real" in var_name:
                var_np = np.random.uniform(
                    scale[0],
                    scale[1],
                    size=(self.N_cutoff[int(var_name[-1])], self.parameters["N_multistart"]),
                )
            else:
                raise ValueError("Invalid optimization parameter present.")
            var_tf = tf.Variable(
                var_np, dtype=tf.float32, trainable=True, name=var_name
            )
            init_vars[var_name] = var_tf
        return init_vars

    def create_optimization_mask(self):
        """
        Returns
        -----------
        Dict of integer arrays with as many items as the dict returned by ``randomize_and_set_vars``.
        This mask is used to exclude some parameters from the gradient calculation. An entry of "1" indicates that the parameter
        will be optimized. An entry of "0" excludes it from the optimization. The shape of each array should be (N_blocks, # parallel circuits).
        """
        masks = {}
        if self.parameters["optimization_masks"] is None: # if no masks are provided, make a dictionary of none masks
            self.parameters["optimization_masks"] = {
                var_name: None for var_name in self.gateset.parameter_names
            }
        for var_name in self.gateset.parameter_names: # construct default masks for the "none" entries and correctly tile any provided masks to account for N_multistart
            if "DC" in var_name:
                var_mask = tf.ones(
                    shape=(
                        1,
                        self.parameters["N_multistart"],
                    ),
                    dtype=tf.float32,
                )
            elif "imag" in var_name or "real" in var_name:
                var_mask = tf.ones(
                    shape=(
                        self.N_cutoff[int(var_name[-1])],
                        self.parameters["N_multistart"],
                    ),
                    dtype=tf.float32,
                )
            else:
                raise ValueError("Invalid optimization parameter present.")

            masks[var_name] = var_mask


        return masks


    @tf.function
    def get_IQ_time_series(self, I_DC : tf.Variable, I_real : tf.Variable, I_imag : tf.Variable, Q_DC : tf.Variable, Q_real : tf.Variable, Q_imag : tf.Variable, N_cutoff):
        I_comps = tf.dtypes.complex(I_real, I_imag)
        Q_comps = tf.dtypes.complex(Q_real, Q_imag)

        DC_comps = tf.cast(I_DC, dtype=tf.complex64) + 1j * tf.cast(Q_DC, dtype=tf.complex64)
        positive_comps = I_comps + 1j * Q_comps
        negative_comps = tf.math.conj(tf.reverse(I_comps, axis=[0])) + 1j * tf.math.conj(tf.reverse(Q_comps, axis=[0]))

        freq_comps = tf.concat([DC_comps, positive_comps, 
                                 tf.zeros((self.parameters["N_blocks"] - 1 - 2 * N_cutoff, self.parameters["N_multistart"]), dtype=tf.complex64), negative_comps], axis=0)

        ringup_env = tf.cast((1 - tf.math.cos(tf.linspace(0.0, np.pi, self.ringup))) / 2, dtype=tf.complex64) # factor of 2 already in self.N_cutoff definition
        envelope = tf.concat([ringup_env, tf.ones([self.parameters["N_blocks"] - 2 * self.ringup], dtype=tf.complex64), tf.reverse(ringup_env, axis=[0])], axis=0)

        signal =  tf.transpose(tf.signal.ifft(tf.transpose(freq_comps))) # ifft the sequence of frequency components, this produces the Fourier series with components I_comps, Q_comps
        return tf.einsum("k,k...->k...", envelope, signal) # multiple the signal by the envelope

    @tf.function
    def batch_construct_block_operators(self, opt_vars : Dict[str, tf.Variable]):

        control_signal = self.get_IQ_time_series(opt_vars["I_DC" + str(0)], opt_vars["I_real" + str(0)], 
                                                    opt_vars["I_imag" + str(0)], opt_vars["Q_DC" + str(0)], 
                                                    opt_vars["Q_real" + str(0)], opt_vars["Q_imag" + str(0)],
                                                    self.parameters["N_cutoff"][0])
        control_Is = tf.cast(tf.math.real(control_signal), dtype=tf.complex64)
        control_Qs = tf.cast(tf.math.imag(control_signal), dtype=tf.complex64)
        
        # n_block, n_batch, Hilbert dimension, Hilbert dimension
        H_cs = tf.einsum("ab,cd->abcd", control_Is, self.H_control[0]) + tf.einsum(
            "ab,cd->abcd", control_Qs, self.H_control[1]
        )
        for k in range(1, self.N_drives // 2):
            control_signal = self.get_IQ_time_series(opt_vars["I_DC" + str(k)], opt_vars["I_real" + str(k)], 
                                                    opt_vars["I_imag" + str(k)], opt_vars["Q_DC" + str(k)], 
                                                    opt_vars["Q_real" + str(k)], opt_vars["Q_imag" + str(k)],
                                                    self.parameters["N_cutoff"][k])
            control_Is = tf.cast(tf.math.real(control_signal), dtype=tf.complex64)
            control_Qs = tf.cast(tf.math.imag(control_signal), dtype=tf.complex64)
            
            # n_block, n_batch, Hilbert dimension, Hilbert dimension
            H_cs += tf.einsum("ab,cd->abcd", control_Is, self.H_control[2 * k]) + tf.einsum(
                "ab,cd->abcd", control_Qs, self.H_control[2 * k + 1]
            )

        blocks = self.U(H_cs)

        return blocks

    # @tf.function
    def preprocess_params_before_saving(self, opt_params : Dict[str, tf.Variable], *args):
        processed_params = {}

        for k in range(self.N_drives // 2):
            control_signal = self.get_IQ_time_series(opt_params["I_DC" + str(k)], opt_params["I_real" + str(k)], 
                                                    opt_params["I_imag" + str(k)], opt_params["Q_DC" + str(k)], 
                                                    opt_params["Q_real" + str(k)], opt_params["Q_imag" + str(k)],
                                                    self.parameters["N_cutoff"][k])
            processed_params["I" + str(k)] = tf.math.real(control_signal)
            processed_params["Q" + str(k)] = tf.math.imag(control_signal)

        return processed_params
