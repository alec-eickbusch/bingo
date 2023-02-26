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


class GRAPETimeDomain(GateSet, GateSynthesizer):
    def __init__(
        self,
        H_static,
        H_control: List,
        control_delta_t=10,
        simulation_interpolation_ratio = 5,
        inplace=False,
        name="GRAPE_time_domain_control",
        bandwidth_MHz = 50.0,
        pulse_len_ns = 1000,
        ringup_sigma_ns = 20,
        **kwargs
    ):
        GateSet.__init__(self, name)
        # combine all keyword arguments
        self.parameters = {
            **self.parameters,
            **kwargs,
        }  # python 3.9: self.parameters | kwargs
        self.H_static = tfq.qt2tf(H_static)
        self.N = self.H_static.shape[0]
        self.control_delta_t = control_delta_t
        self.num_control_pts = int(pulse_len_ns/control_delta_t)
        self.pulse_len_ns = self.num_control_pts*self.control_delta_t
        self.simulation_interpolation_ratio = int(simulation_interpolation_ratio)
        self.simulation_delta_t = self.control_delta_t/self.simulation_interpolation_ratio
        self.num_simulation_pts = self.num_control_pts*self.simulation_interpolation_ratio
        self.ts_controls = np.linspace(0,self.pulse_len_ns, self.num_control_pts)
        self.ts_simulation = np.linspace(0,self.pulse_len_ns, self.num_simulation_pts)
        self.fft_pad_length = int((self.num_simulation_pts - self.num_control_pts)/2.0)
        self.paddings = tf.constant([[self.fft_pad_length,self.fft_pad_length],[0,0]])
        self.ringup_sigma_ns = ringup_sigma_ns

        ringup_len = int(self.ringup_sigma_ns/self.simulation_delta_t)
        ringup_env = tf.cast((1 - tf.math.cos(tf.linspace(0.0, np.pi, ringup_len))) / 2, dtype=tf.complex64) # factor of 2 already in self.N_cutoff definition
        self.envelope = tf.concat([ringup_env, tf.ones([self.num_simulation_pts - 2 * ringup_len], dtype=tf.complex64), tf.reverse(ringup_env, axis=[0])], axis=0)

        print("control_delta_t: %.3f num_control_pts: %d pulse_len_ns: %.3f\n \
             simulation_interpolation_ratio: %d simulation_delta_t: %.3f num_simulation_pts: %d\n \
             fft_pad_length:%d ringup_sigma_ns: %.3f ringup_len: %d" % (self.control_delta_t, self.num_control_pts, self.pulse_len_ns,\
                self.simulation_interpolation_ratio, self.simulation_delta_t, self.num_simulation_pts,
                self.fft_pad_length, ringup_sigma_ns, ringup_len))

        self.N_drives = int(len(H_control))
        assert self.N_drives % 2 == 0

        self.H_control = []
        for k in H_control:
            self.H_control.append(tfq.qt2tf(k))

        if inplace:
            self.U = HamiltonianEvolutionOperatorInPlace(
                N=self.H_static.shape[-1], H_static=self.H_static, delta_t=self.simulation_delta_t
            )
        else:
            self.U = HamiltonianEvolutionOperator(
                N=self.H_static.shape[-1], H_static=self.H_static, delta_t=self.simulation_delta_t
            )
        self.bandwidth_MHz = bandwidth_MHz
        self.construct_filter()
        

    def construct_filter(self):
        """
        Should return a tensor that is multipled by the signal fft (with shifted fft.)
        the filter is applied BEFORE the interpolation. so fft based on control delta t, not simulation delta t
        """
        freqs_MHz = np.fft.fftfreq(self.num_control_pts, self.control_delta_t) * 1000
        freqs_MHz_shift = np.fft.fftshift(freqs_MHz) #also converts it into a tensor
        # Make a brick wall filter. Note, could use a more sophisticated filter function later.
        idx_L = np.argmin(np.abs(freqs_MHz_shift + self.bandwidth_MHz))
        idx_R = np.argmin(np.abs(freqs_MHz_shift - self.bandwidth_MHz))
        filter = np.zeros_like(freqs_MHz_shift)
        filter[idx_L:idx_R + 1] = 1
        #filter = np.fft.ifftshift(filter)
        self.filter =  tf.expand_dims(tf.constant(filter, dtype=tf.complex64), axis=1)
    
    @property
    def parameter_names(self):

        params = []
        for k in range(int(self.N_drives // 2)):
            params.append("I" + str(k))
            params.append("Q" + str(k))

        return params

    def randomization_ranges(self):
        ranges = {}
        for k in range(int(self.N_drives // 2)):
            ranges["I" + str(k)] = (
                -1 * self.parameters["scale"],
                1 * self.parameters["scale"],
            )
            ranges["Q" + str(k)] = (
                -1 * self.parameters["scale"],
                1 * self.parameters["scale"],
            )

        return ranges

    @tf.function
    def process_signal(self, controls_I, controls_Q):
        signal = controls_I + 1j*controls_Q
        sig_fft = tf.transpose(tf.signal.fftshift(tf.signal.fft(tf.transpose(signal))))
        sig_fft_filtered =  sig_fft*self.filter
        sig_fft_filtered_padded = tf.pad(sig_fft_filtered, self.paddings)
        signal_filtered_interpolated = tf.transpose(tf.signal.ifft(tf.signal.ifftshift(tf.transpose(sig_fft_filtered_padded))))
        signal_processed =  tf.einsum("k,k...->k...", self.envelope, signal_filtered_interpolated) # multiple the signal by the envelope
        return tf.cast(tf.math.real(signal_processed), dtype=tf.complex64), tf.cast(tf.math.imag(signal_processed), dtype=tf.complex64)

    @tf.function
    def batch_construct_block_operators(self, opt_vars : Dict[str, tf.Variable]):
        
        control_Is = tf.cast(opt_vars['I0'], dtype=tf.complex64)
        control_Qs = tf.cast(opt_vars['Q0'], dtype=tf.complex64)

        #filter signal
        control_Is, control_Qs = self.process_signal(control_Is, control_Qs)

        # n_block, n_batch, Hilbert dimension, Hilbert dimension
        H_cs = tf.einsum("ab,cd->abcd", control_Is, self.H_control[0]) + tf.einsum(
            "ab,cd->abcd", control_Qs, self.H_control[1]
        )
        for k in range(1, self.N_drives // 2):
            control_Is = tf.cast(opt_vars['I%d' % k], dtype=tf.complex64)
            control_Qs = tf.cast(opt_vars['Q%d' % k], dtype=tf.complex64)

            #filter signal
            control_Is, control_Qs = self.process_signal(control_Is, control_Qs)
            
            # n_block, n_batch, Hilbert dimension, Hilbert dimension
            H_cs += tf.einsum("ab,cd->abcd", control_Is, self.H_control[2 * k]) + tf.einsum(
                "ab,cd->abcd", control_Qs, self.H_control[2 * k + 1]
            )

        blocks = self.U(H_cs)

        return blocks

    @tf.function
    def preprocess_params_before_saving(self, opt_params : Dict[str, tf.Variable], *args):
        processed_params = {}

        for k in range(self.N_drives // 2):
            control_Is = tf.cast(opt_params['I%d' % k], dtype=tf.complex64)
            control_Qs = tf.cast(opt_params['Q%d' % k], dtype=tf.complex64)
            #filter signal
            control_Is, control_Qs = self.process_signal(control_Is, control_Qs)
            processed_params["I" + str(k)] = control_Is
            processed_params["Q" + str(k)] = control_Qs

        return processed_params