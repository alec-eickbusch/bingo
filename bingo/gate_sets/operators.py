from math import pi, sqrt
import tensorflow as tf
from tensorflow import complex64 as c64
from .utils import tensor
import numpy as np

from distutils.version import LooseVersion

import bingo.optimizer.tf_quantum as tfq

if LooseVersion(tf.__version__) >= "2.2":
    diag = tf.linalg.diag
else:
    import numpy as np

    diag = np.diag  # k=1 option is broken in tf.linalg.diag in TF 2.1 (#35761)


### Constant operators


def sigma_x():
    return tf.constant([[0.0, 1.0], [1.0, 0.0]], dtype=c64)


def sigma_y():
    return tf.constant([[0.0j, -1.0j], [1.0j, 0.0j]], dtype=c64)


def sigma_z():
    return tf.constant([[1.0, 0.0], [0.0, -1.0]], dtype=c64)


def sigma_m():
    return tf.constant([[0.0, 1.0], [0.0, 0.0]], dtype=c64)


def hadamard():
    return 1 / sqrt(2) * tf.constant([[1.0, 1.0], [1.0, -1.0]], dtype=c64)


def identity(N):
    """Returns an identity operator in the Fock basis.
    Args:
        N (int): Dimension of Hilbert space
    Returns:
        Tensor([N, N], tf.complex64): NxN identity operator
    """
    return tf.eye(N, dtype=c64)


def destroy(N):
    """Returns a destruction (lowering) operator in the Fock basis.
    Args:
        N (int): Dimension of Hilbert space
    Returns:
        Tensor([N, N], tf.complex64): NxN creation operator
    """
    a = diag(tf.sqrt(tf.range(1, N, dtype=tf.float32)), k=1)
    return tf.cast(a, dtype=c64)


def create(N):
    """Returns a creation (raising) operator in the Fock basis.
    Args:
        N (int): Dimension of Hilbert space
    Returns:
        Tensor([N, N], tf.complex64): NxN creation operator
    """
    return tf.linalg.adjoint(destroy(N))


def num(N):
    """Returns the number operator in the Fock basis.
    Args:
        N (int): Dimension of Hilbert space
    Returns:
        Tensor([N, N], tf.complex64): NxN number operator
    """
    return tf.cast(diag(tf.range(0, N)), dtype=c64)


def position(N):
    """Returns the position operator in the Fock basis.
    Args:
        N (int): Dimension of Hilbert space
    Returns:
        Tensor([N, N], tf.complex64): NxN position operator
    """
    # Preserve max precision in intermediate calculations until final cast
    sqrt2 = tf.sqrt(tf.constant(2, dtype=c64))
    a_dag = create(N)
    a = destroy(N)
    return tf.cast((a_dag + a) / sqrt2, dtype=c64)


def momentum(N):
    """Returns the momentum operator in the Fock basis.
    Args:
        N (int): Dimension of Hilbert space
    Returns:
        Tensor([N, N], tf.complex64): NxN momentum operator
    """
    # Preserve max precision in intermediate calculations until final cast
    sqrt2 = tf.sqrt(tf.constant(2, dtype=c64))
    a_dag = create(N)
    a = destroy(N)
    return tf.cast(1j * (a_dag - a) / sqrt2, dtype=c64)


def parity(N):
    """Returns the photon number parity operator in the Fock basis.
    Args:
        N (int): Dimension of Hilbert space
    Returns:
        Tensor([N, N], tf.complex64): NxN photon number parity operator
    """
    pm1 = tf.where(tf.math.floormod(tf.range(N), 2) == 1, -1, 1)
    return tf.cast(diag(pm1), dtype=c64)


def projector(n, N):
    """
    Returns a projector onto n-th basis state in N-dimensional Hilbert space.
    Args:
        n (int): index of basis vector
        N (int): Dimension of Hilbert space
    Returns:
        Tensor([N, N], tf.complex64): NxN photon number parity operator
    """
    assert n < N
    return tf.cast(diag(tf.one_hot(n, N)), c64)


### Parametrized operators


class ParametrizedOperator:
    def __init__(self, N, tensor_with=None):
        """
        Args:
            N (int): dimension of Hilbert space
            tensor_with (list, LinearOperator): a list of operators to compute
                tensor product. By convention, <None> should be used in place
                of this operator in the list. For example, [identity(2), None] 
                will create operator in the Hilbert space of size 2*N acting
                trivially on the first component in the tensor product.
        """
        self.N = N
        self.tensor_with = tensor_with

    @tf.function
    def __call__(self, *args, **kwargs):
        this_op = self.compute(*args, **kwargs)
        if self.tensor_with is not None:
            ops = [T if T is not None else this_op for T in self.tensor_with]
            return tensor(ops)
        else:
            return this_op

    def compute(self):
        """ To be implemented by the subclass. """
        pass


class HamiltonianEvolutionOperator(ParametrizedOperator):
    """ Unitary evolution according to some Hamiltonian. Note that this implemenation uses matrix
        exponentiation rather than working in a diagonal basis for speed considerations. Matrix
        diagonalization on GPUs is very, very slow. """

    def __init__(self, N, H_static, delta_t, *args, **kwargs):
        """
        Args:
            H_static: the static portion of the Hamiltonian
        """
        self.H_static = H_static
        self.delta_t = tf.cast(delta_t, c64)
        
        super().__init__(N=N, *args, **kwargs)

    @tf.function
    def compute(self, H_controls):
        """
        This function calculates a set of time propagators given piecewise-constant controls.
        Args:
            H_static: a tf.constant of dimensions NxN
            H_controls: a tf.Variable of dimensions [block, batch, row, column] with shape (block, batch, N, N). Note that batch can have multiple dimensions
        Returns:
            a tf.Variable of dimensions [block, batch, row, column] with shape (block, batch, N, N) where each [i, :, :] is a piecewise-constant time propagator
        """
        U_control = tf.linalg.expm(-1j * self.delta_t * (H_controls + self.H_static))
        return U_control

class HamiltonianEvolutionOperatorInPlace(ParametrizedOperator):
    """ Unitary evolution according to some Hamiltonian, but requires less memory at the cost of running slower."""

    def __init__(self, N, H_static, delta_t, *args, **kwargs):
        """
        Args:
            H_static: the static portion of the Hamiltonian
        """
        self.H_static = H_static
        self.delta_t = tf.cast(delta_t, c64)
        
        super().__init__(N=N, *args, **kwargs)

    @tf.function
    def compute(self, H_controls):
        """
        This function calculates a set of time propagators given piecewise-constant controls by calculating a propagator for each circuit and multiplying it
        into the previous propagator.
        Args:
            H_static: a tf.constant of dimensions NxN
            H_controls: a tf.Variable of dimensions [block, batch, row, column] with shape (block, batch, N, N). Note that batch can have multiple dimensions
        Returns:
            a tf.Variable of dimensions [batch, row, column] with shape (batch, N, N) where each [i, :, :] is a piecewise-constant time propagator
        """
        U = tf.eye(self.N, batch_shape=[H_controls.shape[1]], dtype=c64)
        for k in tf.range(H_controls.shape[0]):
            U = tf.linalg.expm(-1j * self.delta_t * (self.H_static + H_controls[k, :, :, :])) @ U
        return tf.reshape(U, (1, H_controls.shape[1], self.N, self.N))


class TranslationOperator(ParametrizedOperator):
    """ 
    Translation in phase space.
    
    Example:
        T = TranslationOperator(100)
        alpha = tf.constant([1.23+0.j, 3.56j, 2.12+1.2j])
        T(alpha) # shape=[3,100,100]
    """

    def __init__(self, N, *args, **kwargs):
        """ Pre-diagonalize position and momentum operators."""
        p = momentum(N)
        q = position(N)

        # Pre-diagonalize
        (self._eig_q, self._U_q) = tf.linalg.eigh(q)
        (self._eig_p, self._U_p) = tf.linalg.eigh(p)
        self._qp_comm = tf.linalg.diag_part(q @ p - p @ q)
        #self._qp_comm = tf.linalg.diag_part(tf.constant(1j, dtype=tf.complex64)*tf.eye(N, dtype=tf.complex64))
        super().__init__(N=N, *args, **kwargs)

    @tf.function
    def compute(self, amplitude):
        """Calculates T(amplitude) for a batch of amplitudes using BCH.
        Args:
            amplitude (Tensor([B1, ..., Bb], c64)): A batch of amplitudes
        Returns:
            Tensor([B1, ..., Bb, N, N], c64): A batch of T(amplitude)
        """
        # Reshape amplitude for broadcast against diagonals
        amplitude = tf.cast(tf.expand_dims(amplitude, -1), dtype=c64)

        # Take real/imag of amplitude for the commutator part of the expansion
        re_a = tf.cast(tf.math.real(amplitude), dtype=c64)
        im_a = tf.cast(tf.math.imag(amplitude), dtype=c64)

        # Exponentiate diagonal matrices
        expm_q = tf.linalg.diag(tf.math.exp(1j * im_a * self._eig_q))
        expm_p = tf.linalg.diag(tf.math.exp(-1j * re_a * self._eig_p))
        expm_c = tf.linalg.diag(tf.math.exp(-0.5 * re_a * im_a * self._qp_comm))

        # Apply Baker-Campbell-Hausdorff
        return tf.cast(
            self._U_q
            @ expm_q
            @ tf.linalg.adjoint(self._U_q)
            @ self._U_p
            @ expm_p
            @ tf.linalg.adjoint(self._U_p)
            @ expm_c,
            dtype=c64,
        )


class DisplacementOperator(TranslationOperator):
    """ 
    Displacement in phase space D(amplitude) = T(amplitude * sqrt(2)).
    
    """

    def __call__(self, amplitude):
        return super().__call__(amplitude * sqrt(2))


class RotationOperator(ParametrizedOperator):
    """ Rotation in phase space."""

    @tf.function
    def compute(self, phase):
        """Calculates R(phase) = e^{i*phase*n} for a batch of phases.
        Args:
            phase (Tensor([B1, ..., Bb], c64)): A batch of phases
        Returns:
            Tensor([B1, ..., Bb, N, N], c64): A batch of R(phase)
        """
        phase = tf.squeeze(phase)
        phase = tf.cast(tf.expand_dims(phase, -1), dtype=c64)
        exp_diag = tf.math.exp(1j * phase * tf.cast(tf.range(self.N), c64))
        return tf.linalg.diag(exp_diag)


class QubitRotationXY(ParametrizedOperator):
    """
    Qubit rotation in xy plane.
    R(angle, phase) = e^(-i*angle/2*[cos(phase)*sx + sin(phase*sy]))
    
    """

    def __init__(self, *args, **kwargs):
        super().__init__(N=2, *args, **kwargs)

    @tf.function
    def compute(self, angle, phase):
        """Calculates rotation matrix for a batch of rotation angles.
        Args:
            angle (Tensor([B1, ..., Bb], float32)): batched angle of rotation
                in radians, i.e. angle=pi corresponds to full qubit flip.
            phase (Tensor([B1, ..., Bb], float32)): batched axis of rotation
                in radians, where by convention 0 is x axis.
        Returns:
            Tensor([B1, ..., Bb, 2, 2], c64): A batch of R(angle, phase)
        """
        if type(angle) is float or type(angle) is int:
            angle = tf.constant(angle)
        if type(phase) is float or type(phase) is int:
            phase = tf.constant(phase)
        assert angle.shape == phase.shape
        angle = tf.cast(tf.reshape(angle, angle.shape + [1, 1]), c64)
        phase = tf.cast(tf.reshape(phase, phase.shape + [1, 1]), c64)

        sx = sigma_x()
        sy = sigma_y()
        I = identity(2)

        R = tf.math.cos(angle / 2) * I - 1j * tf.math.sin(angle / 2) * (
            tf.math.cos(phase) * sx + tf.math.sin(phase) * sy
        )
        return R


class QubitRotationZ(ParametrizedOperator):
    """ Qubit rotation around z zxis. R(angle) = e^(-i*angle/2*sz)"""

    def __init__(self, *args, **kwargs):
        super().__init__(N=2, *args, **kwargs)

    def compute(self, angle):
        """Calculates rotation matrix for a batch of rotation angles.
        Args:
            angle (Tensor([B1, ..., Bb], float32)): batched angle of rotation
                in radians, i.e. angle=pi corresponds to full qubit flip.
        Returns:
            Tensor([B1, ..., Bb, 2, 2], c64): A batch of R(angle)
        """
        angle = tf.cast(tf.reshape(angle, angle.shape + [1, 1]), c64)

        sz = sigma_z()
        I = identity(2)

        R = tf.math.cos(angle / 2) * I - 1j * tf.math.sin(angle / 2) * sz
        return R


class Phase(ParametrizedOperator):
    """ Simple phase factor."""

    def __init__(self, *args, **kwargs):
        super().__init__(N=1, *args, **kwargs)

    def compute(self, angle):
        """
        Calculates batch phase factor e^(i*angle)
        Args:
            angle (Tensor([B1, ..., Bb], float32)): batch of angles in radians
            
        Returns:
            Tensor([B1, ..., Bb, 1, 1], c64): A batch of phase factors
        """
        angle = tf.squeeze(angle)  # TODO: get rid of this
        angle = tf.cast(tf.reshape(angle, angle.shape + [1, 1]), c64)
        return tf.math.exp(1j * angle)


class SNAPOperator(ParametrizedOperator):
    """
    Selective Number-dependent Arbitrary Phase (SNAP) gate.
    SNAP(theta) = sum_n( e^(i*theta_n) * |n><n| )
    
    """

    def __init__(self, N, phase_offset=None, *args, **kwargs):
        """
        Args:
            N (int): dimension of Hilbert space    
            phase_offset (Tensor([N], c64)): static offset added to the rota-
                tion phases to model miscalibrated gate.             
        """
        self.phase_offset = 0 if phase_offset is None else phase_offset
        super().__init__(N=N, *args, **kwargs)

    @tf.function
    def compute(self, theta):
        """Calculates ideal SNAP(theta) for a batch of SNAP parameters.
        Args:
            theta (Tensor([block index, ..., batch, SNAP phase], c64)): A batch of parameters.
            Note that the length of the SNAP phase axis does NOT need to be equal to N.
            One only need specify the first n non-zero SNAP phases. This function will
            pad the phase axis with N-n zeros to make the NxN unitary
        Returns:
            Tensor([block index, batch, N, N], c64): A batch of SNAP(theta)
        """
        S = theta.shape[-1]  # number of SNAP phases provided
        D = len(theta.shape) - 1 # remaining dimension of the thetas array ignoring the phase index
        paddings = tf.convert_to_tensor([[0, 0]] * D + [[0, self.N - S]])
        theta = tf.cast(theta, dtype=c64)
        theta = tf.pad(theta, paddings)
        theta -= self.phase_offset
        exp_diag = tf.math.exp(1j * theta)
        return tf.linalg.diag(exp_diag)

class SqueezeOperator(ParametrizedOperator):
    """
        Args:
            N (int): dimension of Hilbert space                 
    """
    def __init__(self, N, *args, **kwargs):
        self.a2 = destroy(N) @ destroy(N)
        self.a2_dag = create(N) @ create(N)

        super().__init__(N=N, *args, **kwargs)

    @tf.function
    def compute(self, zs):
        return tf.linalg.expm(1 / 2 * (tf.math.conj(zs)[..., None, None] * self.a2[None, :, :] - zs[..., None, None] * self.a2_dag[None, :, :]))

class BeamSplitterOperator(ParametrizedOperator):
    
    """
        Args:
            N (int): dimension of Hilbert space                 
    """
    def __init__(self, N, *args, **kwargs):
        self.a = tfq.two_mode_op(destroy(N),0)
        self.b = tfq.two_mode_op(destroy(N),1)
        self.a_dag = tfq.two_mode_op(create(N),0)
        self.b_dag = tfq.two_mode_op(create(N),1)

        super().__init__(N=N, *args, **kwargs)

    @tf.function
    def compute(self, phis, thetas):
        """Calculates ideal beam splitter(theta, phi) for a batch of parameters.
        Args:
            phis (Tensor([B1, ..., Bb], float32)): A batch of phis.
            thetas (Tensor([B1, ..., Bb], float32)): A batch of thetas.
        Returns:
            Tensor([B1, ..., Bb, N, N], c64): A batch of BS(phi, theta)
        """
        
        phase = tf.math.exp(1j*tf.cast(phis, dtype=tf.complex64)[...,None,None])
        phase_dag = tf.math.conj(phase)
        exp = -1j*tf.cast(thetas, dtype=tf.complex64)[...,None,None]*(phase*(self.a_dag@self.b)[None,:,:] + phase_dag*(self.a@self.b_dag)[None,:,:])
        return tf.linalg.expm(exp)

class TwoModeSqueezeOperator(ParametrizedOperator):
    
    """
        Args:
            N (int): dimension of Hilbert space                 
    """
    def __init__(self, N, *args, **kwargs):
        self.a = tfq.two_mode_op(destroy(N),0)
        self.b = tfq.two_mode_op(destroy(N),1)
        self.a_dag = tfq.two_mode_op(create(N),0)
        self.b_dag = tfq.two_mode_op(create(N),1)

        super().__init__(N=N, *args, **kwargs)

    @tf.function
    def compute(self, phis, zs):
        """Calculates ideal two mode squeezer(theta, phi) for a batch of parameters.
        Args:
            phis (Tensor([B1, ..., Bb], float32)): A batch of phis.
            zs (Tensor([B1, ..., Bb], float32)): A batch of zs.
        Returns:
            Tensor([B1, ..., Bb, N, N], c64): A batch of BS(phi, theta)
        """
        
        phase = tf.math.exp(1j*tf.cast(phis, dtype=tf.complex64)[...,None,None])
        phase_dag = tf.math.conj(phase)
        exp = -1j*tf.cast(zs, dtype=tf.complex64)[...,None,None]*(phase*(self.a_dag@self.b_dag)[None,:,:] + phase_dag*(self.a@self.b)[None,:,:])
        return tf.linalg.expm(exp)

class SingleModeSqueezeOperator(ParametrizedOperator):
    
    """
        Args:
            N (int): dimension of Hilbert space                 
    """
    def __init__(self, N, *args, **kwargs):
        self.a = destroy(N)
        self.a_dag = create(N)

        super().__init__(N=N, *args, **kwargs)

    @tf.function
    def compute(self, phis, zs):
        """Calculates ideal two mode squeezer(theta, phi) for a batch of parameters.
        Args:
            phis (Tensor([B1, ..., Bb], float32)): A batch of phis.
            zs (Tensor([B1, ..., Bb], float32)): A batch of zs (squeezing strengths)
        Returns:
            Tensor([B1, ..., Bb, N, N], c64): A batch of BS(phi, theta)
        """
        
        phase = tf.math.exp(1j*tf.cast(phis, dtype=tf.complex64)[...,None,None])
        phase_dag = tf.math.conj(phase)
        exp = (-1j/2.0)*tf.cast(zs, dtype=tf.complex64)[...,None,None]*(phase*(self.a_dag@self.a_dag)[None,:,:] + phase_dag*(self.a@self.a)[None,:,:])
        return tf.linalg.expm(exp)

#selective qubit rotation operator
# This is the generalization of SNAP.
#defined as: U = \Sum_n R(phi_n, theta_n) * |n><n| 
class SQROperator(ParametrizedOperator):
    """
    Selective Number-dependent Arbitrary Phase (SNAP) gate.
    SNAP(theta) = sum_n( e^(i*theta_n) * |n><n| )
    
    """

    def __init__(self, N,  *args, **kwargs):
        self.R = lambda phi, theta : tfq.two_mode_op(QubitRotationXY()(theta, phi), 0, other_dim=N)
        self.n_projectors = []
        for n in range(N):
            n_proj_np = np.zeros((N,N))
            n_proj_np[n,n] = 1
            n_projector_tf = tf.constant(n_proj_np, dtype=tf.complex64)
            self.n_projectors.append(tfq.two_mode_op(n_projector_tf, 1, other_dim = 2))
        super().__init__(N=N, *args, **kwargs)

    @tf.function
    def compute(self, phis, thetas):
        """Calculates ideal SQR(phis, thetas) for a batch of  parameters.
        Args:
            phis (Tensor([B1, ..., Bb, phis], c64)): A batch of parameters.
            thetas (Tensor([B1, ..., Bb, thetas], c64)): A batch of parameters.
        Returns:
            Tensor([B1, ..., Bb, N, N], c64): A batch of SNAP(theta)
        """
        S = thetas.shape[-1]  # number of phases provided
        D = len(thetas.shape) - 1 # remaining dimension of the thetas array ignoring the phase index
        paddings = tf.constant([[0, 0]] * D + [[0, self.N - S]])
        thetas = tf.cast(thetas, dtype=c64)
        thetas = tf.pad(thetas, paddings)
        phis = tf.cast(phis, dtype=c64)
        phis = tf.pad(phis, paddings)
        U = tf.zeros((*thetas.shape[:-1], self.N*2, self.N*2), dtype=tf.complex64)
        for n in range(self.N):
            thetas_n = tf.transpose(tf.transpose(thetas)[n])
            phis_n = tf.transpose(tf.transpose(phis)[n])
            U = U + self.R(phis_n,thetas_n) @ self.n_projectors[n]
        return U


class ConditionalRotation(ParametrizedOperator):
    """
    Conditional oscillator rotation
    Defined as U = (-i*theta*a^dag*a*sigma_z/2)
    This operator is performed by just 'waiting' under the dispersive Hamiltonian H = chi a^dag a sigma_z/2
    """

    def __init__(self, N,  *args, **kwargs):
        self.n = tfq.two_mode_op(num(N),1,other_dim=2)
        self.sz = tfq.two_mode_op(sigma_z(),0,other_dim=N)
        self.n_sz_prod = self.n @ self.sz
        super().__init__(N=N, *args, **kwargs)

    @tf.function
    def compute(self, theta):
        """Calculates ideal CR(theta) for a batch of  parameters.
        Args:
            theta (Tensor([B1, ..., Bb], f32)): A batch of parameters.
        Returns:
            Tensor([B1, ..., Bb, N, N], c64): A batch of R(theta)
        """
        U = tf.linalg.expm(-1j*tf.cast(theta, dtype=tf.complex64)[...,None,None]*self.n_sz_prod[None,...] / 2.0)
        return U

class ConditionalParity(ConditionalRotation):
    """
    Conditional parity
    Defined as U = (-i*pi*a^dag*a*sigma_z/2)
    This operator is performed by just 'waiting' under the dispersive Hamiltonian H = chi a^dag a sigma_z/2
    """

    @tf.function 
    def compute(self):
        U = tf.linalg.expm(-1j*tf.cast(theta, dtype=tf.complex64)[...,None,None]*self.n_sz_prod[None,...] / 2.0)
        return U

#This beam splitter operator assumes the tensor structure is [Ca, qubit, Cb]. It ignores the qubit.
#TODO: have this inherit from Beam Splitter and just use that class's 'compute' function.
class BeamSplitterOperator2(ParametrizedOperator):
    
    """
        Args:
            N (int): dimension of Hilbert space                 
    """
    def __init__(self, N, *args, **kwargs):
        self.a = tfq.n_mode_op(destroy(N),[-1,2,None])
        self.b = tfq.n_mode_op(destroy(N),[None,2,-1])
        self.a_dag = tfq.n_mode_op(create(N),[-1,2,None])
        self.b_dag = tfq.n_mode_op(create(N),[None,2,-1])

        super().__init__(N=N, *args, **kwargs)

    @tf.function
    def compute(self, phis, thetas):
        """Calculates ideal beam splitter(theta, phi) for a batch of parameters.
        Args:
            phis (Tensor([B1, ..., Bb], float32)): A batch of phis.
            thetas (Tensor([B1, ..., Bb], float32)): A batch of thetas.
        Returns:
            Tensor([B1, ..., Bb, N, N], c64): A batch of BS(phi, theta)
        """
        
        phase = tf.math.exp(1j*tf.cast(phis, dtype=tf.complex64)[...,None,None])
        phase_dag = tf.math.conj(phase)
        exp = -1j*tf.cast(thetas, dtype=tf.complex64)[...,None,None]*(phase*(self.a_dag@self.b)[None,:,:] + phase_dag*(self.a@self.b_dag)[None,:,:])
        return tf.linalg.expm(exp)

#selective qubit rotation operator for two modes
#hilbert space structure is qb, cav, cav.

#NOTE: THIS DOES NOT WORK, IS NOT UNITARY!
class TwoModeSQROperator(ParametrizedOperator):
    """
    Two-mode SQR.
    
    """

    def __init__(self, N,  *args, **kwargs):
        self.R = lambda phi, theta : tfq.n_mode_op(QubitRotationXY()(theta, phi), [-1,N,N])
        self.n_projectors_a = []
        self.n_projectors_b = []
        for n in range(N):
            n_proj_np = np.zeros((N,N))
            n_proj_np[n,n] = 1
            n_projector_tf = tf.constant(n_proj_np, dtype=tf.complex64)
            self.n_projectors_a.append(tfq.n_mode_op(n_projector_tf, [2,-1,N]))
            self.n_projectors_b.append(tfq.n_mode_op(n_projector_tf, [2,N,-1]))
        super().__init__(N=N, *args, **kwargs)

    @tf.function
    def compute(self, phis_a, thetas_a, phis_b, thetas_b):
        """Calculates ideal two mode SQR2(phis_a, thetas_a, phis_b, thetas_b) for a batch of parameters.
        Args:
            phis_a (Tensor([B1, ..., Bb, phis], c64)): A batch of parameters.
            thetas_a (Tensor([B1, ..., Bb, thetas], c64)): A batch of parameters.
            phis_b (Tensor([B1, ..., Bb, phis], c64)): A batch of parameters.
            thetas_b (Tensor([B1, ..., Bb, thetas], c64)): A batch of parameters.
        Returns:
            Tensor([B1, ..., Bb, N, N], c64): A batch of SQR2 operators
        """
        S = thetas_a.shape[-1]  # number of phases provided
        D = len(thetas_a.shape) - 1 # remaining dimension of the thetas array ignoring the phase index
        paddings = tf.constant([[0, 0]] * D + [[0, self.N - S]])
        thetas_a = tf.cast(thetas_a, dtype=c64)
        thetas_a = tf.pad(thetas_a, paddings)
        phis_a = tf.cast(phis_a, dtype=c64)
        phis_a = tf.pad(phis_a, paddings)
        thetas_b = tf.cast(thetas_b, dtype=c64)
        thetas_b = tf.pad(thetas_b, paddings)
        phis_b = tf.cast(phis_b, dtype=c64)
        phis_b = tf.pad(phis_b, paddings)
        U = tf.zeros((*thetas_a.shape[:-1], self.N*self.N*2, self.N*self.N*2), dtype=tf.complex64)
        for n in range(self.N):
            thetas_a_n = tf.transpose(tf.transpose(thetas_a)[n])
            phis_a_n = tf.transpose(tf.transpose(phis_a)[n])
            thetas_b_n = tf.transpose(tf.transpose(thetas_b)[n])
            phis_b_n = tf.transpose(tf.transpose(phis_b)[n])
            U = U + self.R(phis_a_n,thetas_a_n) @ self.n_projectors_a[n] + self.R(phis_b_n,thetas_b_n) @ self.n_projectors_b[n]
        return U

#thetas specified as (0,1,-1,2,-2,...)
class DeltaSNAP(ParametrizedOperator):
    """
    Two-mode delta-SNAP
    
    """

    def __init__(self, N, *args, **kwargs):
        delta_array = np.concatenate([(n, (n+1)) for n in range(N)])[:-1]
        self.delta_array = [num*(-1)**(n+1) for n,num in enumerate(delta_array)]
        #self.delta_array = (0,1,-1,2,-2,...)
        self.delta_projectors = []
        for delta in self.delta_array:
            delta_proj_np = np.zeros((N*N,N*N))
            for n in range(N):
                if n + delta >= 0 and n+delta <N:
                    a_part = np.zeros((N,N), dtype = np.float32)
                    a_part[n,n] = 1.0
                    b_part = np.zeros((N,N), dtype = np.float32)
                    b_part[n+delta, n+delta]=1.0
                    delta_proj_np += np.kron(a_part, b_part)
            delta_projector_tf = tf.constant(delta_proj_np, dtype=tf.complex64)
            self.delta_projectors.append(delta_projector_tf)
        super().__init__(N=N, *args, **kwargs)

    @tf.function
    def compute(self, thetas):
        """Calculates ideal two mode delta_SNAP(thetas) for a batch of parameters.
        Args:
            thetas (Tensor([B1, ..., Bb, thetas], c64)): A batch of parameters.
        Returns:
            Tensor([B1, ..., Bb, N, N], c64): A batch of SQR2 operators
        """
        S = thetas.shape[-1]  # number of phases provided
        D = len(thetas.shape) - 1 # remaining dimension of the thetas array ignoring the phase index
        paddings = tf.constant([[0, 0]] * D + [[0, (2*self.N -1)- S]]) #note the factor of 2!
        thetas = tf.cast(thetas, dtype=c64)
        thetas = tf.pad(thetas, paddings)
        U = tf.zeros((*thetas.shape[:-1], self.N*self.N, self.N*self.N), dtype=tf.complex64)
        for delta_idx in range(0,2*self.N-1):
            theta = tf.transpose(tf.transpose(thetas)[delta_idx])
            U = U + tf.math.exp(1j*theta)[...,None,None] * (self.delta_projectors[delta_idx])[None,:,:]
        return U

#sigma snap
#sigmas specified as (0,1,-1,2,-2,...)
class SigmaSNAP(ParametrizedOperator):
    """
    Two-mode sigma-SNAP
    
    """

    def __init__(self, N, *args, **kwargs):
        self.sigma_projectors = []
        sigma_array = np.concatenate([(n, (n+1)) for n in range(N)])[:-1]
        self.sigma_array = [num*(-1)**(n+1) for n,num in enumerate(sigma_array)]
        #self.sigma_array = (0,1,-1,2,-2,...)
        for sigma in self.sigma_array:
            sigma_proj_np = np.zeros((N*N,N*N))
            for n in range(N):
                if n - sigma >= 0 and n-sigma <N:
                    a_part = np.zeros((N,N), dtype = np.float32)
                    a_part[n,n] = 1.0
                    b_part = np.zeros((N,N), dtype = np.float32)
                    b_part[n-sigma, n-sigma]=1.0
                    sigma_proj_np += np.kron(a_part, b_part)
            sigma_projector_tf = tf.constant(sigma_proj_np, dtype=tf.complex64)
            self.sigma_projectors.append(sigma_projector_tf)
        super().__init__(N=N, *args, **kwargs)

    @tf.function
    def compute(self, thetas):
        """Calculates ideal two mode sigma_SNAP(thetas) for a batch of parameters.
        Args:
            thetas (Tensor([B1, ..., Bb, thetas], c64)): A batch of parameters.
        Returns:
            Tensor([B1, ..., Bb, N, N], c64): A batch of SQR2 operators
        """
        S = thetas.shape[-1]  # number of phases provided
        D = len(thetas.shape) - 1 # remaining dimension of the thetas array ignoring the phase index
        paddings = tf.constant([[0, 0]] * D + [[0, (2*self.N -1)- S]]) #note the factor of 2!
        thetas = tf.cast(thetas, dtype=c64)
        thetas = tf.pad(thetas, paddings)
        U = tf.zeros((*thetas.shape[:-1], self.N*self.N, self.N*self.N), dtype=tf.complex64)
        for sigma_idx in range(0,2*self.N-1):
            theta = tf.transpose(tf.transpose(thetas)[sigma_idx])
            U = U + tf.math.exp(1j*theta)[...,None,None] * (self.sigma_projectors[sigma_idx])[None,:,:]
        return U

#thetas specified as (0,1,-1,2,-2,...)
class DeltaSQR(ParametrizedOperator):
    """
    Two-mode delta-SQR
    
    """

    def __init__(self, N, *args, **kwargs):
        delta_array = np.concatenate([(n, (n+1)) for n in range(N)])[:-1]
        self.delta_array = [num*(-1)**(n+1) for n,num in enumerate(delta_array)]
        #self.delta_array = (0,1,-1,2,-2,...)
        self.delta_projectors = []
        for delta in self.delta_array:
            delta_proj_np = np.zeros((N*N,N*N))
            for n in range(N):
                if n + delta >= 0 and n+delta <N:
                    a_part = np.zeros((N,N), dtype = np.float32)
                    a_part[n,n] = 1.0
                    b_part = np.zeros((N,N), dtype = np.float32)
                    b_part[n+delta, n+delta]=1.0
                    delta_proj_np += np.kron(a_part, b_part)
            delta_projector_tf = tfq.n_mode_op(tf.constant(delta_proj_np, dtype=tf.complex64), [2,-1])
            self.delta_projectors.append(delta_projector_tf)
        self.R = lambda phi, theta : tfq.n_mode_op(QubitRotationXY()(theta, phi), [-1,N,N])
        super().__init__(N=N, *args, **kwargs)

    @tf.function
    def compute(self, phis, thetas):
        """Calculates ideal two mode delta_SQR(phis, thetas) for a batch of parameters.
        Args:
            phis (Tensor([B1, ..., Bb, phis], c64)): A batch of parameters.
            thetas (Tensor([B1, ..., Bb, thetas], c64)): A batch of parameters.
        Returns:
            Tensor([B1, ..., Bb, N, N], c64): A batch of delta-SQR operators
        """
        S = thetas.shape[-1]  # number of phases provided
        D = len(thetas.shape) - 1 # remaining dimension of the thetas array ignoring the phase index
        paddings = tf.constant([[0, 0]] * D + [[0, (2*self.N -1)- S]]) #note the factor of 2!
        thetas = tf.cast(thetas, dtype=c64)
        thetas = tf.pad(thetas, paddings)
        U = tf.zeros((*thetas.shape[:-1], self.N*self.N, self.N*self.N), dtype=tf.complex64)
        for delta_idx in range(0,2*self.N-1):
            theta = tf.transpose(tf.transpose(thetas)[delta_idx])
            phi = tf.transpose(tf.transpose(phis)[delta_idx])
            U = U + self.R(phi, theta) @ self.delta_projectors[delta_idx]
        return U

#thetas specified as (0,1,-1,2,-2,...)
class SigmaSQR(ParametrizedOperator):
    """
    Two-mode Sigma-SQR
    
    """
    def __init__(self, N, *args, **kwargs):
        sigma_array = np.concatenate([(n, (n+1)) for n in range(N)])[:-1]
        self.sigma_array = [num*(-1)**(n+1) for n,num in enumerate(sigma_array)]
        #self.delta_array = (0,1,-1,2,-2,...)
        self.sigma_projectors = []
        for sigma in self.sigma_array:
            sigma_proj_np = np.zeros((N*N,N*N))
            for n in range(N):
                if n - sigma >= 0 and n-sigma <N:
                    a_part = np.zeros((N,N), dtype = np.float32)
                    a_part[n,n] = 1.0
                    b_part = np.zeros((N,N), dtype = np.float32)
                    b_part[n-sigma, n-sigma]=1.0
                    sigma_proj_np += np.kron(a_part, b_part)
            sigma_projector_tf = tfq.n_mode_op(tf.constant(sigma_proj_np, dtype=tf.complex64), [2,-1])
            self.sigma_projectors.append(sigma_projector_tf)
        self.R = lambda phi, theta : tfq.n_mode_op(QubitRotationXY()(theta, phi), [-1,N,N])
        super().__init__(N=N, *args, **kwargs)

    @tf.function
    def compute(self, phis, thetas):
        """Calculates ideal two mode sigma_SQR(phis, thetas) for a batch of parameters.
        Args:
            phis (Tensor([B1, ..., Bb, phis], c64)): A batch of parameters.
            thetas (Tensor([B1, ..., Bb, thetas], c64)): A batch of parameters.
        Returns:
            Tensor([B1, ..., Bb, N, N], c64): A batch of sigma-SQR operators
        """
        S = thetas.shape[-1]  # number of phases provided
        D = len(thetas.shape) - 1 # remaining dimension of the thetas array ignoring the phase index
        paddings = tf.constant([[0, 0]] * D + [[0, (2*self.N -1)- S]]) #note the factor of 2!
        thetas = tf.cast(thetas, dtype=c64)
        thetas = tf.pad(thetas, paddings)
        U = tf.zeros((*thetas.shape[:-1], self.N*self.N, self.N*self.N), dtype=tf.complex64)
        for sigma_idx in range(0,2*self.N-1):
            theta = tf.transpose(tf.transpose(thetas)[sigma_idx])
            phi = tf.transpose(tf.transpose(phis)[sigma_idx])
            U = U + self.R(phi, theta) @ self.sigma_projectors[sigma_idx]
        return U

class CD_q_sx(ParametrizedOperator):
    """
    Conditional displacement along q conditioned on sigmax
    Defined as U = (-1j*beta_q*p_op*sx/2)
    """

    def __init__(self, N,  *args, **kwargs):
        self.p = tfq.two_mode_op(momentum(N),1,other_dim=2)
        self.sx = tfq.two_mode_op(sigma_x(),0,other_dim=N)
        self.p_sx_prod = self.p @ self.sx
        super().__init__(N=N, *args, **kwargs)

    @tf.function
    def compute(self, beta_q):
        """Calculates ideal CD_q_sx(beta_q) for a batch of  parameters.
        Args:
            beta_q (Tensor([B1, ..., Bb], f32)): A batch of parameters.
        Returns:
            Tensor([B1, ..., Bb, N, N], c64): A batch of gates
        """
        U = tf.linalg.expm(-1j*tf.cast(beta_q, dtype=tf.complex64)[...,None,None]*self.p_sx_prod[None,...] / 2.0)
        return U

class CD_p_sy(ParametrizedOperator):
    """
    Conditional displacement along p conditioned on sigmay
    Defined as U = (1j*beta_p*q_op*sy/2)
    """

    def __init__(self, N,  *args, **kwargs):
        self.q = tfq.two_mode_op(position(N),1,other_dim=2)
        self.sy = tfq.two_mode_op(sigma_y(),0,other_dim=N)
        self.q_sy_prod = self.q @ self.sy
        super().__init__(N=N, *args, **kwargs)

    @tf.function
    def compute(self, beta_p):
        """Calculates ideal CD_q_sx(beta_q) for a batch of  parameters.
        Args:
            beta_q (Tensor([B1, ..., Bb], f32)): A batch of parameters.
        Returns:
            Tensor([B1, ..., Bb, N, N], c64): A batch of gates
        """
        U = tf.linalg.expm(-1j*tf.cast(beta_p, dtype=tf.complex64)[...,None,None]*self.q_sy_prod[None,...] / 2.0)
        return U