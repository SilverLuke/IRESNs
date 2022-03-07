from typing import Optional

import tensorflow as tf
import numpy as np
from tensorflow.python.types.core import TensorLike


# Split shape in a vector with proportions there are in part
# sum(vec) == shape only if sum(part) == 1 and len(vec) == len(part)
def split_units(shape, absolute_partition):
    partitions = []
    used = 0
    for div in absolute_partition:
        units = int(np.floor(shape * div))
        used += units
        partitions.append(units)
    rest = shape - used  # Rest should be less or equal to len(list)
    assert rest < len(absolute_partition)
    for i in range(rest):
        partitions[i] += 1
    assert shape == sum(partitions)
    return partitions


def get_spectral_radius(tensor, dtype):
    return tf.cast(tf.reduce_max(tf.abs(tf.linalg.eigvals(tensor))), dtype)


def generate_sub_reservoirs(shape, initializer, spectral_radius, connectivity, dtype=None):
    if connectivity == 1.:  # If full connected use this initializer because is faster than others.
        matrix = FullConnected(spectral_radius)(shape, dtype=dtype)
    elif connectivity == 0.:
        matrix = tf.zeros(shape, dtype=dtype)
    else:
        matrix = initializer(shape, dtype=dtype)
        connectivity_mask = tf.cast(tf.math.less_equal(tf.random.uniform(shape), connectivity), dtype)
        matrix = tf.math.multiply(matrix, connectivity_mask)
        if spectral_radius is not None:
            scaling = tf.math.divide_no_nan(spectral_radius, get_spectral_radius(matrix, dtype))
            matrix = tf.multiply(matrix, scaling)
    return matrix


def join_matrices(matrices):
    """
    :param matrices: is a python square Matrix (List of List) of tf.matrix
    :return: A matrix
    """
    ret = tf.concat(matrices[0], axis=1)
    for r_k in matrices[1:]:
        tmp = tf.concat(r_k, axis=1)  # axis=1 horizontal concatenation
        ret = tf.concat([ret, tmp], axis=0)
    return ret


def check_vector(vector, length, name):
    """
    :param vector: The vector to be checked or a float will be converted in a vector with len equals to length.
    :param length: The required length of the vector.
    :param name:  Name to print in errors.
    :return: a good vector.
    """
    if isinstance(vector, float):
        return [vector for _ in range(length)]
    elif isinstance(vector, list):
        if len(vector) == length:
            return vector
        else:
            raise ValueError("The list of " + name + " must have the same length of sub_reservoirs")
    else:
        raise ValueError(
            "Wrong value type for " + name + " required float or list of float. Given {}".format(type(vector)))


def check_matrix(matrix, length, name):
    """
    :param matrix: The square matrix to be checked or a float will be converted in a square matrix with len equals to length.
    :param length: The required length.
    :param name: Name to print in errors.
    :return: A good matrix.
    """
    if isinstance(matrix, float):
        return [[matrix for _ in range(length)] for _ in range(length)]
    elif isinstance(matrix, list):
        if len(matrix) != length:
            raise ValueError("The " + name + " matrix must have the same length of sub_reservoirs")
        else:
            for k in matrix:
                if len(k) != length:
                    raise ValueError("The " + name + " matrix must have the same length of sub_reservoirs")
            return matrix
    else:
        raise ValueError(
            "Wrong value type for " + name + " required float or square matrix of float. Given {}".format(type(matrix)))


def check_partition(partitions, length):
    if partitions is None:
        return [1. / length for _ in range(length)]
    elif len(partitions) == length:
        return partitions
    else:
        raise ValueError("The list of partitions must have the same length of sub_reservoirs")


class SplitBias(tf.keras.initializers.Initializer):
    """
    Bias initializer for multiple sub-reservoirs. Used in IRESN, IIRESN, IIRESNvsr.
    """
    def __init__(self, bias_scaling, sub_reservoirs, partitions=None):
        self.partitions = check_partition(partitions, sub_reservoirs)
        if bias_scaling is None:
            self.minmax = [0. for _ in range(sub_reservoirs)]
        else:
            self.minmax = check_vector(bias_scaling, sub_reservoirs, "bias scaling")

    def __call__(self, shape, dtype=None, **kwargs):
        sub_units = split_units(shape[0], self.partitions)
        sub_bias = []
        for minmax, units in zip(self.minmax, sub_units):
            init = tf.keras.initializers.RandomUniform(minval=-minmax, maxval=minmax)
            piece = init((units,), dtype=dtype)
            sub_bias.append(piece)
        # Concatenate the N sub_bias vectors in one vector
        join = tf.concat(sub_bias, axis=0)
        return join


class Kernel(tf.keras.initializers.Initializer):
    """
    Kernel initializer for ESN.
    """
    def __init__(self, initializer=tf.keras.initializers.GlorotUniform()):
        self.initializer = initializer

    def __call__(self, shape, dtype=None, **kwargs):
        return self.initializer(shape, dtype=dtype)


class SplitKernel(tf.keras.initializers.Initializer):
    """
    Kernel initializer for multiple sub-reservoirs. Used in IRESN, IIRESN, IIRESNvsr.
    """
    def __init__(self, sub_reservoirs, input_scaling, partitions=None):
        self.partitions = check_partition(partitions, sub_reservoirs)

        if input_scaling is None:
            self.initializers = [tf.keras.initializers.GlorotUniform() for _ in sub_reservoirs]
        else:
            min_max = check_vector(input_scaling, sub_reservoirs, "input scaling")
            self.initializers = [tf.keras.initializers.RandomUniform(minval=-val, maxval=val) for val in min_max]


    def __call__(self, shape, dtype=None, **kwargs):
        sub_units = split_units(shape[1], self.partitions)
        sub_kernels = []
        for init, units in zip(self.initializers, sub_units):
            sub_kernel = init((1, units), dtype=dtype)
            sub_kernels.append(tf.linalg.LinearOperatorFullMatrix(sub_kernel))

        # Build a matrix with N*N logical sub-matrices where N is the number of sub-reservoirs, and put sub-kernels in the diagonal.
        # See README.md image.
        ker = tf.linalg.LinearOperatorBlockDiag(sub_kernels).to_dense()
        return ker


class RecurrentKernel(tf.keras.initializers.Initializer):
    """
    Recurrent kernel with variable connectivity. Used in ESN model.
    """
    def __init__(self, connectivity, spectral_radius, initializer=tf.keras.initializers.GlorotUniform()):
        self.connectivity = connectivity
        self.spectral_radius = spectral_radius
        self.recurrent_initializer = initializer

    def __call__(self, shape, dtype=None, **kwargs):
        recurrent_weights = self.recurrent_initializer(shape, dtype)

        connectivity_mask = tf.cast(tf.math.less_equal(tf.random.uniform(shape), self.connectivity), dtype)
        recurrent_weights = tf.math.multiply(recurrent_weights, connectivity_mask)

        abs_eig_values = tf.abs(tf.linalg.eig(recurrent_weights)[0])
        scaling_factor = tf.math.divide_no_nan(
            self.spectral_radius, tf.reduce_max(abs_eig_values)
        )

        recurrent_weights = tf.multiply(recurrent_weights, scaling_factor)
        return recurrent_weights


class FullConnected(tf.keras.initializers.Initializer):
    """
    FullConnected initializer is equal to RecurrentKernel initializer with connectivity to 1 and initializer to tf.keras.initializers.RandomUniform.
    FullConnected is faster than RecurrentKernel. Used in ESN model.
    This initializer uses circular law to determine the values of the recurrent weight matrix
    rif. paper
    Gallicchio, Claudio, Alessio Micheli, and Luca Pedrelli.
    "Fast spectral radius initialization for recurrent neural networks."
    INNS Big Data and Deep Learning conference. Springer, Cham, 2019.
    """

    def __init__(self, spectral_radius):
        self.spectral_radius = spectral_radius

    def __call__(self, shape, dtype=None, **kwargs):
        value = (self.spectral_radius / np.sqrt(shape[0])) * (6. / np.sqrt(12))
        w = tf.random.uniform(shape, minval=-value, maxval=value, dtype=dtype)
        return w


class IRESN(tf.keras.initializers.Initializer):
    """
    Recurrent kernel initializer for IRESN.
    """
    def __init__(self, sub_reservoirs, connectivity, spectral_radius, gsr: Optional[float] = None,
                 initializer=tf.keras.initializers.GlorotUniform()):
        self.spectral_radius = check_vector(spectral_radius, sub_reservoirs, "spectral radius")
        self.connectivity = check_vector(connectivity, sub_reservoirs, "connectivity")
        self.sub_reservoirs = sub_reservoirs
        self.gsr = gsr
        self.initializer = initializer
        self.partitions = [1. / sub_reservoirs for _ in range(sub_reservoirs)]

    def __call__(self, shape, dtype=None, **kwargs):
        # self.connectivity = [tf.cast(connectivity, dtype) for connectivity in self.connectivity]
        # self.spectral_radius = [tf.cast(spectral_radius, dtype) for spectral_radius in self.spectral_radius]

        units = split_units(shape[0], self.partitions)
        recurrent_kernels = [[_ for _ in range(self.sub_reservoirs)] for _ in range(self.sub_reservoirs)]
        for i in range(self.sub_reservoirs):
            for j in range(self.sub_reservoirs):
                size = (units[i], units[j])
                if i == j:  # if the matrix is on the diagonal
                    connectivity = self.connectivity[i]
                    spectral_radius = self.spectral_radius[i]
                    recurrent_kernels[i][j] = generate_sub_reservoirs(size, self.initializer, spectral_radius,
                                                                      connectivity, dtype)
                else:
                    # the matrix is off-diagonal, this will create an zero matrix.
                    recurrent_kernels[i][j] = tf.zeros(size, dtype=dtype)
        matrix = join_matrices(recurrent_kernels)
        if self.gsr is not None:  # Normalize the entire matrix
            scaling = tf.math.divide_no_nan(tf.cast(self.gsr, dtype), get_spectral_radius(matrix, dtype))
            matrix = tf.multiply(matrix, scaling)
        return matrix


class IIRESN(tf.keras.initializers.Initializer):
    """
    Recurrent kernel initializer for IIRESN.
    """
    def __init__(self, sub_reservoirs, reservoirs_connectivity, spectral_radius, off_diagonal_limits,
                 gsr: Optional[float] = None,
                 initializer=tf.keras.initializers.GlorotUniform()):
        self.sub_reservoirs = sub_reservoirs
        self.rc = check_matrix(reservoirs_connectivity, sub_reservoirs, "connectivity")
        self.spectral_radius = check_vector(spectral_radius, sub_reservoirs, "spectral radius")
        self.off_diagonal_limits = check_matrix(off_diagonal_limits, sub_reservoirs, "off diagonal")
        self.gsr = gsr
        self.initializer = initializer
        self.partitions = [1. / sub_reservoirs for _ in range(sub_reservoirs)]

    def __call__(self, shape, dtype=None, **kwargs):
        self.rc = [[tf.cast(connectivity, dtype) for connectivity in row] for row in self.rc]
        self.spectral_radius = [tf.cast(spectral_radius, dtype) for spectral_radius in self.spectral_radius]

        units = split_units(shape[0], self.partitions)
        recurrent_kernels = [[_ for _ in range(self.sub_reservoirs)] for _ in range(self.sub_reservoirs)]
        for i in range(self.sub_reservoirs):
            for j in range(self.sub_reservoirs):
                size = (units[i], units[j])
                connectivity = self.rc[i][j]
                if i == j:
                    # if the matrix is on the diagonal, generate a sub-reservoir
                    spectral_radius = self.spectral_radius[i]
                    recurrent_kernels[i][j] = generate_sub_reservoirs(size, self.initializer, spectral_radius,
                                                                      connectivity, dtype)
                else:
                    # the matrix is off-diagonal, this will create an interconnectivity matrix.
                    minmax = self.off_diagonal_limits[i][j]
                    init = tf.keras.initializers.RandomUniform(minval=-abs(minmax), maxval=abs(minmax))
                    matrix = init(size, dtype=dtype)
                    connectivity_mask = tf.cast(tf.math.less_equal(tf.random.uniform(size), connectivity), dtype)
                    recurrent_kernels[i][j] = tf.math.multiply(matrix, connectivity_mask)
        # Join sub-reservoirs with interconnectivity matrix.
        matrix = join_matrices(recurrent_kernels)
        if self.gsr is not None:  # Normalize the entire recurrent kernel
            scaling = tf.math.divide_no_nan(self.gsr, get_spectral_radius(matrix, dtype))
            matrix = tf.multiply(matrix, scaling)
        return matrix


class IIRESNvsr(IIRESN):
    """
    Recurrent kernel initializer for IIRESNvsr. Equal to IIRESN initializer.
    """
    def __init__(self, sub_reservoirs, partitions, reservoirs_connectivity, spectral_radius, off_diagonal_limits,
                 gsr: Optional[float] = None, initializer=tf.keras.initializers.GlorotUniform()):
        super().__init__(sub_reservoirs, reservoirs_connectivity, spectral_radius, off_diagonal_limits, gsr, initializer)
        self.partitions = check_partition(partitions, sub_reservoirs)
