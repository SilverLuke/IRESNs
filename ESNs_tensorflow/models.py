from typing import Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.types.core import TensorLike

from ESNs_tensorflow import layer
from ESNs_tensorflow import initializers


class ESNInterface(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_bias = False
        self.reservoir = None
        self.readout = None

    def compile(self, *args, **kwargs):
        self.readout.compile(*args, **kwargs)

    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout(reservoir_states)
        return output

    def fit(self, x, y, *args, **kwargs):
        # applies the reservoirs to all the input sequences in the training set
        x_train_out = self.reservoir(x)

        # does the same for the validation set
        x_val, y_val = kwargs['validation_data']
        x_val_out = self.reservoir(x_val)
        kwargs['validation_data'] = (x_val_out, y_val)

        # trains the readout with the reservoir states just computed
        return self.readout.fit(x_train_out, y, *args, **kwargs)

    def evaluate(self, x, y, *args, **kwargs):
        x_train_out = self.reservoir(x)
        return self.readout.evaluate(x_train_out, y, *args, **kwargs)

    @property
    def units(self):
        return self.reservoir.layers[1].units


class ESN(ESNInterface):
    def __init__(self,
                 units: int,
                 output_units: int,
                 spectral_radius: float = 0.9,
                 connectivity: float = 1.,
                 input_scaling: float = 1.,
                 bias_scaling=None,
                 leaky=0.1,
                 reservoir_activation=tf.keras.activations.tanh,
                 readout_activation=tf.keras.activations.softmax,
                 **kwargs
                 ):
        """
            :param units: Positive integer, dimensionality of the recurrent kernel.
            :param output_units: Positive integer, dimensionality of the readout.
            :param spectral_radius: Float between 0 and 1.
                Desired spectral radius of recurrent weight matrix.
                Default: 0.9.
            :param connectivity: Float between 0 and 1.
                Connection probability between two units inside the reservoir.
                Default: 1.
            :param bias_scaling: Optional, if None the model will not use a bias vector, use a float for generate a bias vector.
                Default: None.
            :param input_scaling: Float for generate the kernel matrix.
                Default: 1.
            :param leaky: Float between 0 and 1.
                Leaking rate of the reservoir.
                If you pass 1, it is the special case the model does not have leaky
                integration.
                Default: 0.1
            :param reservoir_activation: Activation function to use in the reservoir.
                Default: hyperbolic tangent (`tf.keras.activations.tanh`).
            :param readout_activation: Activation function to use in the readout.
                Default: hyperbolic tangent (`tf.keras.activations.softmax`).
            :return A ESN model ready to be compiled
        """
        super().__init__(**kwargs)

        kernel_init = initializers.Kernel(initializer=tf.keras.initializers.RandomUniform(
            minval=-input_scaling, maxval=input_scaling)
        )

        if connectivity == 1.0:
            recurrent_kernel_init = initializers.FullConnected(spectral_radius)
        else:
            recurrent_kernel_init = initializers.RecurrentKernel(connectivity, spectral_radius)

        self.use_bias = bias_scaling is not None
        if self.use_bias:
            bias_init = tf.keras.initializers.RandomUniform(minval=-bias_scaling, maxval=bias_scaling)
        else:
            bias_init = None

        self.reservoir = keras.Sequential([
            keras.layers.Masking(),
            layer.ESN(units, leaky,
                    activation=reservoir_activation,
                    use_bias=self.use_bias,
                    kernel_initializer=kernel_init,
                    recurrent_initializer=recurrent_kernel_init,
                    bias_initializer=bias_init),
        ])
        self.readout = keras.Sequential([
            keras.layers.Dense(output_units, activation=readout_activation, name="readout")
        ])


class IRESN(ESNInterface):
    def __init__(self,
                 units: int,
                 sub_reservoirs: int,
                 output_units: int,
                 spectral_radius=0.9,
                 gsr: Optional[float] = None,
                 connectivity=1.,
                 leaky=0.1,
                 input_scaling=1.,
                 bias_scaling=None,
                 reservoir_activation=tf.keras.activations.tanh,
                 readout_activation=tf.keras.activations.softmax,
                 **kwargs
                 ):
        """
        :param units: Positive integer, dimensionality of the recurrent kernel.
        :param sub_reservoirs: Positive integer, how many sub-reservoirs generate.
        :param output_units: Positive integer, dimensionality of the readout.
        :param spectral_radius: Positive Float or List of Float.
            If Float every sub-reservoir will have the same spectral radius.
            If List the List must have the length equals to sub_reservoirs. The first element will be the
            spectral radius of the first sub-reservoir.
            Default: 0.9.
        :param gsr: Optional, Positive Float.
            If Float this will be the spectral radius of the whole reservoir.
            Default: None.
        :param connectivity: Positive Float between 0 and 1 or List of Float between 0 and 1.
            If Float every sub-reservoir will have the same connectivity.
            If List the List must have the length equals to sub_reservoirs. The first element will be the connectivity
            of the first sub-reservoir.
            Default: 0.9.
        :param leaky: Float between 0 and 1.
            Leaking rate of the reservoir.
            If you pass 1, it is the special case the model does not have leaky integration.
            Default: 0.1
        :param input_scaling: Float or List of Float between 0 and 1.for generate the kernel matrix.
            If Float every sub-reservoir will have the same input scaling.
            If List the List must have the length equals to sub_reservoirs. The first element will be the input scaling
            of the first sub-reservoir.
            Default: 1.
        :param bias_scaling: Optional, if None the model will not use a bias vector, use a float or List of Float for
        generate a bias vector.
            If Float every sub-reservoir will have the same bias scaling.
            If List the List must have the length equals to sub_reservoirs. The first element will be the bias scaling
            of the first sub-reservoir.
            Default: None.
        :param reservoir_activation: Activation function to use in the reservoir.
            Default: hyperbolic tangent (`tf.keras.activations.tanh`).
        :param readout_activation: Activation function to use in the readout.
            Default: hyperbolic tangent (`tf.keras.activations.softmax`).
        :return A IRESN model ready to be compiled
        """
        super().__init__(**kwargs)

        kernel_init = initializers.SplitKernel(sub_reservoirs, input_scaling)
        recurrent_kernel_init = initializers.IRESN(sub_reservoirs, connectivity, spectral_radius, gsr=gsr)

        self.use_bias = bias_scaling is not None
        if self.use_bias:
            bias_init = initializers.SplitBias(bias_scaling, sub_reservoirs)
        else:
            bias_init = None

        self.reservoir = keras.Sequential([
            keras.layers.Masking(),
            layer.ESN(units, leaky,
                    activation=reservoir_activation,
                    use_bias=self.use_bias,
                    kernel_initializer=kernel_init,
                    recurrent_initializer=recurrent_kernel_init,
                    bias_initializer=bias_init),
        ])
        self.readout = keras.Sequential([
            keras.layers.Dense(output_units, activation=readout_activation, name="readout")
        ])


class IIRESN(ESNInterface):
    def __init__(self,
                 units: int,
                 sub_reservoirs: int,
                 output_units: int,
                 spectral_radius=0.9,
                 gsr: Optional[float] = None,
                 connectivity=1.,
                 off_diagonal_limits=0.5,
                 leaky=0.1,
                 input_scaling = 1.,
                 bias_scaling=None,
                 reservoir_activation=tf.keras.activations.tanh,
                 readout_activation=tf.keras.activations.softmax,
                 **kwargs
                 ):
        """
        :param units: Positive integer, dimensionality of the recurrent kernel.
        :param sub_reservoirs: Positive integer, how many sub-reservoirs generate.
        :param output_units: Positive integer, dimensionality of the readout.
        :param spectral_radius: Positive Float or List of Float.
            If Float every sub-reservoir will have the same spectral radius.
            If List the List must have the length equals to sub_reservoirs. The first element will be the
            spectral radius of the first sub-reservoir.
            Default: 0.9
        :param gsr: Optional, Positive Float.
            If Float this will be the spectral radius of the whole reservoir.
            Default: None
        :param connectivity: Positive Float between 0 and 1 or square Matrix (List of List) of Float between 0 and 1.
            If Float: every sub-reservoir and every interconnectivity matrix will have the same connectivity.
            If Matrix: the Matrix must be squared and have the length equals to sub_reservoirs. The diagonals values are
             the connectivity iper-parameter of the sub-reservoir,
            instead the off-diagonal values are the connectivity iper-parameter between two reservoirs.
            Default: 1.
        :param off_diagonal_limits: Positive Float or square Matrix (List of List) of Float.
            If Float every interconnectivity will have the same limit (min and max) between any resevoirs.
            If Matrix: the Matrix must be squared and have the length equals to sub_reservoirs. The diagonals values are
             useless, instead the off-diagonal value in the position x,y
            is the limit (min and max) of the interconnectivity matrix between the reservoir x and y.
            Default: 0.9
        :param leaky: Float between 0 and 1.
            Leaking rate of the reservoir.
            If you pass 1, it is the special case the model does not have leaky integration.
            Default: 0.1
        :param input_scaling: Float or List of Float between 0 and 1.for generate the kernel matrix.
            If Float every sub-reservoir will have the same input scaling.
            If List the List must have the length equals to sub_reservoirs. The first element will be the input scaling
            of the first sub-reservoir.
            Default: 1.
        :param bias_scaling: Optional, if None the model will not use a bias vector, use a float or List of Float for
        generate a bias vector.
            If Float every sub-reservoir will have the same bias scaling.
            If List the List must have the length equals to sub_reservoirs. The first element will be the bias scaling
            of the first sub-reservoir.
            Default: None.
        :param reservoir_activation: Activation function to use in the reservoir.
            Default: hyperbolic tangent (`tf.keras.activations.tanh`).
        :param readout_activation: Activation function to use in the readout.
            Default: hyperbolic tangent (`tf.keras.activations.softmax`).
        :return A IRESN model ready to be compiled
        """
        super().__init__(**kwargs)

        kernel_init = initializers.SplitKernel(sub_reservoirs, input_scaling)
        recurrent_kernel_init = initializers.IIRESN(sub_reservoirs, connectivity, spectral_radius, off_diagonal_limits, gsr=gsr)
        self.use_bias = bias_scaling is not None
        if self.use_bias:
            bias_init = initializers.SplitBias(bias_scaling, sub_reservoirs)
        else:
            bias_init = None

        self.reservoir = keras.Sequential([
            keras.layers.Masking(),
            layer.ESN(units, leaky,
                    activation=reservoir_activation,
                    use_bias=self.use_bias,
                    kernel_initializer=kernel_init,
                    recurrent_initializer=recurrent_kernel_init,
                    bias_initializer=bias_init),
        ])
        self.readout = keras.Sequential([
            keras.layers.Dense(output_units, activation=readout_activation, name="readout")
        ])


class IIRESNvsr(ESNInterface):
    """
    The IIRESNvsr (Multiple ESN Interconnected Sized) model allows a static partitioning of the sub-reservoir.
    """

    def __init__(self,
                 units: int,
                 sub_reservoirs: int,
                 partitions: TensorLike,
                 output_units: int,
                 input_scaling,
                 off_diagonal_limits,
                 reservoir_activation=tf.keras.activations.tanh,
                 readout_activation=tf.keras.activations.softmax,
                 bias_scaling=None,
                 connectivity=1.,
                 spectral_radius=0.9,
                 gsr: Optional[float] = None,
                 leaky=0.1,
                 **kwargs
                 ):
        """
        :param units: Positive integer, dimensionality of the recurrent kernel.
        :param sub_reservoirs: Positive integer, how many sub-reservoirs generate.
        :param partitions: List of Positive Float, the length must be equals to sub_reservoir and the sum of the list
            must be 1. Every element is the percentage of units the i-th sub-reservoir  have.
        :param output_units: Positive integer, dimensionality of the readout.
        :param spectral_radius: Positive Float or List of Float.
            If Float every sub-reservoir will have the same spectral radius.
            If List the List must have the length equals to sub_reservoirs. The first element will be the
            spectral radius of the first sub-reservoir.
            Default: 0.9
        :param gsr: Optional, Positive Float.
            If Float this will be the spectral radius of the whole reservoir.
            Default: None
        :param connectivity: Positive Float between 0 and 1 or square Matrix (List of List) of Float between 0 and 1.
            If Float: every sub-reservoir and every interconnectivity matrix will have the same connectivity.
            If Matrix: the Matrix must be squared and have the length equals to sub_reservoirs. The diagonals values are
             the connectivity iper-parameter of the sub-reservoir,
            instead the off-diagonal values are the connectivity iper-parameter between two reservoirs.
            Default: 1.
        :param off_diagonal_limits: Positive Float or square Matrix (List of List) of Float.
            If Float every interconnectivity will have the same limit (min and max) between any resevoirs.
            If Matrix: the Matrix must be squared and have the length equals to sub_reservoirs. The diagonals values are
            useless, instead the off-diagonal value in the position x,y is the limit (min and max) of the
            interconnectivity matrix between the reservoir x and y.
            Default: 0.9
        :param leaky: Float between 0 and 1.
            Leaking rate of the reservoir.
            If you pass 1, it is the special case the model does not have leaky integration.
            Default: 0.1
        :param input_scaling: Float or List of Float between 0 and 1.for generate the kernel matrix.
            If Float every sub-reservoir will have the same input scaling.
            If List the List must have the length equals to sub_reservoirs. The first element will be the input scaling
            of the first sub-reservoir.
            Default: 1.
        :param bias_scaling: Optional, if None the model will not use a bias vector, use a float or List of Float for
        generate a bias vector.
            If Float every sub-reservoir will have the same bias scaling.
            If List the List must have the length equals to sub_reservoirs. The first element will be the bias scaling
            of the first sub-reservoir.
            Default: None.
        :param reservoir_activation: Activation function to use in the reservoir.
            Default: hyperbolic tangent (`tf.keras.activations.tanh`).
        :param readout_activation: Activation function to use in the readout.
            Default: hyperbolic tangent (`tf.keras.activations.softmax`).
        :return A IRESN model ready to be compiled
        """
        super().__init__(**kwargs)
        kernel_init = initializers.SplitKernel(sub_reservoirs, input_scaling, partitions=partitions)
        recurrent_kernel_init = initializers.IIRESNvsr(sub_reservoirs, partitions, connectivity, spectral_radius,
                                          off_diagonal_limits, gsr=gsr)

        self.use_bias = bias_scaling is not None
        if self.use_bias:
            bias_init = initializers.SplitBias(bias_scaling, sub_reservoirs)
        else:
            bias_init = None

        self.reservoir = keras.Sequential([
            keras.layers.Masking(),
            layer.ESN(units, leaky,
                    activation=reservoir_activation,
                    kernel_initializer=kernel_init,
                    recurrent_initializer=recurrent_kernel_init,
                    bias_initializer=bias_init),
        ])
        self.readout = keras.Sequential([
            keras.layers.Dense(output_units, activation=readout_activation, name="readout")
        ])
