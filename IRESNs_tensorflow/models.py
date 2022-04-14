from typing import Optional, List, Union
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.types.core import TensorLike

from IRESNs_tensorflow import layer
from IRESNs_tensorflow import initializers


class ESNInterface(keras.Model):
    """
    Every model use the same methods.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_bias = False
        self.reservoir = None
        self.readout = None

    def compile(self, *args, **kwargs):
        self.readout.compile(*args, **kwargs)

    def get_bias(self):
        return self.reservoir.bias

    def call(self, inputs):
        state = self.reservoir(inputs)
        output = self.readout(state)
        return output

    def fit(self, x, y, *args, **kwargs):
        # does the same for the validation set
        x_val, y_val = kwargs['validation_data']
        x_val_out = self.reservoir(x_val)
        kwargs['validation_data'] = (x_val_out, y_val)

        # applies the reservoirs to all the input sequences in the training set
        train_state = self.reservoir(x)
        # trains the readout with the reservoir states just computed
        return self.readout.fit(train_state, y, *args, **kwargs)

    def evaluate(self, x, y, *args, **kwargs):
        state = self.reservoir(x)
        return self.readout.evaluate(state, y, *args, **kwargs)

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
                Leaking rate of the reservoir. If you pass 1, it is the special case the model does not have leaky
                integration.
                Default: 0.1
            :param reservoir_activation: Activation function to use in the reservoir.
                Default: hyperbolic tangent (`tf.keras.activations.tanh`).
            :param readout_activation: Activation function to use in the readout.
                Default: hyperbolic tangent (`tf.keras.activations.softmax`).
            :return:A ESN model ready to be compiled
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
                      bias_initializer=bias_init,
                      name="reservoir"),
        ])
        self.readout = keras.Sequential([
            keras.layers.Dense(output_units, activation=readout_activation, name="readout")
        ])


class IRESN(ESNInterface):
    def __init__(self,
                 sub_reservoirs: int,
                 output_units: int,
                 units: int = 100,
                 connectivity: Union[List[float], float] = 1.,
                 normalization: Optional[Union[List[float], float]] = None,
                 gsr: Optional[float] = None,
                 vsr: Optional[List[float]] = None,
                 input_scaling: Union[List[float], float] = 1.,
                 bias_scaling: Optional[Union[List[float], float]] = None,
                 leaky: float = 0.1,
                 reservoir_activation=tf.keras.activations.tanh,
                 readout_activation=tf.keras.activations.softmax,
                 **kwargs
                 ):
        """
        :param units: Positive integer, dimensionality of the recurrent kernel.
        :param sub_reservoirs: Positive integer, how many sub-reservoirs generate.
        :param output_units: Positive integer, dimensionality of the readout.
        :param connectivity: Positive Float between 0 and 1 or List of Float between 0 and 1.
            If Float every sub-reservoir will have the same connectivity.
            If List the List must have the length equals to sub_reservoirs. The first element will be the connectivity
            of the first sub-reservoir.
            Default: 0.9.
        :param normalization: Positive Float or List of Float.
            If Float every sub-reservoir will have the same spectral radius.
            If List the List must have the length equals to sub_reservoirs. The first element will be the
            spectral radius of the first sub-reservoir.
            Default: 0.9.
        :param gsr: Optional, Positive Float.
            If Float this will be the spectral radius of the whole reservoir.
            Default: None.
        :param vsr: Optional, List of Positive Float, the length must be equals to sub_reservoir and the sum of the list
            must be 1. Every element is the percentage of units the i-th sub-reservoir will have.
            Default: None
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
        :param leaky: Float between 0 and 1.
            Leaking rate of the reservoir.
            If you pass 1, it is the special case the model does not have leaky integration.
            Default: 0.1
        :param reservoir_activation: Activation function to use in the reservoir.
            Default: hyperbolic tangent (`tf.keras.activations.tanh`).
        :param readout_activation: Activation function to use in the readout.
            Default: hyperbolic tangent (`tf.keras.activations.softmax`).
        :return:A IRESN model ready to be compiled
        """
        super().__init__(**kwargs)

        kernel_init = initializers.SplitKernel(sub_reservoirs, input_scaling)
        recurrent_kernel_init = initializers.IRESN(sub_reservoirs, connectivity, normalization=normalization, gsr=gsr,
                                                   vsr=vsr)

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
                 connectivity: Union[List[List[float]], float],
                 normalization: Optional[Union[List[List[float]], float]] = None,
                 gsr: Optional[float] = None,
                 vsr: Optional[List[float]] = None,
                 input_scaling: Union[List[float], float] = 1.,
                 bias_scaling: Optional[Union[List[float], float]] = None,
                 use_norm2: bool = True,
                 leaky: float = 0.1,
                 reservoir_activation=tf.keras.activations.tanh,
                 readout_activation=tf.keras.activations.softmax,
                 **kwargs
                 ):
        """
        :param units: Positive integer, dimensionality of the recurrent kernel.
        :param sub_reservoirs: Positive integer, how many sub-reservoirs generate.
        :param output_units: Positive integer, dimensionality of the readout.
        :param normalization: Positive Float or List of Float.
            If Float every sub-reservoir will have the same spectral radius.
            If List the List must have the length equals to sub_reservoirs. The first element will be the
            spectral radius of the first sub-reservoir.
            Default: 0.9
        :param gsr: Optional, Positive Float.
            If Float this will be the spectral radius of the whole reservoir.
            Default: None
        :param vsr: Optional, List of Positive Float, the length must be equals to sub_reservoir and the sum of the list
            must be 1. Every element is the percentage of units the i-th sub-reservoir will have.
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
        :return:A IRESN model ready to be compiled
        """
        super().__init__(**kwargs)

        kernel_init = initializers.SplitKernel(sub_reservoirs, input_scaling)
        recurrent_kernel_init = initializers.IIRESN(sub_reservoirs, connectivity, normalization, gsr, vsr, use_norm2)
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
