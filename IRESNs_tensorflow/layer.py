import tensorflow as tf
from tensorflow.python.types.core import TensorLike


class ESN(tf.keras.layers.RNN):
    def __init__(
            self,
            units: TensorLike,
            leaky: TensorLike,
            activation=tf.keras.activations.tanh,
            kernel_initializer=tf.keras.initializers.GlorotUniform,
            recurrent_initializer=tf.keras.initializers.GlorotUniform,
            use_bias: bool = False,
            bias_initializer=tf.keras.initializers.Zeros,
            return_sequences=False,
            **kwargs,
    ):
        """
        :param units: Positive integer, dimensionality of the recurrent kernel.
        :param leaky: Float between 0 and 1.
                Leaking rate of the reservoir. If you pass 1, it is the special case the model does not have leaky
                integration.
                Default: 0.1
        :param activation: Activation function to use in the reservoir.
               Default: hyperbolic tangent (`tf.keras.activations.tanh`).
        :param kernel_initializer: Initializer for the kernel. See initializers.py and
               https://www.tensorflow.org/api_docs/python/tf/keras/initializers
               Default: tf.keras.initializers.GlorotUniform.
        :param recurrent_initializer: Initializer for the recurrent kernel. See initializers.py and
               https://www.tensorflow.org/api_docs/python/tf/keras/initializers
               Default: tf.keras.initializers.GlorotUniform.
        :param use_bias: Boolean. This layer use bias
        :param bias_initializer: Initializer for the bias vector. See initializers.py and
               https://www.tensorflow.org/api_docs/python/tf/keras/initializers
               Default: tf.keras.initializers.Zeros.
        :param return_sequences: Boolean. Test for Deep layer.
        """
        cell = Reservoir(
            units=units,
            leaky=leaky,
            activation=activation,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            use_bias=use_bias,
            bias_initializer=bias_initializer,
            dtype=kwargs.get("dtype")
        )
        super().__init__(cell, return_sequences=return_sequences, **kwargs)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        return super().call(
            inputs,
            mask=mask,
            training=training,
            initial_state=initial_state,
            constants=None
        )

    @property
    def units(self):
        return self.cell.units

    @property
    def leaky(self):
        return self.cell.leaky

    @property
    def spectral_radius(self):
        return self.cell.initializer.spectral_radius

    @property
    def activation(self):
        return self.cell.activation


class Reservoir(tf.keras.layers.AbstractRNNCell):
    def __init__(
            self,
            units: int,
            leaky: float = 0.1,
            activation=tf.keras.activations.tanh,
            kernel_initializer=tf.keras.initializers.GlorotUniform,
            recurrent_initializer=tf.keras.initializers.GlorotUniform,
            use_bias: bool = True,
            bias_initializer=tf.keras.initializers.Zeros,
            **kwargs,
    ):
        """
        :param units: Positive integer, dimensionality of the recurrent kernel.
        :param leaky: Float between 0 and 1.
                Leaking rate of the reservoir. If you pass 1, it is the special case the model does not have leaky
                integration.
                Default: 0.1
        :param activation: Activation function to use in the reservoir.
               Default: hyperbolic tangent (`tf.keras.activations.tanh`).
        :param kernel_initializer: Initializer for the kernel. See initializers.py and
               https://www.tensorflow.org/api_docs/python/tf/keras/initializers
               Default: tf.keras.initializers.GlorotUniform.
        :param recurrent_initializer: Initializer for the recurrent kernel. See initializers.py and
               https://www.tensorflow.org/api_docs/python/tf/keras/initializers
               Default: tf.keras.initializers.GlorotUniform.
        :param use_bias: Boolean. This layer use bias
        :param bias_initializer: Initializer for the bias vector. See initializers.py and
               https://www.tensorflow.org/api_docs/python/tf/keras/initializers
               Default: tf.keras.initializers.Zeros.
        """
        super().__init__(name="reservoir", **kwargs)

        self.units = units
        self.leaky = leaky
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer

        self._state_size = units
        self._output_size = units

        if self.bias_initializer is tf.keras.initializers.Zeros and use_bias:
            print("Warning!! use_bias is true but bias initializer is Zero")
        self.use_bias = use_bias

        self.kernel = None
        self.recurrent_kernel = None
        self.bias = None
        self.built = False

    def build(self, inputs_shape):
        input_size = tf.compat.dimension_value(tf.TensorShape(inputs_shape)[-1])
        if input_size is None:
            raise ValueError(
                "Could not infer input size from inputs.get_shape()[-1]. Shape received is %s"
                % inputs_shape
            )

        self.kernel = self.add_weight(
            name="kernel",
            shape=[input_size, self.units],
            initializer=self.kernel_initializer,
            trainable=False,
            dtype=self.dtype,
        )

        self.recurrent_kernel = self.add_weight(
            name="recurrent_kernel",
            shape=[self.units, self.units],
            initializer=self.recurrent_initializer,
            trainable=False,
            dtype=self.dtype,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=[self.units],
                initializer=self.bias_initializer,
                trainable=False,
                dtype=self.dtype,
            )

        self.built = True

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    @tf.function
    def call(self, inputs, states):
        in_matrix = tf.concat([inputs, states[0]], axis=1, name="in->state")
        weights_matrix = tf.concat([self.kernel, self.recurrent_kernel], axis=0, name="ker->rec")

        output = tf.matmul(in_matrix, weights_matrix, name="bigmul")
        if self.use_bias:
            output = output + self.bias
        output = self.activation(output)
        output = (1 - self.leaky) * states[0] + self.leaky * output

        return output, [output]
