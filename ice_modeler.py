# ---------------------------
# NSIDC Sea Ice Index Modeler
# ---------------------------
# This class defines an object which can be used to fit certain keras temporal
# predictive models with relative ease.
# Dependencies:
# - numpy
# - keras
# - keras-tcn (download from link or install with pip)
# ---------------------------

# ----------------
# Import Libraries
# ----------------
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, TimeDistributed, Flatten
from keras.layers import AveragePooling2D
from keras.layers.convolutional import Conv2D
from tcn import TCN
import keras.backend as kb

# -----------------
# Class Definitions
# -----------------
class IceModeler:
    def __init__(self, image_shape=(304, 448), image_index='daily'):
        self.seq_model = Sequential()
        self.image_shape = image_shape
        self.im_rows = image_shape[1]
        self.im_cols = image_shape[0]
        if image_index not in ['daily', 'monthly', 'yearly']:
            raise Exception('Only daily, monthly, or yearly options allowed.')
        self.image_index = image_index
        self.image_size = self.im_rows * self.im_cols
        #self.conc_scale = 1000
        self.conc_scale = 2550
        self.ext_scale = 255

    def reshape_to_tcn(self, frames):
        reshaped = frames.reshape((*frames.shape[:-3], self.image_size))
        return reshaped

    def reshape_to_image(self, frames):
        return frames.reshape((*frames.shape[:-1], self.im_rows, self.im_cols))

    def fill_pole_hole(self, frames):
        # say its fully ice (it's the pole)
        frames[frames == 2510] = 1000
        return frames

    def scale_to_normal(self, frames, image_type):
        if image_type == 'concentration':
            frames = self.fill_pole_hole(frames)
            frames[frames > 1000] = -self.conc_scale
            return frames / self.conc_scale
        elif image_type == 'extent':
            return frames / self.ext_scale

    def scale_from_normal(self, frames, image_type):
        if image_type == 'concentration':
            scaled = np.round(frames * self.conc_scale)
            scaled[scaled < -850] = 2540
            zeros = np.logical_and(scaled > -200, scaled < 0)
            scaled[zeros] = 0
            return scaled
        elif image_type == 'extent':
            return  np.round(frames * self.ext_scale)

    def add_n_tcn(
            self,
            n,
            nb_filters=10, 
            kernel_size=2,
            nb_stacks=1,
            padding='causal',
            return_sequences=True,
            activation='tanh'):
        # this is the first TCN layer and we need to do some special input shape
        # handling
        if len(self.seq_model.layers) == 0:
            num_same = n-1
            input_shape = (None, self.image_size)
            self.seq_model.add(TCN(
                nb_filters=nb_filters,
                kernel_size=kernel_size,
                nb_stacks = nb_stacks,
                padding=padding,
                return_sequences=return_sequences,
                activation=activation,
                input_shape=input_shape))
        else:
            num_same = n

        for i in range(num_same):
            tcn_layer = TCN(
                nb_filters=nb_filters,
                kernel_size=kernel_size,
                nb_stacks = nb_stacks,
                padding=padding,
                return_sequences=return_sequences,
                activation=activation)
            self.seq_model.add(tcn_layer)

    def add_n_td_conv2d(
            self,
            n,
            filters=4,
            kernel_size=4,
            strides=1,
            padding='same',
            data_format='channels_first',
            activation='tanh'):
        if len(self.seq_model.layers) == 0:
            num_same = n-1
            input_shape = (None, self.im_rows, self.im_cols)
            self.seq_model.add(TimeDistributed(Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                    activation=activation,
                    input_shape=input_shape)))
        else:
            num_same = n

        for i in range(num_same):
            self.seq_model.add(TimeDistributed(Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                    activation=activation)))

    def add_td_flatten(self, data_format='channels_first'):
        if len(self.seq_model.layers) == 0:
            input_shape = (None, self.im_rows, self.im_cols)
            td_flat = TimeDistributed(Flatten(
                data_format=data_format,
                input_shape=input_shape))
        else:
            td_flat = TimeDistributed(Flatten(data_format=data_format))
        self.seq_model.add(td_flat)

    def add_td_im_reshape(self):
        td_reshape = TimeDistributed(Reshape((-1, self.im_rows, self.im_cols)))
        self.seq_model.add(td_reshape)

    def add_td_avg_pool(
            self, 
            pool_size, 
            strides=None, 
            padding='valid',
            data_format='channels_first'):
        td_pool = TimeDistributed(AveragePooling2D(
            pool_size=pool_size, 
            strides=strides,
            padding=padding,
            data_format=data_format))
        self.seq_model.add(td_pool)

    def add_n_td_dense(
            self,
            n,
            units,
            activation='tanh'):
        if len(self.seq_model.layers) == 0:
            num_same = n-1
            input_shape = (None, self.image_size)
            self.seq_model.add(TimeDistributed(Dense(
                units=units,
                activation=activation,
                input_shape=input_shape)))
        else:
            num_same = n

        for i in range(num_same):
            dense_layer = TimeDistributed(
                Dense(units=units, activation=activation))
            self.seq_model.add(dense_layer)


    def make_power_error(self, p):
        def power_error(true, pred, p):
            diff_pow = kb.abs(true-pred)**p
            mean = kb.mean(diff_pow)
            return mean

        p_err = lambda true, pred: power_error(true, pred, p)
        return p_err

    def make_hard_tanh(self, alpha=5):
        def scaled_tanh(x, alpha):
            tanh = kb.tanh(alpha*x)
            return tanh
        alpha_tanh = lambda x: scaled_tanh(x, alpha)
        return alpha_tanh

    def compile(self, loss, optimizer):
        self.seq_model.compile(loss=loss, optimizer=optimizer)

    def fit(self, x_train, y_train, epochs, batch_size=1):
        self.seq_model.fit(
            x_train, 
            y_train, 
            epochs=epochs, 
            batch_size=batch_size)

    def predict(self, test_frames):
        preds = self.seq_model.predict(test_frames)
        return preds

    def reset_model(self):
        self.seq_model = Sequential()

