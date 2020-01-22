# ---------------------------
# NSIDC Sea Ice Index Modeler
# ---------------------------
# This class defines an object which can be used to fit certain keras temporal
# predictive models with relative ease.
#
# Dependencies:
# - numpy
# - keras
# - keras-tcn (download from https://github.com/philipperemy/keras-tcn
#   or install with pip)
# ---------------------------

# ----------------
# Import Libraries
# ----------------
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, TimeDistributed, Flatten
from keras.layers import AveragePooling2D
from keras.layers.convolutional import Conv2D
from keras import regularizers
from tcn import TCN
import keras.backend as kb
import tensorflow as tf

# -----------------
# Class Definitions
# -----------------
class IceModeler:
    """
    This class instantiates an object which can be used to model the
    concentration of sea ice from numpy arrays read from GeoTiff rasters of sea
    ice distributed by the NSIDC in the Sea Ice Index. This is an extension of
    a Keras sequential model which incorporates Temporal Convolutional Networks
    as well some spatial convolution, and follows a roughly similar API. It
    seeks to standardize some image parameters for shaping, as well as provide
    methods for the addition of various time distributed layers with sensible
    defaults for this dataset (though it could be used on other time distributed
    image data as well).

    Attributes:
    -----------
        :Keras Sequential:              Keras sequential model which holds all
                                        the model layers.
        :tuple(int) image_shape:        image shape in width x height format
        :int im_rows:                   number of rows in image array (height)
        :int im_cols:                   number of cols in image array (width)
        :int im_size:                   size of the image in pixels (width * 
                                        height)
        :int conc_scale:                constant 2550, the max value present in
                                        the concentration raster bands
        :int ext_scale:                 constant 255, the max value present in
                                        the extent raster bands
    """
    def __init__(self, image_shape=(304, 448)):
        """
        Initializes an IceModeler object.

        Parameters:
        -----------
            :tuple(int) image_shape:    the shape of the images to be fed into
                                        the model in width x height format
        """
        self.seq_model = Sequential()
        self.image_shape = image_shape
        self.im_rows = image_shape[1]
        self.im_cols = image_shape[0]
        self.im_size = self.im_rows * self.im_cols
        #self.conc_scale = 1000
        self.conc_scale = 2550
        self.ext_scale = 255

    def reshape_to_tcn(self, frames):
        """
        Reshapes the given array of frames to a shape that is suitable for TCN
        input by flattening the individual image arrays

        Parameters:
        -----------
            :np.ndarray frames:         numpy array containing frames/image
                                        arrays to be flattened (assumed to be in
                                        channels_first format).

        Returns:
        --------
            :np.ndarray reshaped:       reshaped array with flattened frames
                                        (maintains sampling)
        """
        reshaped = frames.reshape((*frames.shape[:-3], self.im_size))
        return reshaped

    def reshape_to_image(self, frames):
        """
        Reshapes the given array of frames in flat format to a shape that is 
        suitable to be interpreted as an image in width x height format

        Parameters:
        -----------
            :np.ndarray frames:         numpy array containing flattened images
                                        to be reshaped into width x height
                                        format

        Returns:
        --------
            :np.ndarray reshaped:       reshaped array with frames in width x
                                        height format (maintains sampling)
        """
        reshaped = frames.reshape(
            (*frames.shape[:-1], self.im_rows, self.im_cols))
        return reshaped

    def fill_pole_hole(self, frames):
        """
        Fills the portions of the given numpy array of frames (either image or
        flattened format) which are designated to be for the pole with the value
        for ice. Only applicable for concentration images.

        Parameters:
        -----------
            :np.ndarray frames:         numpy array containing image data to be
                                        filled

        Returns:
        --------
            :np.ndarray filled:         numpy array with pole values replaced
                                        by ice values in the same shape
        """
        # say its fully ice (it's the pole)
        filled = np.copy(frames)
        filled[filled == 2510] = 1000
        return filled

    def process_image_masks(self, frames):
        """
        This method looks at the given array of frames and creates a mask
        identifying regions in which ice is specified to form and those regions
        which correspond to land areas or other fixed values.

        Parameters:
        -----------
            :np.ndarray frames:       numpy array containing image data for
                                        which masks should be generated

        Returns:
        --------
            :np.ndarray masks:          numpy array containing 0s where sea ice
                                        should not be found and 1s where sea ice
                                        should be found
            :np.ndarray original_vals:  numpy array of the original values that
                                        were contained in the now masked
                                        regions
        """
        im_array = self.fill_pole_hole(frames)
        masks = im_array <= 1000
        original_val_masks = im_array > 1000
        original_vals = im_array*original_val_masks.astype(int)
        masks = masks.astype(int)
        masked_frames = im_array*masks
        return masks, masked_frames, original_vals

    def scale_to_normal(self, frames, image_type, masked=False):
        """
        Scales the given array of raw band values to fit between the values of
        -1 and 1 by simply dividing by the maximum value dictated by the image
        type.

        In the case of concentration this function also sets the non-ice values
        to -1 (to get max separation between these pixels and the sea ice pixels
        to be regressed on)

        Attributes:
        -----------
            :np.ndarray frames:         numpy array of frames to be scaled into
                                        the range of -1 to 1
            :str image_type:            one of 'concentration' or 'extent' to
                                        indicate the appropriate scaling factor.

        Returns:
        --------
            :np.ndarray normed:         the scaled frames in the same shape
        """
        normed = np.copy(frames)
        if image_type == 'concentration':
            if not masked:
                normed = self.fill_pole_hole(normed)
                normed[normed > 1000] = -self.conc_scale
                return normed / self.conc_scale
            else:
                return normed / 1000
        elif image_type == 'extent':
            return normed / self.ext_scale

    def scale_from_normal(self, frames, image_type, masked=False):
        """
        Scales the given array of normalized band values back to the standard
        form for the image type by simply multiplying by the appropriate scaling
        factor.

        In the case of concentration images, because this function is used to
        scale back from predicted values, certain ranges have been set to
        appropriate values to facilitate later coloration of images.

        Parameters:
        -----------
            :np.ndarray frames:         numpy array of frames to be scaled back
                                        to the appropriate magnitude for the
                                        image type

            :str image_type:            one of 'concentration' or 'extent' to
                                        indicate the appropriate scaling factor

        Returns:
        --------
            :np.ndarray scaled:         an array of scaled and appropriately
                                        filtered images frames of the same shape
                                        as frames
        """
        scaled = np.copy(frames)
        if image_type == 'concentration':
            if not masked:
                scaled = np.round(scaled * self.conc_scale)
                scaled[scaled < -850] = 2540
                zeros = np.logical_and(scaled > -200, scaled < 0)
                scaled[zeros] = 0
                return scaled
            else:
                return scaled*1000
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
            activation='tanh',
            kernel_initializer='he_normal',
            dropout_rate=0):
        """
        This method adds n Temporal Convolutional Networks to the Sequential
        model with the specified hyperparameters (which will be the same if
        multiple layers are to be added). If the Sequential model has no layers,
        this method will appropriately set the input shape of the first layer
        based on image information stored in the instance's attributes.

        Parameters:
        -----------
            :int n:                     the number of layers to add
            :int nb_filters:            the number of filters for each layer 
                                        (default 10)
            :int kernel_size:           the convolutional kernel size for each
                                        layer (default 2)
            :int nb_stacks:             the number of residual blocks to use for
                                        each layer (default 1)
            :str padding:               one of 'causal' or 'valid' indicating
                                        the style of padding to use (default 
                                        'causal') (note that 'valid' padding
                                        removes validity for prediction of
                                        future events)
            :bool return_sequences:     boolean indicating whether to return
                                        the full sequence or the last state
                                        (default True)
            :str or valid activation:   activation function to use for each
                                        layer -- can be a custom function or
                                        activation layer or simply a Keras str
            :str kernel_initializer:    string for a keras kernel-initializer to
                                        use, for purposes of changing by
                                        activation function
        """

        # this is the first TCN layer and we need to do some special input shape
        # handling
        if len(self.seq_model.layers) == 0:
            num_same = n-1
            input_shape = (None, self.im_size)
            self.seq_model.add(TCN(
                nb_filters=nb_filters,
                kernel_size=kernel_size,
                nb_stacks = nb_stacks,
                padding=padding,
                return_sequences=return_sequences,
                activation=activation,
                input_shape=input_shape,
                kernel_initializer=kernel_initializer,
                dropout_rate=dropout_rate))
        else:
            num_same = n

        for i in range(num_same):
            tcn_layer = TCN(
                nb_filters=nb_filters,
                kernel_size=kernel_size,
                nb_stacks = nb_stacks,
                padding=padding,
                return_sequences=return_sequences,
                activation=activation,
                kernel_initializer=kernel_initializer,
                dropout_rate=dropout_rate)
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
        """
        This method adds n TimeDistributed 2D Convolutional layers to the
        Sequential model with the specified hyperparameters (which will be the
        same if multiple layers are to be added). If the Sequential model has no
        layers, this method will appropriately set the input shape of the first
        layer from image information stored in the instance's attributes.

        Parameters:
        -----------
            :int n:                         the number of layers to add
            :int filters:                   the number of 2D filters per layer
                                            (default 4)
            :int/tuple(int) kernel_size:    the kernel size to use for
                                            convolution (default 4)
            :int/tuple(int) strides:        the stride length to use for
                                            convolution (default 1)
            :str padding:                   one of 'same' or 'valid' indicating
                                            the style of padding to use
            :str data_format:               one of 'channels_first' or
                                            'channels_last' indicating whether
                                            the input shape denotes image
                                            channels before or after the image
                                            shape
            :str or valid activation:       activation function to use for each
                                            layer -- can be a custom function
                                            or activation layer or a simple
                                            Keras string
        """
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
        """
        Adds a single TimeDistributed image flattening layer to the Sequential
        model, expanding each frame to a single array. If the Sequential model
        has no layers, this method will appropriately set the input shape based
        on image information stored in the instance's attributes.

        Parameters:
            :str data_format:           one of 'channels_first' or 
                                        'channels_last' indicating whether the
                                        input shape denotes image bands before
                                        or after the image shape
        """
        if len(self.seq_model.layers) == 0:
            input_shape = (None, self.im_rows, self.im_cols)
            td_flat = TimeDistributed(Flatten(
                data_format=data_format,
                input_shape=input_shape))
        else:
            td_flat = TimeDistributed(Flatten(data_format=data_format))
        self.seq_model.add(td_flat)

    def add_td_im_reshape(self):
        """
        Adds a TimeDistributed reshaping layer to unflatten flattened image
        frames to width x height format. This layer cannot be used as the first
        layer in the model. Image shape information is gathered from the
        instance's attributes.
        """
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
        """
        Adds a TimeDistributed 2D average pooling layer with the given
        hyperparameters. This pools each frame of a time series of image frames.
        This layer cannot be used as the first layer in the model.

        Parameters:
        -----------
            :int/tuple(int) pool_size:      integer or tuple of integers
                                            indicating the size of the pooling
                                            window
            :int/tuple(int) strides:        integer or tuple of integers which
                                            represents stride length to use
                                            during pooling
            :str padding:                   one of 'valid' or 'same' indicating
                                            the style of padding to use during
                                            pooling
            :str data_format:               one of 'channels_first' or
                                            'channels_last' indicating whether
                                            the input shape denotes image bands
                                            before or after image shape
        """
        self.seq_model.add(td_pool)

    def add_n_td_dense(
            self,
            n,
            units,
            activation='tanh',
            kernel_initializer='glorot_uniform',
            l2_amt=0):
        """
        Adds n TimeDistributed dense layers with the given hyperparameters
        to the Sequential model. If the Sequential model has no layers, then
        the input shape of the first layer added is appropriately set using
        image information stored in the instance's attributes. This applies
        a Dense layer to each frame individually.

        Parameters:
        -----------
            :int n:                         number of layers to add
            :int units:                     number of units to use for each
                                            layer
            :str or valid activation:       activation to use in the layers --
                                            can be a custom activation function
                                            or layer or a simple Keras string
        """
        if len(self.seq_model.layers) == 0:
            num_same = n-1
            input_shape = (None, self.im_size)
            self.seq_model.add(TimeDistributed(Dense(
                units=units,
                activation=activation,
                input_shape=input_shape,
                kernel_initializer=kernel_initializer,
                activity_regularizer=regularizers.l2(l2_amt))))
        else:
            num_same = n

        for i in range(num_same):
            dense_layer = TimeDistributed(
                Dense(
                    units=units, 
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    activity_regularizer=regularizers.l2(l2_amt)))
            self.seq_model.add(dense_layer)


    def make_power_error(self, p):
        """
        Method that returns a tensor loss function which is the mean-power-error
        for a specified power. Used to produce custom loss functions for
        model compilation.

        Parameters:
        -----------
            :float p:                       power to use in computing the mean
                                            power error

        Returns:
        --------
            :func(true, pred) p_err:        function which can evaluate the
                                            mean-power-error between the true
                                            and predicted tensors
        """
        def power_error(true, pred, p):
            diff_pow = kb.abs(true-pred)**p
            mean = kb.mean(diff_pow)
            return mean

        p_err = lambda true, pred: power_error(true, pred, p)
        return p_err

    def make_hard_tanh(self, alpha):
        """
        Method that returns an activation function which is the hyperbolic
        tangent with inputs scaled so as to produce a sharper contour. Used
        for creating custom activations with different sensitivities.

        Parameters:
        -----------
            :float alpha:                   float which is the scaling factor
                                            to use for inputs to the tanh
                                            function

        Returns:
        --------
            :func(x) alpha_tanh:            scaled hyperbolic tangent function
                                            on tensor input x
        """
        def scaled_tanh(x, alpha):
            tanh = kb.tanh(alpha*x)
            return tanh
        alpha_tanh = lambda x: scaled_tanh(x, alpha)
        return alpha_tanh

    def make_soft_sigmoid(self, alpha):
        """
        Method that returns an activation function which is the sigmoid with 
        inputs scaled so as to produce a softer contour. Used for creating 
        custom activations with different sensitivities.

        Parameters:
        -----------
            :float alpha:                   float which is the scaling factor
                                            to use for inputs to the tanh
                                            function

        Returns:
        --------
            :func(x) alpha_sigmoid:            scaled hyperbolic tangent function
                                            on tensor input x
        """
        def scaled_sigmoid(x, alpha):
            sigm = kb.sigmoid(alpha*x)
            return sigm
        alpha_sigmoid = lambda x: scaled_sigmoid(x, alpha)
        return alpha_sigmoid

    def make_ssim_loss(
        self, 
        max_val=1):
        def ssim_loss(
            y_true, 
            y_pred, 
            max_val):
            return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val))

        ssim = lambda y_true, y_pred: ssim_loss(y_true, y_pred, max_val)
        return ssim


    def compile(self, loss, optimizer):
        """
        Compiles the Sequential model with the given loss and optimizer.

        Parameters:
        -----------
            :str or valid loss:             a loss function, either custom or
                                            a Keras string
            :str optimizer:                 an optimizer, not all Keras
                                            optimizers are suitable for TCN
        """
        self.seq_model.compile(loss=loss, optimizer=optimizer)

    def fit(self, x_train, y_train, epochs, batch_size=1):
        """
        Fits the Sequential model to the given training data and labels.

        Parameters:
        -----------
            :np.ndarray x_train:            the training dataset
            :np.ndarray y_train:            the labels/output values for the
                                            training dataset
            :int epochs:                    the number of epochs to run for
                                            fitting
            :int batch_size:                the number of samples to be fed
                                            into the model at a time during
                                            fitting
        """
        self.seq_model.fit(
            x_train, 
            y_train, 
            epochs=epochs, 
            batch_size=batch_size)

    def predict(self, test_frames):
        """
        Predicts the next frames from the given testing frames.

        Parameters:
        -----------
            :np.ndarray test_frames:        array of test samples to predict on

        Returns:
        --------
            :np.ndarray preds:              array of predicted values of the
                                            same shape as test_frames
        """
        preds = self.seq_model.predict(test_frames)
        return preds

    def reset_model(self):
        """
        Removes all layers from the Sequential model by resetting the attribute
        to a fresh Sequential model.
        """
        self.seq_model = Sequential()

