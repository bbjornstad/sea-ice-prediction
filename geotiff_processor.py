# ------------------------------
# NSIDC Sea Ice Index Processing
# ------------------------------
# This script contains some necessary class and method definitions for working
# with the NSIDC Geotiff Image Files. In particular, we need to extract the
# concentration data from the image values, as the image values encode the
# sea ice concentration pixel by pixel.
#
# Dependencies:
# - Pandas
# - Numpy
# - Rasterio (download with conda or pip -- conda preferred)
# - Matplotlib
# ------------------------------

# ----------------
# Import Libraries
# ----------------
import pandas as pd
import numpy as np
import rasterio
import rasterio.plot as rplt
import matplotlib.pyplot as plt
from datetime import datetime, date
from pickle import load

# -----------------
# Class Definitions
# -----------------
class GeotiffProcessor:
    """
    This class instantiates an object that can be used to easily manipulate and
    extract the needed data from the GeoTiff files. Requires rasterio, a python
    package for extraction of special raster images.

    Attributes:
    -----------
        :str image_folder:          path to a folder containing GeoTiff files
        :df or str image_index:     either a Pandas dataframe or a path to a CSV
                                    file which can be read as a Pandas dataframe
                                    containing metadata information about the
                                    images. Must at least contain a column
                                    'file_name' describing the relative path of
                                    each image to the image_folder attribute.
    """
    def __init__(self, image_folder, image_index):
        """
        Initializes a new NSIDCProcessor object.

        Parameters:
        -----------
            :str image_folder:          path to a folder containing GeoTiff
                                        files
            :str or df image_index:     either a Pandas dataframe or a path to a
                                        CSV file which can be read as a Pandas
                                        dataframe containing metadata
                                        information about the images. Must 
                                        have a column 'file_name' describing
                                        the relative path of each image to
                                        the image_folder attribute.
        """
        self.image_folder = image_folder
        if isinstance(image_index, str):
            self.image_index = pd.read_csv(image_index, index_col = 0)
            if 'date' in self.image_index.columns:
                self.image_index.date = pd.to_datetime(
                    self.image_index.date, 
                    infer_datetime_format = True)
        else:
            self.image_index = image_index

        self.conc_scale = 2550
        self.ext_scale = 255
        self.default_extent_cmap = None
        self.default_concentration_cmap = None

    def load_default_colormaps(self, extent=None, concentration=None):
        """
        Loads default raster color mappings from the given paths to pickle
        dictionaries in the same format as rasterio.

        Parameters:
        -----------
            :str extent:                string indicating a pickle path of a
                                        colormap following the NSIDC extent
                                        guidelines (default None)
            :str concentration:         string indicating a pickle path of a
                                        colormap following the NSIDC
                                        concentration guidelines (default None)
        """
        if extent is not None:
            with open(extent, 'rb') as extent_dict:
                self.default_extent_cmap = load(extent_dict)
        if concentration is not None:
            with open(concentration, 'rb') as concentration_dict:
                self.default_concentration_cmap = load(concentration_dict)


    def impute_missing_index_dates(self, set_this_index=False):
        """
        Creates a new index dataframe where new rows have been created for
        missing dates and where the values are taken from the nearest known
        values. This allows for the creation of a uniform time series even if
        the raster data is read from the previous day. Note that this assumes
        that the image_index is indexed by date already.

        Parameters:
        -----------
            :bool set_this_index:       boolean value indicating whether or not
                                        this instance's image index should be
                                        set to the new index.

        Returns:
        --------
            :df reindex:                reindexed dataframe with imputed values
        """
        min_date = min(self.image_index.index)
        max_date = max(self.image_index.index)

        new_idx = pd.date_range(min_date, max_date)
        reindex = self.image_index.reindex(new_idx, method='nearest')
        reindex.index.name = 'date'
        if set_this_index:
            self.image_index = reindex
        return reindex

    def get_bands(self, image_path):
        """
        Uses rasterio to open the raster file at the specified path and returns
        the object.

        Attributes:
        -----------
            :str image_path:            a relative path within the image_folder
                                        to fetch the image from

        Returns:
        --------
            :rasterio dataset:          a dataset of the image loaded with
                                        rasterio
        """
        image_data = rasterio.open(self.image_folder + image_path, 'r')
        return image_data

    def load_by_date(self, date_str):
        """
        Loads an image from the index using an appropriately formatted string
        date in YYYY-MM-DD format. This method will error if the instance does
        not contain an index with a date column in datetime format. Also
        requires an image_type column specifying either extent or concentration
        """
        if self.image_index.index.name != 'date':
            temp_index = self.image_index.set_index('date')
        else:
            temp_index = self.image_index
        index_entry = temp_index.loc[date_str]
        file_to_load = index_entry.file_name
        file_image_type = index_entry.image_type
        return self.get_bands(file_to_load)

    def make_colored_tiff(self, tiff_raster):
        """
        The NSIDC tiff rasters have special colormappings which rasterio seems
        to be unable to handle appropriately. We'll have to do the color
        conversion manually.

        Parameters:
        -----------
            :rasterio dataset tiff_raster:  rasterio loaded dataset of a
                                            concentration image

        Returns:
        --------
            :plt.figure fig:                appropriately colored matplotlib 
                                            figure
        """
        cmap = tiff_raster.colormap(1)
        band = tiff_raster.read(1)
        rgba_vals = [[cmap[i] for i in row] for row in band]
        image_array = np.asarray(rgba_vals)
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(image_array)
        ax.axis('off')
        plt.close()
        return fig

    def show_by_date(self, date_str):
        tiff_data = self.load_by_date(date_str)
        return self.make_colored_tiff(tiff_data)

    def batch_generate_colored_tiffs(self, new_image_folder):
        """
        Generates a colored tiff file in the specified folder for each image in
        the image index.

        Parameters:
        -----------
            :str new_image_folder:      a string indicating a path to a folder
                                        in which to save the images.
        """
        for idx, row in self.image_index.iterrows():
            if self.image_index.index.name == 'date':
                image_date = idx
            else:
                image_date = row['date']
            image_type = row['image_type']
            hemi = row['hemisphere']
            file_name = row['file_name']
            image_name = f'{hemi}_{image_date}_{image_type}.png'
            img_band = self.get_bands(file_name)
            fig = self.make_colored_tiff(img_band)
            fig.savefig(f'{new_image_folder}{image_name}')

    def process_images_channels_first(self):
        """
        Gets the raster for each image in the instance's index from the images
        folder and yields the array. Assumes that the image index is
        appropriately sorted by date. Does not sample the images in any way.

        Returns:
        --------
            :np.ndarray band_array:     an array of the raster bands in channels
                                        first format.
        """
        band_seq = []
        for idx, row in self.image_index.iterrows():
            file_name = row['file_name']

            rasters = self.get_bands(file_name)
            bands = rasters.read().astype(float)
            band_seq.append(bands)
        band_array = np.asarray(band_seq)
        return band_array

    def process_images_channels_first_yearly_sample(self):
        """
        Processes all of the images contained in the instance's image_index in a
        yearly sampled fastion. Note that this method REQUIRES that the
        image_index has a valid datetime index column. In other words, this
        method typically necessitates having a unique image type and hemisphere
        selection.

        This method will divide the raster bands into groups by year (removing
        the last day, New Year's Eve in the case of a leap year) to create
        a collection of yearly time series of image data in a channels first
        fashion.

        Returns:
        --------
            :np.ndarray yearly_array:       an array of shape (n_samples,
                                            n_timesteps, n_channels, image_rows,
                                            image_columns) with the image data
                                            processed channels first.
            :list(int) years:               the associated years for each of the
                                            samples in yearly_array
        """
        yearly_sampling = []
        for y in self.image_index.index.year.unique():
            band_seq = []
            index_subset = self.image_index.loc[self.image_index.index.year == y]
            if index_subset.shape[0] == 366:
                # sorry new year's eve
                index_subset = index_subset.iloc[:-1]
            for idx, row in index_subset.iterrows():
                file_name = row['file_name']
                rasters = self.get_bands(file_name)
                bands = rasters.read().astype(float)
                band_seq.append(bands)
            band_array = np.asarray(band_seq)
            yearly_sampling.append(band_array)
        yearly_array = np.asarray(yearly_sampling)
        return yearly_array, list(self.image_index.index.year.unique())

    def make_colored_prediction_image(self, pred, image_type):
        """
        Makes a colored prediction image from the given prediction frame and
        specified image type (which defines the color mapping to use).

        Parameters:
        -----------
            :np.ndarray pred:               an array corresponding to a single
                                            image frame in channels_first width
                                            x height format.
            :str image_type:                one of 'extent' or 'concentration'
                                            to denote the desired color mapping
                                            to use

        Returns:
        --------
            :figure fig:                    a matplotlib figure containing the
                                            colored image without axes
        """
        if image_type == 'extent':
            cmap = self.default_extent_cmap
            im_array = pred.astype('uint8')
        elif image_type == 'concentration':
            cmap = self.default_concentration_cmap
            im_array = pred.astype('uint16')

        print(im_array)
        print(pred)

        flatten_band = im_array.reshape(
            im_array.shape[-2], im_array.shape[-1])
        #flatten_band[flatten_band > 60000] = 0
        rgba_vals = [[cmap[i] for i in row] for row in flatten_band]
        im_array = np.asarray(rgba_vals)
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(im_array)
        ax.axis('off')
        #plt.close()
        return fig

    def make_side_by_side_image(self, pred, actual, image_type):
        """
        Makes a colored prediction image from the given prediction frame and
        specified image type (which defines the color mapping to use).

        Parameters:
        -----------
            :np.ndarray pred:               an array corresponding to a single
                                            image frame in channels_first width
                                            x height format
            :np.ndarray actual:             an array which is the corresponding
                                            actual image data
            :str image_type:                one of 'extent' or 'concentration'
                                            to denote the desired color mapping
                                            to use

        Returns:
        --------
            :figure fig:                    a matplotlib figure containing the
                                            colored image without axes
        """
        if image_type == 'extent':
            cmap = self.default_extent_cmap
            pred_array = pred.astype('uint8')
            actual_array = actual.astype('uint8')
        elif image_type == 'concentration':
            cmap = self.default_concentration_cmap
            pred_array = pred.astype('uint16')
            actual_array = actual.astype('uint16')

        #print(im_array)
        #print(pred)

        flatten_pred_band = pred_array.reshape(
            pred_array.shape[-2], pred_array.shape[-1])
        flatten_actual_band = actual_array.reshape(
            actual_array.shape[-2], actual_array.shape[-1])

        #flatten_band[flatten_band > 60000] = 0
        pred_rgba_vals = [[cmap[i] for i in row] for row in flatten_pred_band]
        actual_rgba_vals = [[cmap[i] for i in row] for row in flatten_actual_band]
        pred_array = np.asarray(pred_rgba_vals)
        actual_array = np.array(actual_rgba_vals)
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(pred_array)
        ax[0].set_title('Predicted')
        ax[1].imshow(actual_array)
        ax[1].set_title('Actual')
        ax[0].axis('off')
        ax[1].axis('off')
        #plt.close()
        return fig