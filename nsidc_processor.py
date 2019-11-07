# ------------------------------
# NSIDC Sea Ice Index Processing
# ------------------------------
# This script contains some necessary class and method definitions for working
# with the NSIDC Geotiff Image Files. In particular, we need to extract the
# concentration data from the image values, as the image values encode the
# sea ice concentration pixel by pixel.
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
class NSIDCProcessor:
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
            self.default_extent_cmap = load(extent)
        if concentration is not None:
            self.default_concentration_cmap = load(concentration)


    def impute_missing_index_date_info(self, set_this_index=False):
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
        if set_this_index:
            self.image_index = reindex
        return reindex

    def get_rasters(self, image_path):
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
        if self.image_index.index.name is not 'date':
            temp_index = self.image_index.set_index('date')
        else:
            temp_index = self.image_index
        index_entry = temp_index.loc[date_str]
        file_to_load = index_entry.file_name
        file_image_type = index_entry.image_type
        return self.get_rasters(file_to_load)


    def rasters_dimensions(self, rasters):
        """
        Gets the width and height of the specified image raster loaded with
        rasterio.

        Parameters:
        -----------
            :rasterio dataset rasters:  rasterio loaded dataset of an image

        Returns:
        --------
            :tuple(int) dimensions:     tuple of integers which is the size of
                                        the rasterio image           
        """
        return (raster.width, raster.height)

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
            img_band = self.get_rasters(file_name)
            fig = self.make_colored_tiff(img_band)
            fig.savefig(f'{new_image_folder}{image_name}')

    def process_images_keras_conv(self):
        """
        Gets the raster for each image in the instance's index from the images
        folder and yields the array. Assumes that the image index is
        appropriately sorted by date!

        Returns:
        --------
            :gen (width, height) ndarray:   generator object returning the numpy
                                            arrays of the rasters
        """
        for idx, row in self.image_index.iterrows():
            file_name = row['file_name']
            image_type = row['image_type']

            rasters = self.get_rasters(file_name)

            yield self.scale_to_normal(rasters.read(1), image_type)

    def scale_from_normal(self, normed_array, image_type):
        """
        Scales a normalized array into an appropriately valued array based on
        which type of image we desire.
        """
        if image_type == 'extent':
            array = normed_array * self.ext_scale
            return array.astype('uint8')
        elif image_type == 'concentration':
            array = normed_array * self.conc_scale
            return array.astype('uint16')

    def create_raster_from_normal(self, normed_array, image_type):
        """
        Creates a rasterio dataset from a normalized image array and the given
        image type
        """
        pass

    def scale_to_normal(self, array, image_type):
        """
        Scales the given raster array to a shape between 0 and 1 by specifying
        the image type.

        Parameters:
        -----------
            :np.ndarray array:          raster array to scale
            :str image_type:            one of 'extent' or 'concentration' to
                                        specify the appropriate scaling

        Returns:
        --------
            :np.ndarray normed_array:   scaled array to fit between 0 and 1
        """
        if image_type == 'extent':
            return array / self.ext_scale
        elif image_type == 'concentration':
            return array / self.conc_scale
