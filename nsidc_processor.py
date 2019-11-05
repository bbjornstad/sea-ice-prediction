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
        else:
            self.image_index = image_index

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
        image_data = rasterio.open(self.image_folder + image_path)
        return image_data

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

    def rasters_index(self, rasters):
        """
        Gets the index of rasters contained in the rasters data loaded with
        rasterio.

        Parameters:
        -----------
            :rasterio dataset rasters:  rasterio loaded dataset of an image

        Returns:
        --------
            :dict(int, dtype) index:    dictionary associating the raster index
                                        to its datatype
        """
        index = {
            i: dtype for i, dtype in zip(rasters.indexes, rasters.dtypes)
        }
        return index

    def get_raster_at_index(self, rasters, i):
        """
        Gets the raster from the rasters at the given index.

        Parameters:
        -----------
            :rasterio dataset rasters:  rasterio loaded dataset of an image
    
        Returns:
        --------
            :np.ndarray raster:         numpy array which is the raster from the
                                        dataset
        """
        raster = rasters.read(i)
        return raster