# ---------------------
# NSIDC Data Downloader
# ---------------------
# This script is a helper to download the required GeoTiff files in the desired
# format from the NSIDC website. In particular, the NSIDC database uses an FTP
# to distribute the image data. As a result we will define some needed tools to
# help automate the data retrieval process and put things into the appropriate
# place.
# ---------------------

# ----------------
# Import Libraries
# ----------------
import shutil
import urllib.request as request
from contextlib import closing
from datetime.datetime import now


# -----------------
# Class Definitions
# -----------------
class NSIDCDownloader:
    """
    Defines a class which can be used to download the required GeoTiff or
    shapefile data from the Sea Ice Index.

    Attributes:
    -----------
        :str results_folder:    string indicating a file path to hold the
                                downloaded data
        :str databse_path:      string indicating the link to the NSIDC database
    """

    def __init__(self, results_folder):
        """
        Initializes a new NSIDCDownloader object.

        Parameters:
        -----------
            :str results_folder:    string indicating a file path to hold the
                                    downloaded data
        """
        self.database_path = 'ftp://sidads.colorado.edu/DATASETS/NOAA/G02135/'
        self.results_folder = results_folder

    def ftp_fetch(self, ftp_sublevel, file_name):
        """
        Helper function to fetch a file by name from the given ftp_sublevel 
        using urllib.

        Parameters:
        -----------
            :str ftp_sublevel:      string indicating a path to a sublevel of
                                    the NSIDC FTP server
            :str file_name:         string indicating the name of the file to be
                                    downloaded from the given ftp_server
        """
        ftp_file_path = self.database_path + ftp_sublevel + file_name
        with closing(request.urlopen(ftp_file_path)) as r:
            save_path = self.results_folder + ftp_sublevel + file_name
            with open(save_path, 'wb') as f:
                shutil.copyfileobj(r, f)
    
    def fetch_monthly_geotiffs(
        self,
        date_range='all',
        hemispheres='both',
        image_type='both'):
        """
        Gets the monthly GeoTiff files from the NSIDC database for the selected
        range of years and hemispheres. Saves to the instance's results_folder.

        Parameters:
        -----------
            :tuple(int) date_range:     either the string 'all' or a range of
                                        years given as a tuple in (YYYY, YYYY)
                                        format. If 'all', all of the years
                                        years present in the database will be
                                        downloaded (default 'all')
            :str hemispheres:           one of ['both', 'north', 'south']
                                        indicating which hemispheres should have
                                        their images downloaded (default 'both')
            :str image_type:            one of ['both', 'extent', 
                                        'concentration'] indicating whether
                                        extent and/or concentration images
                                        should be downloaded (default 'both')
        """
        if date_range == 'all':
            date_start = 1978
            date_end = now().year
        else:
            date_start = date_range[0]
            date_end = date_range[1]

        hemispheres = ['north', 'south'] if hemispheres == 'both' else [hemispheres]
        image_type = ['extent', 'concentration'] if image_type == 'both' else [image_type]

    def fetch_daily_geotiffs(
        self,
        date_range='all',
        hemispheres='both',
        image_type='both'):
        """
        Gets the daily GeoTiff files from the NSIDC databse for the selected
        range of dates and hemispheres. Saves to the instance's results_folder.

        Parameters:
        -----------
            :tuple(int) date_range:     either the string 'all' or a range of
                                        dates given as a tuple in (MMYYYY, 
                                        MMYYYY format). If 'all' all of the
                                        years present in the database will be
                                        downloaded (default 'all')
            :str hemispheres:           one of ['both', 'north', 'south']
                                        indicating which hemispheres should have
                                        their images downloaded
            :str image_type:            one of ['both', 'extent', 
                                        'concentration'] indicating whether
                                        extent and/or concentration images
                                        should be downloaded (default 'both')
        """
        pass

    def fetch_monthly_shapefiles(
        self,
        date_range='all',
        hemispheres='both',
        image_type='both'):
        """
        Gets the monthly shapefiles from the NSIDC database for the  selected 
        range of years and hemispheres. Saves to the instance's results_folder.

        Parameters:
        -----------
            :tuple(int) date_range:     either the string 'all' or a range of
                                        years given as a tuple in (YYYY, YYYY)
                                        format. If 'all', all of the years
                                        years present in the database will be
                                        downloaded (default 'all')
            :str hemispheres:           one of ['both', 'north', 'south']
                                        indicating which hemispheres should have
                                        their images downloaded (default 'both')
            :str image_type:            one of ['both', 'extent', 'median'] 
                                        indicating whether extent and/or median 
                                        images should be downloaded (default 
                                        'both')
        """ 
        pass

    def fetch_daily_shapefiles(
        self,
        date_range='all',
        hemispheres='both',
        image_type='both'):
        """
        Gets the daily shapefiles from the NSIDC databse for the selected range 
        of dates and hemispheres. Saves to the instance's results_folder.

        Parameters:
        -----------
            :tuple(int) date_range:     either the string 'all' or a range of
                                        dates given as a tuple in (MMYYYY, 
                                        MMYYYY format). If 'all' all of the
                                        years present in the database will be
                                        downloaded (default 'all')
            :str hemispheres:           one of ['both', 'north', 'south']
                                        indicating which hemispheres should have
                                        their images downloaded
            :str image_type:            one of ['both', 'extent', 'median'] 
                                        indicating whether extent and/or median 
                                        images should be downloaded (default 
                                        'both')
        """
        pass