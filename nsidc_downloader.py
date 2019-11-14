# ---------------------
# NSIDC Data Downloader
# ---------------------
# This script is a helper to download the required GeoTiff files in the desired
# format from the NSIDC website. In particular, the NSIDC database uses an FTP
# to distribute the image data. As a result we will define some needed tools to
# help automate the data retrieval process and put things into the appropriate
# place.
#
# Dependencies:
# - Pandas (for index generation)
# ---------------------

# ----------------
# Import Libraries
# ----------------
import shutil
import urllib.request as request
from urllib.error import URLError
from contextlib import closing
from datetime import datetime, date, timedelta
from itertools import product
import os
import pandas as pd


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
        :dict month_dict:       dictionary associating integer keys representing
                                months to their formatted strings used in the
                                NSIDC sea ice index database folder structure.
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
        self.month_dict = {
            1: '01_Jan',
            2: '02_Feb',
            3: '03_Mar',
            4: '04_Apr',
            5: '05_May',
            6: '06_Jun',
            7: '07_Jul',
            8: '08_Aug',
            9: '09_Sep',
            10: '10_Oct',
            11: '11_Nov',
            12: '12_Dec'
        }

    def ftp_fetch(self, ftp_sublevel, file_name, results_subfolder):
        """
        Helper function to fetch a file by name from the given ftp_sublevel 
        using urllib.

        Parameters:
        -----------
            :str ftp_sublevel:      string indicating a path to a sublevel of
                                    the NSIDC FTP server
            :str file_name:         string indicating the name of the file to be
                                    downloaded from the given ftp_server
            :str results_subfolder: string indicating a subfolder of the
                                    results_folder to save the downloaded FTP
                                    files.
        """
        try:
            ftp_file_path = self.database_path + ftp_sublevel + file_name
            with closing(request.urlopen(ftp_file_path)) as r:
                save_path = self.results_folder + results_subfolder + file_name
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w+b') as f:
                    shutil.copyfileobj(r, f)
        except URLError:
            print(f'Could not find {ftp_sublevel}{file_name} on the FTP server')

    def generate_index(self):
        """
        Parses the files currently in the results_folder to create a dataframe
        which contains a file_name column as well as date, hemisphere, and
        image_type columns

        Returns:
        --------
            :df index:              pandas dataframe which is an index of the
                                    images as well as some metadata
        """
        file_list = os.walk(self.results_folder)
        df = pd.DataFrame()
        for r, d, f in file_list:
            for file in f:
                # currently only scans for tif files
                if '.tif' in file:
                    file_name = f'{r}/{file}'.replace(self.results_folder, '')
                    
                    if file[0] == 'N':
                        hemisphere = 'north'
                    elif file[0] == 'S':
                        hemisphere == 'south'
                    
                    if 'extent' in file:
                        image_type = 'extent'
                    elif 'concentration' in file:
                        image_type = 'concentration'

                    date_str = file[2:10]
                    parsed_datetime = datetime.strptime(date_str, '%Y%m%d')
                    file_date = parsed_datetime.date()

                    row_to_add = {
                        'date': file_date,
                        'hemisphere': hemisphere,
                        'image_type': image_type,
                        'file_name': file_name
                    }

                    df = df.append(pd.Series(row_to_add, name=file[:-4]))
        
        return df.sort_values(by=['image_type', 'date'])

    def geotiff_name_format(self, hemisphere, date, img_type):
        """
        Helper function that formats the geotiff metadata into the appropriate
        string to fetch from the FTP server

        Parameters:
        -----------
            :str hemisphere:        string indicating whether the desired image
                                    is from 'north' or 'south'
            :str date:              datetime date object representing the date
                                    of the image in the database
            :str img_type:          string indicating whether 'extent' or
                                    'concentration' is desired

        Returns:
        --------
            :str file_name:         returns the properly formatted string stub
                                    that can be used to access the file from the
                                    server
        """
        if hemisphere == 'north':
            hemi = 'N'
        elif hemisphere == 'south':
            hemi = 'S'

        year = str(date.year)
        month = str(date.month)
        day = str(date.day)

        if len(month) == 1:
            month = f'0{month}'
        if len(day) == 1:
            day = f'0{day}'

        date_str = f'{year}{month}{day}'

        file_name = f'{hemi}_{date_str}_{img_type}_v3.0.tif'
        return file_name

    
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
        pass

    def fetch_daily_geotiffs(
        self,
        results_subfolder,
        date_range='all',
        hemispheres='both',
        image_type='both'):
        """
        Gets the daily GeoTiff files from the NSIDC databse for the selected
        range of dates and hemispheres. Saves to the instance's results_folder
        in the subfolder specified by the results_subfolder parameter. This
        folder's structure will be further divided by the years that images were
        taken.

        Parameters:
        -----------
            :str results_subfolder:     string indicating a subfolder of the
                                        results_folder for which to save this
                                        collection of images.
            :tuple(int) date_range:     either the string 'all' or a range of
                                        dates given as a tuple in (YYYY, YYYY 
                                        format). If 'all' all of the years 
                                        present in the database will be
                                        downloaded (default 'all')
            :str hemispheres:           one of ['both', 'north', 'south']
                                        indicating which hemispheres should have
                                        their images downloaded
            :str image_type:            one of ['both', 'extent', 
                                        'concentration'] indicating whether
                                        extent and/or concentration images
                                        should be downloaded (default 'both')
        """
        if date_range == 'all':
            date_start = date(1978, 10, 26)
            today = datetime.today()
            date_end = date(today.year, today.month, today.day)
        else:
            date_start = date(date_range[0], 1, 1)
            if date_range[1] == 'today':
                today = datetime.today()
                date_end = date(today.year, today.month, today.day)
            else:
                date_end = date(date_range[1], 12, 31)

        date_range = [date_start + timedelta(days=i) for i in range((date_end-date_start).days + 1)]

        hemispheres = ['north', 'south'] if hemispheres == 'both' else [hemispheres]
        image_type = ['concentration', 'extent'] if image_type == 'both' else [image_type]

        iterable_params = product(hemispheres, date_range, image_type)

        for p in iterable_params:
            file_name = self.geotiff_name_format(p[0], p[1], p[2])
            formatted_month = self.month_dict[p[1].month]
            year = p[1].year
            ftp_sublevel = f'{p[0]}/daily/geotiff/{year}/{formatted_month}/'
            self.ftp_fetch(
                ftp_sublevel,
                file_name,
                f'{results_subfolder}{year}/')