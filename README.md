# Sea Ice Prediction
As a capstone project for my time in the Data Science Immersive program at the Chicago branch of the Flatiron School, I decided to try my hand at prediction of the extent and concentration of sea ice into the future. I have a strong urge to use the tools of data science and mathematics to help us predict the future health of our planet, and one of the most salient litmus tests we have is the health of the sea ice.

## The Data
The data for this project comes directly from the National Snow and Ice Data Center ([website](https://nsidc.org)). In particular they have a data set called the [Sea Ice Index](https://nsidc.org/data/G02135/versions/3) which is an overview of the extent and concentration of sea ice around the world since 1978. The data are available as CSV files indicating extent in aggregate or as various different geographical files such as ESRI Shapefiles, GeoTiff files, or raw images. The data is available in both daily and monthly resolutions. However, the NSIDC website indicates that the monthly version is the best data set to use for observing long term trends in the extent of sea ice. For now, we will be working with this version, although the exact data requirements for the implementation are still unclear, and more images may be required.

## Project Goals
This project aims to:
- Predict from previous image data the future boundary of sea ice
- Predict from previous image data the future concentration of sea ice
- Create interactive displays to see the progression of the prediction into future years
- Create an interactive system to see when certain conditions are met by sea ice in the predictive models.

## Project Structure
This project is broken up roughly into the following components:
- NSIDC Data Downloader: this script defines tools that can be used to download the needed GeoTiff raster files from the NSIDC FTP server
- Feature Extraction Classes: these classes define tools that are used to extract the relevant information from GeoTiff files and preprocess this for use in our modeling.
- Modeling Notebooks: depending on how things pan out, there will be multiple jupyter notebooks that will make use of the aforementioned classes and scripts to import and process data, as well as perform the overall modeling.

## Model Overview
At start, the base model I will attempt to implement will be a simple SARIMAX model. We can definitely assume that this data is seasonal, and so this will be a good starting point for our time series analysis and future predictions. However, I would eventually like to make use of a Recurrent Neural Network to create a time series forecasting that includes the ability to generate boundary and concentration images.

## Links
- [NSIDC Website](https://nsidc.org)
- [Sea Ice Index](https://nsidc.org/data/G02135/versions/3)
- [About Sea Ice](https://nsidc.org/cryosphere/seaice/index.html)
