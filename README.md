# CASSINI_Hackathon
11th CASSINI Hackathon entry


## Multi-Layer Perceptron (MLP): Greenhouse and vegetation classification 
An MLP was developed and trained on Sentinel-2 data of the Segura River Basin in South Eastern Spain. The development of this MLP is detailed below:

1. Use QGIS to create classes (leafy vegetation, field vegetation, greenhouses, etc.) and export in .GEOJSON
2. Collect Sentinel-2, including one image per month from time series analysis (2021-2023)
3. Calculate desired indices of Sentinel-2 data 
    a. Enhanced Vegetation Index (EVI)
    b. Plastic Greenhouse Index (PGHI)
    c. Colour Steel Building Index (CSBI)
4. Create set of desired training data: RGB and index images of training months
5. Using the polygons from the classes defined in step 1, extract the labels from the desired Sentinel-2 images. 
6. Split the training set into test, train and validation sets
7. Build a MLP using pytorch