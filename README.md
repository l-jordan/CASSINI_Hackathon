# CASSINI_Hackathon
11th CASSINI Hackathon entry

## File & Folder Description
1. MLP_files: includes final model and the mean and std of the training data
2. feature_engineering: downsamples the labels to ensure fair classification of the MLP
3. MLP_inference: Infers the final desired images using the trained model
4. MLP_train: Training script of the MLP
5. pixel_tabulator: Creates input table for MLP training in .parquet
6. preprocessing: Preprocessing of the Sentinel-2 data for training of MLP
7. training_clipper: Clipped preprocessed indices from Sentinel-2 data to sczle correctly

## Multi-Layer Perceptron (MLP): Greenhouse and vegetation classification 
An MLP was developed and trained on Sentinel-2 data of the Segura River Basin in South Eastern Spain. The development of this MLP is detailed below:

1. Use QGIS to create classes (leafy vegetation, field vegetation, greenhouses, etc.) and export in .GEOJSON
2. Collect Sentinel-2, including one image per month from time series analysis (2021-2023)
3. Calculate desired indices of Sentinel-2 data 
    a. Normalised Difference Vegetation Index (NDVI)
    b. Plastic Greenhouse Index (PGHI)
4. Create set of desired training data: RGB and index images of training months
5. Using the polygons from the classes defined in step 1, extract the labels from the desired Sentinel-2 images. 
6. Split the training set into test, train and validation sets
7. Build a MLP (includes 4 layers, 256 hidden nodes, and has a dropout of 0.2)
8. Generate a confusion matrix and error log
9. Export the model, mean and std of the training data and build an inference script that estimates for your desired images, the classes of the pixels
10. Get final land classification images 


DISCLAIMER: This model currently is overfitted. However, it shows proof-of-concept that Sentinel-2 data can indeed be used to train a classifier to classify different regions of land. It is suggested to increase the amount of polygons and images from Sentinel-2 during training to improve the generalisation of the model.