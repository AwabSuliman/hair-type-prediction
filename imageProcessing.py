
import joblib
from skimage.io import imread
from skimage.transform import resize #we need this resize all images to one specific size
import os
import numpy as np
from skimage.feature import hog

def resize_images(pathDataSet, labels, imageSize=(64,64)):
    '''
    pathDataSet is path to images we will use to train our module
    labels are hair types i want to predicted
    
    '''
    X = [] #contains the images
    Y = [] #contains the labels



    for label in labels: #looping thru each label (each hair type)
        folder = os.path.join(pathDataSet, label) #building the path to one hair type
        for file in os.listdir(folder):
            if file.endswith(('.jpg', '.jpeg', '.png')): #shouldn't be needed but just incase since i haven't checked the data set.
                try:
                    imagePath = os.path.join(folder, file)
                    image = imread(imagePath)
                    # print(f"Reading {imagePath}, shape={image.shape}")
                    if len(image.shape) == 2:  # grayscale
                        image = np.stack((image,)*3, axis=-1)

                    elif image.shape[-1] == 4: #for images w alpha 
                        image = image[...,:3] #to take first three dimension (so that alpha is dropped)

                    imageResize = resize(image, imageSize, anti_aliasing=True)
                    imageResize = np.clip(imageResize, 0, 1)
                    imageResize = imageResize.astype(np.float32)

                    if not np.all(np.isfinite(imageResize)):
                        print(f"Skipping {imagePath}: non-finite values found")
                    


                    if imageResize.shape != (*imageSize, 3):
                        print(f" Skipping {imagePath}: unexpected shape {imageResize.shape}")
                    
                    # hog_features = hog(
                    #     imageResize,
                    #     orientations=9,
                    #     pixels_per_cell=(8, 8),
                    #     cells_per_block=(2, 2),
                    #     channel_axis=-1
                    #         )

                    X.append(imageResize.flatten()) #cause scikitlearn takes vectors so we convert to vectors
                    Y.append(label)  #index of x is supposed to correspond to index of y
                except Exception as e:
                    print(f" skipping {imagePath} {e}")


    X = np.array(X)
    Y = np.array(Y)
    return X, Y

