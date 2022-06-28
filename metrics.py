import numpy as np

def Rmse(imageA, imageB):
    # img1, img2: [0, 255]
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    # err /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE, the lower the error, the more "similar"    

    err = np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)
    return np.sqrt(err)
