import cv2
import numpy as np

def gradient_thresh(img, thresh_min=25, thresh_max=100):
        """
        Apply sobel edge detection on input image in x, y direction
        """

        #1. Convert the image to gray scale
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("grayImg", grayImg)
        #2. Gaussian blur the image
        gaussianImg = cv2.GaussianBlur(grayImg,(5,5),0)
        cv2.imshow("gaussianImg", gaussianImg)
        # #3. Use cv2.Sobel() to find derievatives for both X and Y Axis
        sobelImg = cv2.Sobel(gaussianImg, -1, 1, 1)
        cv2.imshow("sobelImg", sobelImg)
        # #4. Use cv2.addWeighted() to combine the results
        # weightedImg = cv2.addWeighted(img, 1, sobelImg, 1)
        # #5. Convert each pixel to unint8, then apply threshold to get binary image
        # binary_output = cv2.threshold(weightedImg, thresh_min, thresh_max, cv2.THRESH_BINARY)

        cv2.waitKey(0)

        return binary_output

def color_thresh(img, thresh=(100, 255)):
        """
        Convert RGB to HSL and threshold to binary image using S channel
        """
        #1. Convert the image from RGB to HSL
        hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #2. Apply threshold on S channel to get binary image
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        sensitivity = 75
        lower_white = np.array([0,0,255-sensitivity])
        upper_white = np.array([255,sensitivity,255])
        yellowImg = cv2.inRange(hsvImg, lower_yellow, upper_yellow)
        whiteImg = cv2.inRange(hsvImg, lower_white, upper_white)
        binary_output = cv2.add(yellowImg, whiteImg)
        cv2.imshow("binary_output", binary_output)
        #Hint: threshold on H to remove green grass

        cv2.waitKey(0)

        return binary_output


img = cv2.imread('../images/road.jpg')
cv2.imshow("img", img)

# gradient_thresh(img)
color_thresh(img)