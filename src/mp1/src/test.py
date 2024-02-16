import cv2
import numpy as np
from skimage import morphology

def gradient_thresh(img, thresh_min=25, thresh_max=100):
        """
        Apply sobel edge detection on input image in x, y direction
        """

        #1. Convert the image to gray scale
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("grayImg", grayImg)
        #2. Gaussian blur the image
        gaussianImg = cv2.GaussianBlur(grayImg,(5,5),0)
        # cv2.imshow("gaussianImg", gaussianImg)
        # #3. Use cv2.Sobel() to find derievatives for both X and Y Axis
        sobelImg = cv2.Sobel(gaussianImg, -1, 1, 1)
        # cv2.imshow("sobelImg", sobelImg)
        # #4. Use cv2.addWeighted() to combine the results
        weightedImg = cv2.addWeighted(gaussianImg, 1, sobelImg, 5, 0)
        # cv2.imshow("weightedImg", weightedImg)
        # #5. Convert each pixel to unint8, then apply threshold to get binary image
        uint8Img = cv2.convertScaleAbs(weightedImg)
        # cv2.imshow("uint8Img", uint8Img)
        thresh, binary_output = cv2.threshold(uint8Img, 127, 255, cv2.THRESH_BINARY)
        cv2.imshow("binary_output", binary_output)
        binary_output = sobelImg

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
        # cv2.imshow("binary_output", binary_output)
        #Hint: threshold on H to remove green grass

        cv2.waitKey(0)

        return binary_output

def combinedBinaryImage(img):
        """
        Get combined binary image from color filter and sobel filter
        """
        #1. Apply sobel filter and color filter on input image
        SobelOutput = gradient_thresh(img)
        ColorOutput = color_thresh(img)
        #2. Combine the outputs
        ## Here you can use as many methods as you want.
        # binary_output = cv2.add(SobelOutput, ColorOutput)

        ## TODO

        ####

        binaryImage = np.zeros_like(SobelOutput)
        binaryImage[(ColorOutput==1)|(SobelOutput==1)] = 1
        # Remove noise from binary image
        binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'),min_size=50,connectivity=2)

        # cv2.imshow("binaryImage", binaryImage)

        return binaryImage

def perspective_transform(img, verbose=False):
        """
        Get bird's eye view from input image
        """
        height, width = img.shape[:2]

        #1. Visually determine 4 source points and 4 destination points
        pt_A = [width * 0.4, height * 0.63]
        pt_B = [0, height-1]
        pt_C = [width - 1, height - 1]
        pt_D = [width * 0.60, height * 0.63]
        input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
        output_pts = np.float32([[0, 0],
                        [0, height - 1],
                        [width - 1, height - 1],
                        [width - 1, 0]])
        #2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
        M = cv2.getPerspectiveTransform(input_pts,output_pts)
        Minv = np.linalg.inv(M)
        #3. Generate warped image in bird view using cv2.warpPerspective()
        warped_img = cv2.warpPerspective(img,M,(width, height))
        
        cv2.imshow("warped_img", warped_img)
        cv2.waitKey(0)

        return warped_img, M, Minv


img = cv2.imread('../images/road.jpg')
cv2.imshow("img", img)

gradient_thresh(img)
# color_thresh(img)
# combinedBinaryImage(img)
# perspective_transform(img)