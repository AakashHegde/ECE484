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
        sobelImgX = cv2.Sobel(gaussianImg, -1, 1, 0)
        sobelImgY = cv2.Sobel(gaussianImg, -1, 0, 1)
        # cv2.imshow("sobelImgX", sobelImgX)
        # cv2.imshow("sobelImgY", sobelImgY)
        # #4. Use cv2.addWeighted() to combine the results
        weightedImg = cv2.addWeighted(sobelImgX, 0.5, sobelImgY, 0.5, 0)
        # cv2.imshow("weightedImg", weightedImg)
        # #5. Convert each pixel to unint8, then apply threshold to get binary image
        uint8Img = cv2.convertScaleAbs(weightedImg)
        # cv2.imshow("uint8Img", uint8Img)
        thresh, binary_output = cv2.threshold(uint8Img, 75, 255, cv2.THRESH_BINARY)
        # cv2.imshow("binary_output", binary_output)

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

        # cv2.waitKey(0)

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
        binaryImage[(ColorOutput==255)|(SobelOutput==255)] = 255
        # Remove noise from binary image
        # binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'),min_size=50,connectivity=2)

        # cv2.imshow("binaryImage", binaryImage)

        return binaryImage

def perspective_transform(img, verbose=False):
        """
        Get bird's eye view from input image
        """
        height, width = img.shape[:2]

        #1. Visually determine 4 source points and 4 destination points
        # pt_A = [width * 0.4, height * 0.63]
        # pt_B = [0, height-1]
        # pt_C = [width - 1, height - 1]
        # pt_D = [width * 0.60, height * 0.63]
        pt_A = [width * 0.35, height * 0.55]
        pt_B = [0, height-1]
        pt_C = [width - 1, height - 1]
        pt_D = [width * 0.65, height * 0.55]
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
        
        # cv2.imshow("warped_img", warped_img)
        # cv2.waitKey(0)

        return warped_img, M, Minv

def line_fit(binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
        midpoint = np.int32(histogram.shape[0]/2)
        # print(midpoint)
        leftx_base = np.argmax(histogram[100:midpoint]) + 100
        # print(leftx_base)
        rightx_base = np.argmax(histogram[midpoint:-100]) + midpoint
        # print(rightx_base)

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int32(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        # print(nonzero[0])
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 75
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows-1,0,-1):
                # Identify window boundaries in x and y (and right and left)
                ##TO DO
                high_y = window_height * window
                low_y = window_height * (window + 1) - 1
                leftx_low = leftx_current - margin
                leftx_high = leftx_current + margin
                rightx_low = rightx_current - margin
                rightx_high = rightx_current + margin
                ####
                # Draw the windows on the visualization image using cv2.rectangle()
                ##TO DO
                cv2.rectangle(binary_warped, (leftx_low, high_y), (leftx_high, low_y), 255)
                cv2.rectangle(binary_warped, (rightx_low, high_y), (rightx_high, low_y), 255)
                cv2.imshow("binary_warped", binary_warped)
                cv2.waitKey(0)
                ####
                # Identify the nonzero pixels in x and y within the window
                ##TO DO
                nonzero_left_lane = ((nonzeroy <= low_y) & (nonzeroy >= high_y) & (nonzerox >= leftx_low) & (nonzerox < leftx_high)).nonzero()[0]
                nonzero_right_lane = ((nonzeroy <= low_y) & (nonzeroy >= high_y) & (nonzerox >= rightx_low) & (nonzerox < rightx_high)).nonzero()[0]
                ####
                # Append these indices to the lists
                ##TO DO
                # print(np.shape(nonzero_left_lane))
                left_lane_inds.append(nonzero_left_lane)
                right_lane_inds.append(nonzero_right_lane)
                ####
                # If you found > minpix pixels, recenter next window on their mean position
                ##TO DO
                # print(len(nonzero_left_lane))
                if len(nonzero_left_lane) > minpix:
                        leftx_current = np.int32(np.mean(nonzerox[nonzero_left_lane]))
                if len(nonzero_right_lane) > minpix:        
                        rightx_current = np.int32(np.mean(nonzerox[nonzero_right_lane]))
                ####
                pass

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        # print(left_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each using np.polyfit()
        # If there isn't a good fit, meaning any of leftx, lefty, rightx, and righty are empty,
        # the second order polynomial is unable to be sovled.
        # Thus, it is unable to detect edges.
        try:
        ##TODO
                left_fit = np.polyfit(leftx, lefty, 2)
                right_fit = np.polyfit(rightx, righty, 2)
        ####
        except TypeError:
                print("Unable to detect lanes")
                return None


        # Return a dict of relevant variables
        ret = {}
        ret['left_fit'] = left_fit
        ret['right_fit'] = right_fit
        ret['nonzerox'] = nonzerox
        ret['nonzeroy'] = nonzeroy
        ret['out_img'] = out_img
        ret['left_lane_inds'] = left_lane_inds
        ret['right_lane_inds'] = right_lane_inds

        return ret
        
def detection(img):
        binary_img = combinedBinaryImage(img)
        img_birdeye, M, Minv = perspective_transform(binary_img)

        # cv2.imshow("img_birdeye", img_birdeye)
        # cv2.waitKey(0)

        ret = line_fit(img_birdeye)

img = cv2.imread('test.png')
# img = cv2.imread('../images/road.jpg')
# cv2.imshow("img", img)

# gradient_thresh(img)
# color_thresh(img)
# combinedBinaryImage(img)
# perspective_transform(img)
detection(img)