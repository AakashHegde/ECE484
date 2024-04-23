#!/usr/bin/env python3

import time
import math
import numpy as np
import cv2
import rospy

from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32, Float32MultiArray, Bool
from skimage import morphology

import collections

class waypoint_type():
    def __init__(self, x=0, y=0, angle=0):
        self.x = x
        self.y = y
        self.angle = angle

class lanenet_detector():
    def __init__(self):

        self.bridge = CvBridge()
        # NOTE
        # Uncomment this line for lane detection of GEM car in Gazebo
        # self.sub_image = rospy.Subscriber('/gem/front_single_camera/front_single_camera/image_raw', Image, self.img_callback, queue_size=1)
        # Uncomment this line for lane detection of videos in rosbag
        # self.sub_image = rospy.Subscriber('camera/image_raw', Image, self.img_callback, queue_size=1)
        # Uncomment this line for GEM car live lane detection
        #self.sub_image = rospy.Subscriber('/zed2/zed_node/rgb_raw/image_raw_color', Image, self.img_callback, queue_size=1)
        self.sub_image = rospy.Subscriber('/front_single_camera/image_raw', Image, self.img_callback, queue_size=1)
        self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)
        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False
        self.hist = True

        self.pub_waypoint = rospy.Publisher("wheels/waypoints", Float32MultiArray, queue_size=1)

        self.waypoint_arr_len = 40
        self.waypoint_arr_x = collections.deque(maxlen=self.waypoint_arr_len)
        self.waypoint_arr_y = collections.deque(maxlen=self.waypoint_arr_len)
        self.prev_coords = [0, 50, 0]

    def img_callback(self, data):

        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()
        mask_image, bird_image = self.detection(raw_img)

        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image.publish(out_img_msg)
            self.pub_bird.publish(out_bird_msg)


    def gradient_thresh(self, img, thresh_min=25, thresh_max=100):
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

        return binary_output


    def color_thresh(self, img, thresh=(100, 255)):
        """
        Convert RGB to HSL and threshold to binary image using S channel
        """
        #1. Convert the image from RGB to HSL
        hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #2. Apply threshold on S channel to get binary image
        #Hint: threshold on H to remove green grass
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        lower_white = np.array([0,0,180]) #formerly 0,0,180
        upper_white = np.array([255,75,255])
        yellowImg = cv2.inRange(hsvImg, lower_yellow, upper_yellow)
        whiteImg = cv2.inRange(hsvImg, lower_white, upper_white)
        binary_output = cv2.add(yellowImg, whiteImg)

        return whiteImg


    def combinedBinaryImage(self, img):
        """
        Get combined binary image from color filter and sobel filter
        """
        #1. Apply sobel filter and color filter on input image
        SobelOutput = self.gradient_thresh(img)
        ColorOutput = self.color_thresh(img)
        #2. Combine the outputs
        ## Here you can use as many methods as you want.
        # binary_output = cv2.add(SobelOutput, ColorOutput)

        ## TODO

        ####

        binaryImage = np.zeros_like(SobelOutput)
        binaryImage[(ColorOutput==255)|(SobelOutput==255)] = 255
        # Remove noise from binary image
        binaryImage = morphology.remove_small_objects(binaryImage, min_size=64, connectivity=2)

        # cv2.imshow("binaryImage", binaryImage)

        return binaryImage

    def perspective_transform(self, img, verbose=False):
        """
        Get bird's eye view from input image
        """
        height, width = img.shape[:2]

        #1. Visually determine 4 source points and 4 destination points
        
        # gazebo
        # pt_A = [width * 0.35, height * 0.55]
        # pt_B = [0, height-1]
        # pt_C = [width - 1, height - 1]
        # pt_D = [width * 0.65, height * 0.55]

        # gazebo 2
        # pt_A = [290, 230] #top left
        # pt_B = [0, 400] #bottom left
        # pt_C = [600, 400] #bottom right
        # pt_D = [327, 230] #top right

        # # rosbag 0056 and 0011
        # pt_A = [width * 0.41, height * 0.55] #top left
        # pt_B = [width*0.20, height-1] #bottom left
        # pt_C = [width*0.75, height - 1] #bottom right
        # pt_D = [width * 0.6, height * 0.55] #top right 

        # rosbag 0484
        # pt_A = [width * 0.41, height * 0.5] #top left
        # pt_B = [width*0.20, height-1] #bottom left
        # pt_C = [width*0.75, height - 1] #bottom right
        # pt_D = [width * 0.6, height * 0.5] #top right

        # final project
        # pt_A = [width * 0.33, height * 0.6] #top left
        # pt_B = [width*0, height-1] #bottom left
        # pt_C = [width*1, height - 1] #bottom right
        # pt_D = [width * 0.66, height * 0.6] #top right

        pt_A = [width * 0.1, height * 0.8] #top left
        pt_B = [width*0, height-1] #bottom left
        pt_C = [width*1, height - 1] #bottom right
        pt_D = [width * 0.9, height * 0.8] #top right 

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


    def detection(self, img):

        binary_img = self.combinedBinaryImage(img)
        img_birdeye, M, Minv = self.perspective_transform(binary_img)
        height, width = img_birdeye.shape[:2]

        if not self.hist:
            # Fit lane without previous result
            ret = line_fit(img_birdeye)
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds']
            right_lane_inds = ret['right_lane_inds']

        else:
            # Fit lane with previous result
            if not self.detected:
                ret = line_fit(img_birdeye)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                    self.detected = True

            else:
                left_fit = self.left_line.get_fit()
                right_fit = self.right_line.get_fit()
                ret = tune_fit(img_birdeye, left_fit, right_fit)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                else:
                    self.detected = False

            # Annotate original image
            bird_fit_img = None
            combine_fit_img = None
            if ret is not None:
                combine_fit_img = final_viz(img, left_fit, right_fit, Minv)
                # Extract left and right line pixel positions
                leftx = nonzerox[left_lane_inds]
                lefty = nonzeroy[left_lane_inds]
                rightx = nonzerox[right_lane_inds]
                righty = nonzeroy[right_lane_inds]

                # x,y - x,y coordinates of the pixel in the image
                # x = leftx[10] + abs(rightx[10] - leftx[10]) / 2
                # y = (lefty[10] + righty[10]) / 2
                tempx = leftx
                if(np.average(leftx) > np.average(rightx)):
                    leftx = rightx
                    rightx = tempx
                x = np.average(leftx) + abs(np.average(rightx) - np.average(leftx))/2
                # y = (np.average(lefty) + np.average(righty))/2
                y = 50 # y does not matter anyway; we just want to turn on time

                self.waypoint_arr_x.append(x)
                self.waypoint_arr_y.append(y)
                self.avg_x = np.average(self.waypoint_arr_x)
                self.avg_y = np.average(self.waypoint_arr_y)
                
                # calculate heading
                bottom_centre = [width/2, height] #[x,y]

                bird_fit_img = bird_fit(img_birdeye, ret, self.avg_x, self.avg_y, save_file=None)

                # corrected x, y - considering origin to be bottom centre of the image
                x = self.avg_x - bottom_centre[0]
                y = bottom_centre[1] - self.avg_y
                # TODO: check heading
                heading = math.atan2(x, y)

                print('waypoint: ',x,y,math.degrees(heading))

                # print(math.degrees(heading))
                #heading = math.atan2(righty[10] - lefty[10], rightx[10] - leftx[10])
                self.prev_coords = [x, y, heading]
                msg = Float32MultiArray()
                msg.data = [x, y, heading]
                self.pub_waypoint.publish(msg) # Publish waypoint
            else:
                print("Unable to detect lanes")
                msg = Float32MultiArray()
                msg.data = self.prev_coords
                self.pub_waypoint.publish(msg) # Publish waypoint

            return combine_fit_img, bird_fit_img


if __name__ == '__main__':
    # init args
    rospy.init_node('lanenet_node', anonymous=True)
    lanenet_detector()
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)
