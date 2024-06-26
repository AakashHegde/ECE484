import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import math
import collections
# from combined_thresh import combined_thresh
# from perspective_transform import perspective_transform

# feel free to adjust the parameters in the code if necessary

def line_fit(binary_warped):
	"""
	Find and fit lane lines
	"""
	# print("Line fit")
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	midpoint = np.int32(histogram.shape[0]/2)
	# print(midpoint)
	# Set the width of the windows +/- margin
	margin = 25
	leftx_base = np.argmax(histogram[margin:midpoint]) + margin
	# print(leftx_base)
	rightx_base = np.argmax(histogram[midpoint:-margin]) + midpoint
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
	
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows-1,0,-1):
			# margin = 150 - (11 * (nwindows - window))
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
			# cv2.imshow("binary_warped", binary_warped)
			# cv2.waitKey(0)
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
	# print(left_lane_inds, right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# print(left_lane_inds)
	# Fit a second order polynomial to each using np.polyfit()
	# If there isn't a good fit, meaning any of leftx, lefty, rightx, and righty are empty,
	# the second order polynomial is unable to be sovled.
	# Thus, it is unable to detect edges.
	try:
	##TODO
			left_fit = np.polyfit(leftx, lefty, 2)
			right_fit = np.polyfit(rightx, righty, 2)
			# print(left_fit, right_fit)
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


def tune_fit(binary_warped, left_fit, right_fit):
	"""
	Given a previously fit line, quickly try to find the line based on previous lines
	"""
	# return(None)
	# print("Tune fit")
	# Assume you now have a new warped binary image
	# from the next frame of video (also called "binary_warped")
	# It's now much easier to find line pixels!
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	margin = 25
	# histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
	# midpoint = np.int32(histogram.shape[0]/2)
	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))# & nonzerox < midpoint)
	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))# & nonzerox > midpoint)

	# Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]
	# print(leftx, rightx)
	# print(leftx)

	# If we don't find enough relevant points, return all None (this means error)
	min_inds = 10
	if lefty.shape[0] < min_inds or righty.shape[0] < min_inds:
		return None

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	# when both line snap together and become one, recalculate fit
	if (left_fit == right_fit).all():
	# if len(np.intersect1d(left_fit, right_fit)) > 15:
		return None
	
	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Return a dict of relevant variables
	ret = {}
	ret['left_fit'] = left_fit
	ret['right_fit'] = right_fit
	ret['nonzerox'] = nonzerox
	ret['nonzeroy'] = nonzeroy
	ret['left_lane_inds'] = left_lane_inds
	ret['right_lane_inds'] = right_lane_inds

	return ret


def viz1(binary_warped, ret, save_file=None):
	"""
	Visualize each sliding window location and predicted lane lines, on binary warped image
	save_file is a string representing where to save the image (if None, then just display)
	"""
	# Grab variables from ret dictionary
	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	out_img = ret['out_img']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	plt.imshow(out_img)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)
	if save_file is None:
		plt.show()
	else:
		plt.savefig(save_file)
	plt.gcf().clear()

waypoint_arr_len = 40
waypoint_arr_x = collections.deque(maxlen=waypoint_arr_len)
waypoint_arr_y = collections.deque(maxlen=waypoint_arr_len)

def draw_waypoint(img, x, y):
	waypoint_arr_x.append(x)
	waypoint_arr_y.append(y)
	avg_x = np.average(waypoint_arr_x)
	avg_y = np.average(waypoint_arr_y)
	cv2.circle(img, (int(avg_x), int(avg_y)), 25, (0, 0, 255), -1)
	return img

def bird_fit(binary_warped, ret, save_file=None):
	"""
	Visualize the predicted lane lines with margin, on binary warped image
	save_file is a string representing where to save the image (if None, then just display)
	"""
	# Grab variables from ret dictionary
	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']

	# Create an image to draw on and an image to show the selection window
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	window_img = np.zeros_like(out_img)
	# Color in left and right line pixels
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Generate a polygon to illustrate the search window area
	# And recast the x and y points into usable format for cv2.fillPoly()
	margin = 25  # NOTE: Keep this in sync with *_fit()
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))

	# Visualization of waypoints
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]
	x = np.average(leftx) + abs(np.average(rightx) - np.average(leftx)) / 2
	y = (np.average(lefty) + np.average(righty)) / 2
	window_img = draw_waypoint(window_img, int(x), int(y))

	# Draw the lane onto the warped blank image
	# cv2.fillPoly(window_img, np.int32([left_line_pts]), (0,255, 0))
	# cv2.fillPoly(window_img, np.int32([right_line_pts]), (0,255, 0))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

	plt.imshow(result)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)

	# cv2.imshow('bird',result)
	# cv2.imwrite('bird_from_cv2.png', result)

	# if save_file is None:
	# 	plt.show()
	# else:
	# 	plt.savefig(save_file)
	# plt.gcf().clear()

	return result


def final_viz(undist, left_fit, right_fit, m_inv):
	"""
	Final lane line prediction visualized and overlayed on top of original image
	"""
	# Generate x and y values for plotting
	ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Create an image to draw the lines on
	#warp_zero = np.zeros_like(warped).astype(np.uint8)
	#color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
	color_warp = np.zeros((720, 1280, 3), dtype='uint8')  # NOTE: Hard-coded image dimensions

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int32([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))
	# Combine the result with the original image
	# Convert arrays to 8 bit for later cv to ros image transfer
	undist = np.array(undist, dtype=np.uint8)
	newwarp = np.array(newwarp, dtype=np.uint8)
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

	return result
