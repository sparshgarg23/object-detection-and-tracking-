# -*- coding: utf-8 -*-
"""
Created on Mon May  7 14:53:49 2018

@author: voldemort
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from combined_thresh import combined_thresh
from perspective_transform import perspective_transform
def linefit(binary_warped):
    histogram=np.sum(binary_warped[binary_warped.shape[0]//2:,:],axis=0)
    outputimg=(np.dstack((binary_warped,binary_warped,binary_warped))*255).astype('uint8')
    midpt=np.int(histogram.shape[0]/2)
    leftbase=np.argmax(histogram[100:midpt]) + 100
    rightbase=np.argmax(histogram[midpt:-100])+midpt
    numwind=9
    windowh=np.int(binary_warped.shape[0]/numwind)
    nonzero=binary_warped.nonzero()
    nonzeroy=np.array(nonzero[0])
    nonzerox=np.array(nonzero[1])
    leftcurr=leftbase
    rightcurr=rightbase
    margin=100
    minpix=50
    leftidx=[]
    rightidx=[]
    #2.Sliding window check for lanes
    for w in range(numwind):
        windowylow=binary_warped.shape[0]-(w+1)*windowh
        windowyhigh=binary_warped.shape[0]-w*windowh
        windowxleftlow=leftcurr-margin
        windowxlefthigh=leftcurr+margin
        windowxrightlow=rightcurr-margin
        windowxrighthigh=rightcurr+margin
        cv2.rectangle(outputimg,(windowxleftlow,windowylow),(windowxlefthigh,windowyhigh),(0,255,0),2)
        cv2.rectangle(outputimg,(windowxrightlow,windowylow),(windowxrighthigh,windowyhigh),(0,255,0),2)
        good_left_inds = ((nonzeroy >= windowylow) & (nonzeroy < windowyhigh) & (nonzerox >= windowxleftlow) & (nonzerox < windowxlefthigh)).nonzero()[0]
        good_right_inds = ((nonzeroy >= windowylow) & (nonzeroy < windowyhigh) & (nonzerox >= windowxrightlow) & (nonzerox < windowxrighthigh)).nonzero()[0]
        leftidx.append(good_left_inds)
        rightidx.append(good_right_inds)
        if len(good_left_inds)>minpix:
            leftcurr=np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds)>minpix:
            rightcurr=np.int(np.mean(nonzerox[good_right_inds]))
    leftidx=np.concatenate(leftidx)
    rightidx=np.concatenate(rightidx)
    leftx=nonzerox[leftidx]
    rightx=nonzerox[rightidx]
    lefty=nonzeroy[leftidx]
    righty=nonzeroy[rightidx]
    #3. PolyFIT
    leftfit=np.polyfit(lefty,leftx,2)
    rightfit=np.polyfit(righty,rightx,2)
    #return dictionary
    ret={}
    ret['left_fit']=leftfit
    ret['right_fit']=rightfit
    ret['nonzerox']=nonzerox
    ret['nonzeroy']=nonzeroy
    ret['out_img']=outputimg
    ret['left_lane_inds']=leftidx
    ret['right_lane_inds']=rightidx
    return ret
def tunedfit(binary_warped,left_fit,right_fit):
    nonzero=binary_warped.nonzero()
    nonzeroy=np.array(nonzero[0])
    nonzerox=np.array(nonzero[1])
    margin=100
    leftinds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    rightinds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))
    leftx=nonzerox[leftinds]
    lefty=nonzeroy[leftinds]
    rightx=nonzerox[rightinds]
    righty=nonzeroy[rightinds]
    minidx=10
    if lefty.shape[0]<minidx or righty.shape[0]<minidx:
        return None
    leftfit=np.polyfit(lefty,leftx,2)
    rightfit=np.polyfit(righty,rightx,2)
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = leftfit[0]*ploty**2 + leftfit[1]*ploty + leftfit[2]
    right_fitx = rightfit[0]*ploty**2 + rightfit[1]*ploty + rightfit[2]
    ret={}
    ret['left_fit']=leftfit
    ret['right_fit']=rightfit
    ret['nonzerox']=nonzerox
    ret['nonzeroy']=nonzeroy
    ret['left_lane_inds']=leftinds
    ret['right_lane_inds']=rightinds
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


def viz2(binary_warped, ret, save_file=None):
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
	margin = 100  
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
	plt.imshow(result)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)
	if save_file is None:
		plt.show()
	else:
		plt.savefig(save_file)
	plt.gcf().clear()


def calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy):
	"""
	Calculate radius of curvature in meters
	"""
	y_eval = 719  # 720p video/image, so last (lowest on screen) y index is 719

	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meters per pixel in x dimension

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
	# Calculate the new radii of curvature
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
	# Now our radius of curvature is in meters

	return left_curverad, right_curverad


def calc_vehicle_offset(undist, left_fit, right_fit):
	"""
	Calculate vehicle offset from lane center, in meters
	"""
	# Calculate vehicle center offset in pixels
	bottom_y = undist.shape[0] - 1
	bottom_x_left = left_fit[0]*(bottom_y**2) + left_fit[1]*bottom_y + left_fit[2]
	bottom_x_right = right_fit[0]*(bottom_y**2) + right_fit[1]*bottom_y + right_fit[2]
	vehicle_offset = undist.shape[1]/2 - (bottom_x_left + bottom_x_right)/2

	# Convert pixel offset to meters
	xm_per_pix = 3.7/700 # meters per pixel in x dimension
	vehicle_offset *= xm_per_pix

	return vehicle_offset


def final_viz(undist, left_fit, right_fit, m_inv, left_curve, right_curve, vehicle_offset):
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
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))
	# Combine the result with the original image
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

	# Annotate lane curvature values and vehicle offset from center
	avg_curve = (left_curve + right_curve)/2
	label_str = 'Radius of curvature: %.1f m' % avg_curve
	result = cv2.putText(result, label_str, (30,40), 0, 1, (0,0,0), 2, cv2.LINE_AA)

	label_str = 'Vehicle offset from lane center: %.1f m' % vehicle_offset
	result = cv2.putText(result, label_str, (30,70), 0, 1, (0,0,0), 2, cv2.LINE_AA)

	return result
    
    
    
    