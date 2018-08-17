#!/usr/bin/python

import cv2, os
import sys
import numpy as np
import math
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize
import imutils
import skimage
from sklearn import svm


np.set_printoptions(threshold=np.nan)
np.seterr(divide='ignore', invalid='ignore')


def get_histogram(magnitude, orientation):
	magnitude = magnitude.flatten()
	orientation = orientation.flatten()
	histogram = np.zeros(9, dtype=np.float32)

	for i in xrange(0, 8*8):
		index = np.int(orientation[i]/20)%9
		weight2 = (orientation[i]%20) / 20
		weight1 = 1 - weight2
		
		histogram[(index%9)]   += weight1 * magnitude[i]
		histogram[(index+1)%9] += weight2 * magnitude[i]

	#print np.int16(histogram)
	return histogram

def get_end_points(point, angle, length):
     x, y = point

     endy = int(y + length * math.sin(angle))
     endx = int(x + length * math.cos(angle))

     return endx, endy

def visualize_histogram (histogram, image):
	img = cv2.resize(image,(256, 512), interpolation = cv2.INTER_CUBIC)
	cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	
	for i in xrange(0,histogram.shape[0]):
		for j in xrange(0,histogram.shape[1]):
			histogram_arr = histogram[i,j,:] / np.linalg.norm(histogram[i,j,:], ord=2)

			for h in xrange(0,9):
				begin = (j*32+16,i*32+16)
				end = get_end_points((j*32+16,i*32+16), np.radians(h*20+10), np.int16(histogram_arr[h]*16))
				cv2.line(cimg, pt1=begin, pt2=end, color=(0,0,255), thickness=1)

				new_begin = (begin[0] - (end[0] - begin[0]), begin[1] - (end[1] - begin[1]))
				cv2.line(cimg, pt1=new_begin, pt2=begin, color=(0,0,255), thickness=1)
			
	
	cv2.imshow('norm', cimg)
	cv2.waitKey(0)


#from https://www.pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/
def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image

	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
 
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
 
		# yield the next image in the pyramid
		yield image


def show_image (image, path):
	cimg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
	cv2.putText(cimg,path,(00,20), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,100,255),2,True)
	cv2.imshow('norm', cimg)
	cv2.waitKey(0)


def get_positive_features(file_path):
	#read positive samples
	pos_path = './INRIAPerson/96X160H96/Train/pos'
	pos_paths = [os.path.join(pos_path, f) for f in os.listdir(pos_path)]
	pos_paths.sort()
	pos_features = []


	print "WARNING: Visualizando histograma antes da normalizacao em blocos"

	for pos_path in pos_paths:
		print 'reading {}'.format(pos_path)
		img = cv2.imread(pos_path, 0)
		img = img[16:-16,16:-16]
		#show_image (img, pos_path[39:])

		feature_vector = get_features(img)
		pos_features.append(feature_vector)

	pos_features = np.asarray(pos_features)
	print 'saving pos_features:', pos_features.shape
	np.save(file_path, pos_features)


def get_negative_features(file_path, num_windows):
	#read positive samples
	neg_path = './INRIAPerson/train_64x128_H96/neg'
	neg_paths = [os.path.join(neg_path, f) for f in os.listdir(neg_path)]
	neg_paths.sort()
	neg_features = []

	#print "WARNING: Visualizando histograma antes da normalizacao em blocos"


	for neg_path in neg_paths:
		print 'reading {}'.format(neg_path)
		img = cv2.imread(neg_path, 0)

		for i in xrange(0, num_windows):
			a = int((img.shape[0]-128-1) * np.random.random())
			b = int((img.shape[1]-64-1)  * np.random.random())
			img_window = img[a:a+128,b:b+64]
			#show_image (img, neg_path[39:])

			feature_vector = get_features(img_window)
			neg_features.append(feature_vector)


	neg_features = np.asarray(neg_features)
	print 'saving neg_features:', neg_features.shape
	np.save(file_path, neg_features)


def get_features(img):
	kernelx = np.array([[0,0,0],[-1,0,1],[0,0,0]])
	sx = cv2.filter2D (img, cv2.CV_32F, kernelx)
	
	kernely = np.array([[0,-1,0],[0,0,0],[0,1,0]])
	sy = cv2.filter2D (img, cv2.CV_32F, kernely)

	magnitude = np.sqrt(np.square(sx) + np.square(sy))
	orientation = np.arctan2(sy, sx) + np.pi/2
	orientation[orientation < 0] += np.pi
	orientation[orientation > np.pi] -= np.pi
	orientation = np.degrees(orientation)

	histogram_matrix = np.zeros((16,8,9), dtype=np.float32)
	for i in xrange(0,128,8):
		for j in xrange(0,64,8):
			histogram_matrix[i/8, j/8, :] = get_histogram (magnitude[i:i+8,j:j+8], orientation[i:i+8,j:j+8])

	#visualize_histogram (histogram_matrix, img)

	feature_vector = np.zeros((15,7,36), dtype=np.float32)
	for i in xrange(0,15):
		for j in xrange(0,7):
			block = histogram_matrix[i:i+2,j:j+2,:].flatten()
			block_normalized = block / np.linalg.norm(block, ord=2)
			if np.linalg.norm(block, ord=2) == 0:
				block_normalized[:] = 0
			feature_vector[i,j,:] = block_normalized
			#show_image (img, neg_path[39:])

	return feature_vector.flatten()


def main():
	get_negative_features('neg_features.npy', num_windows=2)
	get_positive_features('pos_features.npy')

	print 'reading neg_features.npy'
	neg_features = np.load('neg_features.npy')
	print neg_features.shape
	print 'reading pos_features.npy'
	pos_features = np.load('pos_features.npy')
	print pos_features.shape
	neg_y = np.zeros(neg_features.shape[0])
	pos_y = np.ones(pos_features.shape[0])
	


	pos_features = pos_features.tolist()
	neg_features = neg_features.tolist()
	features = np.asarray(neg_features + pos_features)

	pos_y = pos_y.tolist()
	neg_y = neg_y.tolist()
	y = np.asarray(neg_y + pos_y)


	print 'fitting svm'
	clf = svm.SVC(kernel='linear', C = 1.0)
	clf.fit(features, y)



	'''
	[-64:-128] etc
	test_img = cv2.imread('./INRIAPerson/test_64x128_H96/neg/00001147.png', 0)
	show_image (test_img, ' ')

	feature_test = get_features(test_img)
	print(clf.predict(np.array([feature_test])))



	test_img = cv2.imread('./INRIAPerson/test_64x128_H96/pos/crop001501c.png', 0)
	show_image (test_img, ' ')
	print(clf.predict(np.array([feature_test])))
	'''




if __name__ == "__main__":
	main()