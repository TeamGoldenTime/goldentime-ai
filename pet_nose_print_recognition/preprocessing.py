import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import os
# import cv2 as cv
from glob import glob
import os
import numpy as np
# from utils.poincare import calculate_singularities
from utils.segmentation import create_segmented_and_variance_images
from utils.normalization import normalize
from utils.gabor_filter import gabor_filter
from utils.frequency import ridge_freq
from utils import orientation
# from utils.crossing_number import calculate_minutiaes
from tqdm import tqdm
from utils.skeletonize import skeletonize

from sklearn.feature_extraction import img_to_graph

def open_images(directory):
    images_paths = glob(directory + '*.tif')
    return np.array([cv2.imread(img_path,0) for img_path in images_paths]), images_paths

def detect_ridges(gray, sigma=3.0):
	H_elems = hessian_matrix(gray, sigma=sigma, order='rc')
	maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
	return maxima_ridges, minima_ridges

def plot_images(*images):
	images = list(images)
	n = len(images)
	fig, ax = plt.subplots(ncols=n, sharey=True)
	for i, img in enumerate(images):
		ax[i].imshow(img, cmap='gray')
		ax[i].axis('off')
		extent = ax[i].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
		plt.savefig('fig'+str(i)+'.png', bbox_inches=extent)
	plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97)
	plt.show()

def remove_back(image_path):
    imgo = cv2.imread(image_path)

    #Removing the background
    height, width = imgo.shape[:2]

    #Create a mask holder
    mask = np.zeros(imgo.shape[:2],np.uint8)

    #Grab Cut the object
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    #Hard Coding the Rect… The object must lie within this rect.
    rect = (10,10,width-30,height-30)
    cv2.grabCut(imgo,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask==2)|(mask==0),0,1).astype("uint8")
    img1 = imgo*mask[:,:,np.newaxis]

    #Get the background
    background = cv2.absdiff(imgo,img1)

    #Change all pixels in the background that are not black to white
    background[np.where((background > [0,0,0]).all(axis = 2))] = [255,255,255]

    #Add the background and the image
    final = background + img1

    file_name = os.path.split(image_path)[-1]
    path = "pet_nose_print_recognition/background/" + file_name
    cv2.imwrite(path,final)
    # return final

# def enhance_image(img, file):
# 	# -------------------------- Step 1: import the image whose background has been removed ----------
# 	# img = cv2.imread("input.jpg",1)

# 	# -------------------------- Step 2: Sharpen the image -------------------------------------------
# 	kernel = np.array([[-1,-1,-1], 
#                    [-1, 9,-1],
#                    [-1,-1,-1]])
# 	sharpened = cv2.filter2D(img, -1, kernel)
# 	# cv2.imshow("sharpened",sharpened)

# 	# --------------------------- Step 3: Grayscale the image------------------------------------------
# 	gray = cv2.cvtColor(sharpened,cv2.COLOR_BGR2GRAY)
# 	# cv2.imshow("gray",gray)

# 	# --------------------------- Step 4: Perform histogram equilisation ------------------------------
# 	hist,bins = np.histogram(gray.flatten(),256,[0,256])
# 	cdf = hist.cumsum()
# 	cdf_normalized = cdf * hist.max()/ cdf.max()

# 	cdf_m = np.ma.masked_equal(cdf,0)
# 	cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
# 	cdf = np.ma.filled(cdf_m,0).astype('uint8')

# 	img2=cdf[gray]
# 	# cv2.imshow("histogram",img2)
# 	# cv2.imwrite('hist.jpeg',img2)

# 	# ----------------------------- Step 5: Ridge detection filter ------------------------------------
# 	#sigma = 2.7
# 	a, b = detect_ridges(img2, sigma=2.7)
# 	plot_images(a, b)

# 	# ----------------------------- Step 6: Convert image to binary image -----------------------------
# 	img = cv2.imread('fig1.png',0)
# 	# img = b # a or b
# 	# cv2.imshow("img",img)
# 	# bg = cv2.dilate(img,np.ones((5,5),dtype=np.uint8))
# 	# bg = cv2.GaussianBlur(bg,(5,5),1)
# 	# # cv2.imshow("bg",bg)
# 	# src_no_bg = 255 - cv2.absdiff(img,bg)
# 	# # cv2.imshow("src_no_bg",src_no_bg)
# 	# ret,thresh = cv2.threshold(src_no_bg,240,255,cv2.THRESH_BINARY)
# 	# # cv2.imshow("threshold",thresh)

# 	# # --------------------------- Step 7: Thinning / Skeletonizing Algorithm ----------------------------
# 	# thinned = cv2.ximgproc.thinning(thresh) # 우수우씨! # a,b로 받으면 문제가 생기는데 왜 그러는지는 나중에 알아보기 우선 지문 인식부터 확인하기!
# 	# # cv2.imshow("thinned",thinned)
# 	file_name = os.path.split(file)[-1]
# 	path = "pet_nose_print_recognition/output/" + file_name
# 	cv2.imwrite(path,img)

# 	# cv2.waitKey(1)
# 	cv2.destroyAllWindows()


def fingerprint_pipline(input_img, file):
	block_size = 16

    # pipe line picture re https://www.cse.iitk.ac.in/users/biometrics/pages/111.JPG
    # normalization -> orientation -> frequency -> mask -> filtering

    # normalization - removes the effects of sensor noise and finger pressure differences.
	normalized_img = normalize(input_img.copy(), float(100), float(100))

    # color threshold
    # threshold_img = normalized_img
    # _, threshold_im = cv.threshold(normalized_img,127,255,cv.THRESH_OTSU)
    # cv.imshow('color_threshold', normalized_img); cv.waitKeyEx()

    # ROI and normalisation
	(segmented_img, normim, mask) = create_segmented_and_variance_images(normalized_img, block_size, 0.2)

    # orientations
	angles = orientation.calculate_angles(normalized_img, W=block_size, smoth=False)
	# orientation_img = orientation.visualize_angles(segmented_img, mask, angles, W=block_size)

    # find the overall frequency of ridges in Wavelet Domain
	freq = ridge_freq(normim, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)

    # create gabor filter and do the actual filterin
	gabor_img = gabor_filter(normim, angles, freq)

    # thinning oor skeletoniz
	thin_image = skeletonize(gabor_img)

	file_name = os.path.split(file)[-1]
	path = "pet_nose_print_recognition/output/" + file_name
	cv2.imwrite(path,thin_image)