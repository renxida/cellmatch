# The goal of this script is to convert A SELECTED CHANNEL from nd2 files into opencv-readable videos
from pims import ND2_Reader
import numpy as np
from matplotlib import pyplot as plt
frames = ND2_Reader('./160910SPE6_st_13.5_D/160910SPE6_st_13.5.nd2')
frames.bundle_axes = 'tyx'
frames.iter_axes = 'c'

#%%
import cv2
img1 = c8bit[0]          # queryImage
img2 = c8bit[50]# trainImage
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

out = np.zeros((512,512))

# Draw first 10 matches.
out = cv2.drawMatches(img1,kp1,img2,kp2,matches[:50], out, flags=2)

plt.imshow(out)
plt.savefig(filename = 'match.jpg')