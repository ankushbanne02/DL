
import cv2 
import matplotlib.pyplot as plt
import numpy as np

img= cv2.imread("download.png",cv2.IMREAD_GRAYSCALE)
(h,w)=img.shape

scaling_matrix =np.array([[2,0,0],[0,2,0]],dtype=np.float32)
img_s = cv2.warpAffine(img,scaling_matrix,(2*w,2*h))

transalation_matrix = np.array([[1,0,10],[0,1,10]],dtype=np.float32)
img_t= cv2.warpAffine(img,transalation_matrix,(h,w))
re_matrix=np.array([[1,0,0],[0,-1,h]],dtype=np.float32)
img_re = cv2.warpAffine(img,re_matrix,(h,w))

center =(w//2,h//2)
anagle=45
ro_matrix= cv2.getRotationMatrix2D(center,anagle,scale=1)
img_ro= cv2.warpAffine(img,ro_matrix,(h,w))

sher_matrix=np.array([[1,0.1,0],[0,1,0]],dtype=np.float32)
sher=cv2.warpAffine(img,sher_matrix,(h,w))

#sheared_image = cv2.warpAffine(image, shear_matrix, (w + int(Sx * h), h + int(Sy * w)))
plt.figure(figsize=(15,10))
plt.subplot(1,2,1)
plt.imshow(img,cmap="gray")
plt.ylabel("Y")
plt.subplot(1,2,2)
plt.imshow(sher,cmap="gray")
plt.xlabel("X")
plt.show()


