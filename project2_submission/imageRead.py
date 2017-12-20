import cv2
import numpy as np
from PIL import Image
import Image
img = cv2.imread("testSample/img_8.jpg", cv2.IMREAD_GRAYSCALE)
print type(img)
# img = np.invert(img)
print img.shape
cv2.imwrite("test19.jpg", img)
img2 = cv2.imread("testSample/img_67.jpg", cv2.IMREAD_GRAYSCALE)
# img2 = np.invert(img2)
cv2.imwrite("test20.jpg", img2)


# print np.divide(img, 255.0)

bg = Image.open("test19.jpg")
ov = Image.open("test20.jpg")
new_img = Image.blend(bg, ov, 0.5)
new_img.save("overlaptest10.jpg")
