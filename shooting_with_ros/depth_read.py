# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


dir_path = "./day/depth"
file_name = "day_depth_0111.npy"
file_path = dir_path + "/" + file_name
print(file_path)

# ### 파일이 png일 경우 읽기
# depth_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
# print(depth_img.dtype)
# plt.imshow(depth_img, cmap='gray')
# plt.colorbar()  # 컬러바 표시 (선택 사항)
# plt.axis('off')  # 축 표시 제거 (선택 사항)
# plt.show()


### 파일이 numpy 배열일 경우 읽기
depth_img = np.load(file_path)
plt.imshow(depth_img)
plt.colorbar()  # 컬러바 표시 (선택 사항)
plt.axis('off')  # 축 표시 제거 (선택 사항)
plt.show()


# depth_img1 = np.load("./0629/depth_01.npy")
# depth_img2 = np.load("./0629/depth_02.npy")
# plt.subplot(1, 2, 1)
# plt.imshow(depth_img1)
# plt.subplot(1,2,2)
# plt.imshow(depth_img2)
# plt.show()

#
