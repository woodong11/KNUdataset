# -- coding: utf-8 --
import cv2
import numpy as np
import matplotlib.pyplot as plt


# depth 이미지일 경우 numpy에서 변환하기
dir_path = "./day/depth"
file_name = "day_depth_0011.npy"
depth_path = dir_path + "/" + file_name
depth_img = np.load(depth_path)
depth_img = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# thermal path
thermal_path = "./day/thermal/day_thermal_0011.png"

# plt로 보여주기
#undistorted_rgb = cv2.cvtColor(undistorted_rgb, cv2.COLOR_BGR2RGB)
left_image = cv2.imread(thermal_path) 
print(left_image.dtype)
right_image = depth_img


fig, axs = plt.subplots(2, 2)

# 첫 번째 subplot에 left_image를 띄움
img1 = axs[0, 0].imshow(left_image, cmap='gray')

# 두 번째 subplot에 right_image를 띄움
img2 = axs[0, 1].imshow(right_image, cmap='gray')

# 세 번째 subplot 초기화 (빈 이미지)
img3 = axs[1, 0].imshow(np.zeros((20, 20)), cmap='gray')

# 네 번째 subplot 초기화 (빈 이미지)
img4 = axs[1, 1].imshow(np.zeros((20, 20)), cmap='gray')

def on_motion(event):
    if event.inaxes == axs[0, 0]:
        x, y = int(event.xdata), int(event.ydata)
        size = 15  # 주변 픽셀을 가져오기 위한 반지름
        region = left_image[max(0, y-size):y+size, max(0, x-size):x+size]  # 주변 픽셀 추출
        img3.set_data(region)  # 세 번째 subplot에 이미지 설정
        img3.set_clim(region.min(), region.max())  # 색상 범위 재설정
        
        region_right = right_image[max(0, y-size):y+size, max(0, x-size):x+size]
        img4.set_data(region_right)  # 네 번째 subplot에 이미지 설정
        img4.set_clim(region_right.min(), region_right.max())  # 색상 범위 재설정
        
        fig.canvas.draw()  # 변경 사항 반영

# 마우스 움직임 이벤트 연결
fig.canvas.mpl_connect('motion_notify_event', on_motion)

plt.show()