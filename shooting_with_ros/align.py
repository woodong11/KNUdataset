# -- coding: utf-8 --

import cv2
import numpy as np
import matplotlib.pyplot as plt

def warpImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]
    temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2), list_of_points_2), axis=0)
    x_min, y_min = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    output_img[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0]] = img1
    return output_img

# RGB camera intrinsic parameters
K_rgb = np.array([[1.95447930e+03, 0.00000000e+00, 2.03419789e+03],
                 [0.00000000e+00, 1.96211093e+03, 1.54255348e+03],
                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

D_rgb = np.array([[ 0.11531545, -0.08222561, -0.0072591 , -0.00503622,  0.01879449]])


# Thermal camera intrinsic parameters
K_thermal = np.array([[412.30837785,   0.        , 315.12693405],
                      [  0.        , 412.31541992, 244.68299369],
                      [  0.        ,   0.        ,   1.        ]])
D_thermal = np.array([[-4.02590909e-01,  2.31041396e-01, -4.89959646e-05, -2.48363343e-04, -8.06806958e-02]])


# Define the rotation matrix (R) and translation vector (T)
# R = np.array([[ 0.89274062, -0.3160481,   0.32113513],
#               [-0.15825411,  0.44737927,  0.88023146],
#               [-0.42186469, -0.83663934,  0.34937774]])

# T = np.array([[ 1296.64402321],
#               [-3915.15802272],
#               [-6116.72314841]])

# extrinsic parameters
R = np.array([[ 0.99994167, -0.00456789, -0.00978681],
            [ 0.00486823,  0.99951101,  0.03088748],
            [ 0.00964093, -0.03093333,  0.99947495]]) 

T = np.array([[33.98420693],
            [49.20347839],
            [16.92335201]])



# Load the RGB image and thermal image



# # intrinsic matrix of the second camera
# K_thermal_inv = np.linalg.inv(K_thermal)

# # Compute the homography
# H = np.dot(K_thermal_inv, np.dot(np.hstack([R, T]), K_rgb))

# # Normalize the homography matrix
# H = H / H[2, 2]

# print("Homography matrix:")
# print(H)


# 좋은것: 0623/14, 0512/12, 0515/05

number = "05"

thermal_img = cv2.imread('./0515/thermal_' +number + '.png')

#inv_thermal_img = thermal_img
inv_thermal_img = 255 - thermal_img
rgb_img = cv2.imread('./0515/color_' + number +'.png')
print("thermal_img.shape: ", thermal_img.shape)
print("rgb_img.shape: ", rgb_img.shape)
# cv2.imshow("thermal", inv_thermal_img)
# cv2.imshow("rgb", rgb_img)

# Undistort RGB image
rgb_h, rgb_w = rgb_img.shape[:2]
new_matrix, roi = cv2.getOptimalNewCameraMatrix(K_rgb, D_rgb, (rgb_w, rgb_h), alpha = 0.01)
undistorted_rgb = cv2.undistort(rgb_img, K_rgb, D_rgb, None, new_matrix)
undistorted_rgb = cv2.resize(undistorted_rgb, (640, 480))
#cv2.imshow('undistorted RGB Image', undistorted_rgb)

# Undistort thermal image
thermal_h, thermal_w = inv_thermal_img.shape[:2]
new_matrix, roi = cv2.getOptimalNewCameraMatrix(K_thermal, D_thermal, (thermal_w, thermal_h), alpha = 0.01)
undistorted_thermal = cv2.undistort(inv_thermal_img, K_thermal, D_thermal, None, new_matrix)
#cv2.imshow('undistorted thermal Image', undistorted_thermal)


# ## 특징점 찾기

# 보다 정확한 결과를 위해 gray scale로 변환
thermal_gray = cv2.cvtColor(undistorted_thermal, cv2.COLOR_BGR2GRAY)
rgb_gray = cv2.cvtColor(undistorted_rgb, cv2.COLOR_BGR2GRAY)
# cv2.imshow('undistorted RGB Image', thermal_gray)
# cv2.imshow('undistorted thermal Image', rgb_gray)
print("thermal_gray.shape: ", thermal_gray.shape)
print("rgb_gray.shape: ", rgb_gray.shape)

# threshold 적용
# thermal_gray[thermal_gray < 88] = 0
# thermal_gray[thermal_gray >= 88] = 255

# rgb_gray[rgb_gray < 61] = 0
# rgb_gray[rgb_gray >= 61] = 255

# plt.imshow(rgb_gray)
# plt.show()


# SIFT descriptor 생성
descriptor = cv2.SIFT_create()
kp_thermal, des_thermal = descriptor.detectAndCompute(undistorted_thermal, None)
kp_rgb, des_rgb = descriptor.detectAndCompute(undistorted_rgb, None)

# 특징점 출력
#cv2.imshow("key_point_left", cv2.drawKeypoints(thermal_gray, kp_thermal, None, (0,0,255)))
#cv2.imshow("key_point_right", cv2.drawKeypoints(rgb_gray, kp_rgb, None, (0,0,255)))


# BFmatcher 객체 생성 및 knn 매칭 (두 이미지간에 특징 일치시키기)
#bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matcher = cv2.DescriptorMatcher_create("BruteForce")
matches = matcher.knnMatch(des_thermal, des_rgb, 2)

# 좋은 매칭점 선별
good_matches = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good_matches.append(m)

# good_matches 시각화(매칭점 연결 전시)
img_good_matches = cv2.drawMatches(thermal_gray, kp_thermal, rgb_gray, kp_rgb, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("Good Matches", img_good_matches)


# 매칭점 좌표 변환
src_pts = np.float32([kp_thermal[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
dst_pts = np.float32([kp_rgb[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
# RANSAC 알고리즘을 이용한 변환 행렬 계산 - 두 이미지 변환 행렬 계산. 임계값은 5.0 이용.

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
# 이미지 수평으로 붙이기
result = warpImages(thermal_gray, rgb_gray, M)

print(M)
# 파노라마 이미지 출력
cv2.imshow("Panorama", result)
#cv2.imwrite("panorama.png", result)

cv2.waitKey(0)
cv2.destroyAllWindows()

