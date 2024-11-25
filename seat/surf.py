# -*- coding: utf-8 -*-

import cv2
import numpy as np
# 读取图片
img = cv2.imread('./image/img.png')
img = cv2.resize(img, dsize=(600, 400))
# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 创建一个SURF对象
surf = cv2.xfeatures2d.SURF_create(20000)
# SIFT对象会使用Hessian算法检测关键点，并且对每个关键点周围的区域计算特征向量。该函数返回关键点的信息和描述符
keypoints, descriptor = surf.detectAndCompute(gray, None)
print(type(keypoints), len(keypoints), keypoints[0])
print(descriptor.shape)
# 在图像上绘制关键点
img = cv2.drawKeypoints(image=img, keypoints=keypoints, outImage=img, color=(255, 0, 255),
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# 显示图像
cv2.imshow('surf_keypoints', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



'''
SURF算法
'''
import cv2
import numpy as np

'''1、加载图片'''
img1 = cv2.imread('./image/cali1.bmp', cv2.IMREAD_GRAYSCALE)
img1 = cv2.resize(img1, dsize=(600, 400))
img2 = cv2.imread('./image/cali2.bmp', cv2.IMREAD_GRAYSCALE)
img2 = cv2.resize(img2, dsize=(600, 400))
image1 = img1.copy()
image2 = img2.copy()

'''2、提取特征点'''
# 创建一个SURF对象
surf = cv2.xfeatures2d.SURF_create(25000)
# SIFT对象会使用Hessian算法检测关键点，并且对每个关键点周围的区域计算特征向量。该函数返回关键点的信息和描述符
keypoints1, descriptor1 = surf.detectAndCompute(image1, None)
keypoints2, descriptor2 = surf.detectAndCompute(image2, None)
print('descriptor1:', descriptor1.shape, 'descriptor2', descriptor2.shape)
# 在图像上绘制关键点
image1 = cv2.drawKeypoints(image=image1, keypoints=keypoints1, outImage=image1, color=(255, 0, 255),
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
image2 = cv2.drawKeypoints(image=image2, keypoints=keypoints2, outImage=image2, color=(255, 0, 255),
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# 显示图像
cv2.imshow('surf_keypoints1', image1)
cv2.imshow('surf_keypoints2', image2)
cv2.waitKey(20)

'''3、特征点匹配'''
matcher = cv2.FlannBasedMatcher()
matchePoints = matcher.match(descriptor1, descriptor2)
print(type(matchePoints), len(matchePoints), matchePoints[0])

# 提取强匹配特征点
minMatch = 1
maxMatch = 0
for i in range(len(matchePoints)):
    if minMatch > matchePoints[i].distance:
        minMatch = matchePoints[i].distance
    if maxMatch < matchePoints[i].distance:
        maxMatch = matchePoints[i].distance
print('最佳匹配值是:', minMatch)
print('最差匹配值是:', maxMatch)

# 获取排雷在前边的几个最优匹配结果
goodMatchePoints = []
for i in range(len(matchePoints)):
    if matchePoints[i].distance < minMatch + (maxMatch - minMatch) / 16:
        goodMatchePoints.append(matchePoints[i])

# 绘制最优匹配点
outImg = None
outImg = cv2.drawMatches(img1, keypoints1, img2, keypoints2, goodMatchePoints, outImg, matchColor=(0, 255, 0),
                         flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
cv2.imshow('matche', outImg)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
SURF算法
'''
import cv2

'''1、加载图片'''
img1 = cv2.imread('./image/match1.jpg')
img1 = cv2.resize(img1, dsize=(600, 400))
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('./image/match2.jpg')
img2 = cv2.resize(img2, dsize=(600, 400))
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
image1 = gray1.copy()
image2 = gray2.copy()

'''2、提取特征点'''
# 创建一个SURF对象
surf = cv2.xfeatures2d.SURF_create(10000)
# SIFT对象会使用Hessian算法检测关键点，并且对每个关键点周围的区域计算特征向量。该函数返回关键点的信息和描述符
keypoints1, descriptor1 = surf.detectAndCompute(image1, None)
keypoints2, descriptor2 = surf.detectAndCompute(image2, None)
print('descriptor1:', descriptor1.shape, 'descriptor2', descriptor2.shape)
# 在图像上绘制关键点
image1 = cv2.drawKeypoints(image=image1, keypoints=keypoints1, outImage=image1, color=(255, 0, 255),
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
image2 = cv2.drawKeypoints(image=image2, keypoints=keypoints2, outImage=image2, color=(255, 0, 255),
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# 显示图像
cv2.imshow('surf_keypoints1', image1)
cv2.imshow('surf_keypoints2', image2)
cv2.waitKey(20)

'''3、特征点匹配'''
matcher = cv2.FlannBasedMatcher()
matchePoints = matcher.match(descriptor1, descriptor2)
print(type(matchePoints), len(matchePoints), matchePoints[0])

# 提取强匹配特征点
minMatch = 1
maxMatch = 0
for i in range(len(matchePoints)):
    if minMatch > matchePoints[i].distance:
        minMatch = matchePoints[i].distance
    if maxMatch < matchePoints[i].distance:
        maxMatch = matchePoints[i].distance
print('最佳匹配值是:', minMatch)
print('最差匹配值是:', maxMatch)

# 获取排雷在前边的几个最优匹配结果
goodMatchePoints = []
for i in range(len(matchePoints)):
    if matchePoints[i].distance < minMatch + (maxMatch - minMatch) / 4:
        goodMatchePoints.append(matchePoints[i])

# 绘制最优匹配点
outImg = None
outImg = cv2.drawMatches(img1, keypoints1, img2, keypoints2, goodMatchePoints, outImg, matchColor=(0, 255, 0),
                         flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
cv2.imshow('matche', outImg)
cv2.waitKey(0)
cv2.destroyAllWindows()