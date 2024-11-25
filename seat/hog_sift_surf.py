import numpy as np
import cv2
from matplotlib import pyplot as plt

'''1、加载图片'''
imgname1 = './zoom_out/temp4_1.png'
imgname2 = './zoom_in/target4_1.png'

'''2、提取特征点'''
MIN_MATCH_COUNT = 10 #设置最低特征点匹配数量为10
template = cv2.imread(imgname1,1) # queryImage
target = cv2.imread(imgname2,1) # trainImage
# print(template)
# Initiate SIFT and SIRF detector创建sift和surf检测器
surf = cv2.xfeatures2d.SURF_create()
# print('surf keypoint：',surf)
sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT or SURF 返回关键点的信息和描述符
kp1, des1 = surf.detectAndCompute(template,None)
kp2, des2 = surf.detectAndCompute(target,None)
print('descriptor1:', des1.shape, 'descriptor2:', des2.shape)

'''3、特征点匹配
要注意的是，两张图的特征点数量一般情况下是不一样的，
opencv算法里默认用第一张图的特征点来匹配,
所以匹配矩阵的行数与第一张图特征点的行数一致
'''
#创建设置FLANN匹配
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
'''
# knn  K近邻匹配，
# 在匹配的时候选择K个和特征点最相似的点，
# 如果这K个点之间的区别足够大，则选择最相似的那个点作为匹配点，通常选择K = 2。
# KNN匹配也会出现一些误匹配，这时候需要对比第一邻近与第二邻近之间的距离大小，
# 假如 distance_1< (0.5~0.7)*distance_2, 则认为是正确匹配
'''
matches = flann.knnMatch(des1,des2,k=2) 
# print(type(matches), len(matches), matches[0])

# 提取强匹配特征点 
# store all the good matches as per Lower's ratio test.  以最低的比例 保存好的匹配
good = []
#  # 去除错误匹配  因为需要可靠的匹配点，所以舍弃大于0.3的匹配  
for m,n in matches:
    if m.distance < 0.5*n.distance:
        good.append(m)
print(len(good))

'''单应性  
我们之前使用了查询图像，找到其中的一些特征点， 
我们取另外一个训练图像，找到里面的特征，我们找到它们中间最匹配的。
这里我们需要在大场景中用矩形框出匹配的小物体，所以就要计算单应性矩阵，然后做投影变换
简单说就是我们在一组图像里找一个目标的某个部分的位置。
'''
if len(good)>MIN_MATCH_COUNT:
    # 获取关键点的坐标
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2) #要查询图片的 关键点坐标
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2) #训练图片的 关键点坐标

    disp = src_pts - dst_pts  #计算视差
    
    # findHomography 函数是计算变换矩阵
    # 参数cv2.RANSAC是使用RANSAC算法寻找一个最佳单应性矩阵H，即返回值M
    # 返回值：M 为变换矩阵，mask是掩模
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1) #mask可以帮我们去掉不可靠的点
    # print('变换矩阵：',M)
     # ravel方法将数据降维处理，最后并转换成列表格式
    matchesMask = mask.ravel().tolist()  #用于绘图的mask

    print(np.floor(disp[mask.ravel().astype('bool')]).shape)
    # print('视差：',np.floor(disp[mask.ravel().astype('bool')]))      #视差
    # print('左图坐标：',np.floor(src_pts[mask.ravel().astype('bool')]))   #左图坐标点

     # 获取query  template 的图像尺寸
    h,w,dim = template.shape
    # print(h,w,dim)
    # pts是图像 template 的四个顶点
    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    print('寻找图像的四个顶点：',pts)
    # 计算变换后的四个顶点坐标位置
    dst = cv2.perspectiveTransform(pts,M)
    print('变换后的四个顶点坐标位置：',dst)

    # 根据四个顶点坐标位置在target图像画出变换后的边框
    target = cv2.polylines(target,[np.int32(dst)],True,(0,0,255),3, cv2.LINE_AA)

else:
    print( "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None
#显示匹配结果
draw_params = dict(matchColor=(0,255,0), #绿色线画匹配
                   matchesMask=matchesMask,  # draw only inliers
                   flags=2)
result = cv2.drawMatches(template,kp1,target,kp2,good,None,**draw_params)

# plt.imshow(result, 'gray')
plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
plt.show()
cv2.imwrite("./match/match_surf_4_1.jpg", result)