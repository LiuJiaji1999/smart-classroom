import cv2
# import dlib

######################################################################################
#视频按帧显示
count = 0
frame_num = 300 #控制显示帧数为前300
cap = cv2.VideoCapture("D:/alpha/seat/dangxiao_1128_duoyuzhidian2.mp4")
while cap.isOpened():
    count += 1
    flag,frame = cap.read()
    if flag and count < frame_num:
        cv2.imshow('frame',frame)
        cv2.waitKey(1)
    else:
        break
print(count)
cap.release()
cv2.destroyAllWindows()
######################################################################################
#连续帧显示为视频
# for i in range(1,101):#显示300帧
#     img = cv2.imread('data/dtc_frames/1-2-light/frame_{}.png'.format(i))
#     print('reading frame_{}.png'.format(i))
#     cv2.imshow("img",img)
#     cv2.waitKey(5)
# cv2.destroyAllWindows()
######################################################################################
# print(dlib.DLIB_USE_CUDA)
# print(dlib.cuda.get_num_devices())