import cv2
import os
import numpy as np
from PIL import Image


# count = 0
# frame_num = 30000 #控制显示帧数为前300
# cap = cv2.VideoCapture("./20220210_20340308003337_20340308012632_111517.mp4")
# while cap.isOpened():
#     count += 1
#     flag, frame = cap.read()
#     if flag and count < frame_num:
#         cv2.imshow('frame', frame)
#         # cv2.imwrite("sp/frame_{}.png".format(count), frame)
#         cv2.waitKey(1)
#     else:
#         break
# print(count)
# cap.release()
# cv2.destroyAllWindows()


def video2frame(videos_path,frames_save_path,time_interval):
 
  '''
  :param videos_path: 视频的存放路径
  :param frames_save_path: 视频切分成帧之后图片的保存路径
  :param time_interval: 保存间隔
  :return:
  '''
  vidcap = cv2.VideoCapture(videos_path)
  frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
  fps = vidcap.get(cv2.CAP_PROP_FPS)
  size = (int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
  print(frame_count)
  print(fps)
  print(size)

  success, image = vidcap.read() #第一个参数success 为True 或者False,代表有没有读取到图片；第二个参数 image 表示截取到一帧的图片。
  # print(image.shape)
  count = 0 # 共抽了几帧
  while success:
    success, image = vidcap.read()
    count += 1
    if count % time_interval == 0:
      # cv2.imencode('.jpg', image)[1].tofile(frames_save_path + "/c3_front_singleZ_%d.png" % count)
      cv2.imencode('.jpg', image)[1].tofile(frames_save_path + "%d.jpg" % count)
    # if count == 20:
    #   break
  print(count)

if __name__ == '__main__':
   videos_path = 'ch03_20220311082155.mp4'  #20220210_20340308003337_20340308012632_111517
   frames_save_path = '1-2_ch03/1-2_ch03_'
   time_interval = 1000  #  帧间隔 ，  隔 帧保存一次 ，人眼差不多是 1s24帧
   video2frame(videos_path, frames_save_path, time_interval)
   print('cut sucess!')



# def frame2video(im_dir,video_dir,fps):
 
#     im_list = os.listdir(im_dir)
#     # im_list.sort(key=lambda x: int(x.replace("frame","").split('.')[0]))  #最好再看看图片顺序对不
#     img = Image.open(os.path.join(im_dir,im_list[0]))
#     img_size = img.size #获得图片分辨率，im_dir文件夹下的图片分辨率需要一致
 
 
#     # fourcc = cv2.cv.CV_FOURCC('M','J','P','G') #opencv版本是2
#     fourcc = cv2.VideoWriter_fourcc(*'XVID') #opencv版本是3
#     videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
#     # count = 1
#     for i in im_list:
#         im_name = os.path.join(im_dir+i)
#         frame = cv2.imdecode(np.fromfile(im_name, dtype=np.uint8), -1)
#         videoWriter.write(frame)
#         # count+=1
#         # if (count == 200):
#         #     print(im_name)
#         #     break
#     videoWriter.release()
#     print('finish')
 
# if __name__ == '__main__':
#     im_dir = './hao/'#帧存放路径
#     video_dir = '3a102.avi' #合成视频存放的路径
#     fps = 1 #帧率，每秒钟帧数越多，所显示的动作就会越流畅
#     frame2video(im_dir, video_dir, fps)




# FPS = video_capture.get(cv2.CAP_PROP_FPS) #获取帧率单位帧每秒
# frame_count = capture.get(cv2.CAP_PEOP_FRAME_COUNT) #获取视频帧数

