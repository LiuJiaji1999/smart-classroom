#-*- coding: utf-8 -*-

import os
import time
import pynvml
import psutil
# # export CUDA_VISIBLE_DEVICES=0
# #耗时
# def count_time(func):
#     def int_time():
#         start_time = time.time()
#         func()
#         over_time = time.time()
#         total_time = over_time - start_time
#         print("程序运行了%.5s秒:" % total_time)
#     return int_time
# #内存占用
# def count_info(func):
#     def float_info():
#         pid = os.getpid()
#         p = psutil.Process(pid)
#         info_start = p.memory_full_info().uss/1024**2
#         func()
#         info_end=p.memory_full_info().uss/1024**2
#         print("程序占用了内存:",str(info_end-info_start),"MB")
#     return float_info

# #显存占用
# def count_gpu(func):
# 	def float_gpu():
# 		pynvml.nvmlInit()
# 		handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 0表示显卡标号
# 		meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
# 		func()
# 		# print(meminfo.total/1024**2) #总的显存大小
# 		# print('已用显存大小:',str(meminfo.used/1024**2),'MB')  #已用显存大小
# 		# print(meminfo.free/1024**2)  #剩余显存大小
# 	return float_gpu

os.environ["CUDA_VISIBLE_DEVICES"]="1"
def fun():
	# python skeleton.py param1 param2 param3 param4
	# param1 : 要读取的图片的路径
	# param2 : 存储results_poseKeypoints文件的路径
	# param3 : 存储Skeleton图片的路径
	# param4 : 存储Box图片的路径
# 	for camera_index in range(2):
# 		for frame_index in range(5):
# #			print(index)
# 			os.system("python /data/workspace/snapshot/model/skeleton/skeleton.py /data/znk/frame_img/cam%s/%s.jpg /data/znk/PoseKeypoints/cam%s/%s /data/znk/skeleton-img/cam%s/%s.png /data/znk/box-img/cam%s/%s.png" % (camera_index+1, frame_index, camera_index+1, frame_index, camera_index+1, frame_index, camera_index+1, frame_index))

	path_list = os.listdir('/data/liujiaji/action/kedaAlldata/test/3C102/img')
	for file in path_list:
		os.system('python /data/workspace/snapshot/model/skeleton/skeleton.py /data/liujiaji/action/kedaAlldata/test/3C102/img/%s.jpg /data/liujiaji/action/kedaAlldata/test/5104_2/npy/%s.npy  /data/liujiaji/action/kedaAlldata/test/3C102/box_ske/%s.jpg /data/liujiaji/action/kedaAlldata/test/3C102/box_ske/%s.png' % (os.path.splitext(file)[0],os.path.splitext(file)[0],os.path.splitext(file)[0],os.path.splitext(file)[0]))

	# os.system('python /data/workspace/snapshot/model/skeleton/skeleton.py /data/liujiaji/action/skeleton_action/img/c3_front_singleZ_700.png /data/liujiaji/action/skeleton_action/npy/c3_front_singleZ_700.npy  /data/liujiaji/action/skeleton_action/ske/c3_front_singleZ_700.jpg /data/liujiaji/skeleton_action/box/c3_front_singleZ_700.jpg')





if __name__ == "__main__":
	# while(1):
	fun()
 