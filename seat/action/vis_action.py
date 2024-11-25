import os
import cv2

# data = []
# for line in open("action.txt","r"): #设置文件对象并读取每一行文件
#     data.append(line)               #将每一行文件加入到list中
# for i in range(len(data)):
#     print(data[i][5])


# image_path_list = os.listdir('/data/liujiaji/action/skeleton_16/all_datasets/') #/data/liujiaji/action/action_test/
# image_path_list.sort(key = lambda x : x[:-4])
# for file in image_path_list:
#     img = cv2.imread(file)
#     cv2.putText(img, 'text', (200,100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
#     cv2.imwrite('/data/liujiaji/action/skeleton_16/labelimg/'+file,img)

img = cv2.imread('./skeleton_16/same/c1_back_raisehand_dispersed.png')
print(img)