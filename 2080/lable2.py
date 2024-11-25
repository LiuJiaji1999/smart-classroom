import json
import numpy as np
import json
import cv2
import os
import matplotlib.pyplot as plt


path = '/data/liujiaji/action/kedaAlldata/5107_11-export.json' # 5107 json

#图片目录
dir_path = '/data/liujiaji/action/kedaAlldata/img/'
imageName_list = os.listdir(dir_path)
imagePath_list = [os.path.join(dir_path, imageName) for imageName in imageName_list]
# print(imageName_list)#仅含文件名无路径

t = {}
index=238
for filename in imageName_list:
        with open(path,'r') as f:  # 转换json文件内容为python中的列表或字典，根据自己的json文件来决定
            data = json.load(f)

        for k in data.get('assets').keys() :
            #print(data.get('assets')[k]['asset'].get('name'))
            temp_dict = {}
            submit = '/data/liujiaji/action/kedaAlldata/keda.json'
            if(filename in data.get('assets')[k]['asset'].get('name')):
                temp_dict=data.get('assets')[k]
                #print(temp_dict)
                t[index]=temp_dict
                index=index+1
                print(index)

        
with open(submit, 'a') as f:
    f.write(json.dumps(t))
print('finish merge json')
 





