import json
import os
import re
import numpy as np
# img = cv2.imread('./img/skeda_20223241103504484.jpg')
# origin_size=316
# target_size=224
# left=(origin_size-target_size)/2  
# bottom=(origin_size-target_size)/2
# img_cut=img[left:left+target_size,bottom:bottom+target_size]
pathlist = os.listdir('target')
fileprefixs = []
for path in pathlist:
    if(path.endswith('.json')):
        fileprefix=re.findall(r'(.+?)\.',path)
        fileprefixs.append(fileprefix[0])
print(len(fileprefixs))
for j in range(len(fileprefixs)):
    dict = json.load(open('./target/%s.json' %fileprefixs[j],'r',encoding='utf-8'))
    frame_png = dict['asset']['name']
    # print(frame_png)
    pngprefix=re.findall(r'(.+?)\.',frame_png)
    print(pngprefix[0])

    
    ps = []
    ts = []
    for i in range(len(dict['regions'])):
        tags =  dict['regions'][i]['tags']   #多个标签
        ts.append(tags)

        pointxy = []
        point = dict['regions'][i]['points']  # 该 key 图片下的目标框
        for j in range(len(point)):
            pointxy.append([point[j]['x'],point[j]['y']]) # 该 key 图片下的矩形目标框
        ps.append(pointxy) # 该 key 图片下的 多个目标框
    # print(ps)
    # print(ts)
    # pose = np.load('/data/liujiaji/action/kedaclassroom/5107/npy/%s.npy' %fileprefix[0] )
