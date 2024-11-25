# data 数据集制作

label：三个txt文件，是标的数据集标签，跟图片信息一一对应
skeleton_16:存放的是图片信息和skeleton.py跑出的骨架信息
fengzhuang_youhua.py : 是在本地运行的，直接修改 match_sekeleton_behavior()下的bs数组标签，即可得到诸多[骨骼id,label]

'''
python /data/liujiaji/action/fengzhuang_youhua.py
'''

# action SVM训练

action_test:新的测试集，只有npy文件（供方便打印输出看测试结果的），测试结果保存在action.txt中
action_train:训练数据集，包含 sebe目录和temp_balance目录下的所有文件
            sebe：
                制作的数据集（前期随便抽取视频帧，每个帧中包含的动作都不统一）
            temp_balance：
                制作的数据集（抽取的视频帧，帧中包含的动作相对统一，未制作全是其他动作的数据集，是因为其他标签的数据集中，也含有不少的‘其他’动作）
                数据情况大约为：举手77个，玩手机100个，睡觉98个，记笔记81个 
                <tran_c1_front_XXXX>表示原始c1_front_XXXX图片经过左右翻转后的情况


# 开始训练、测试
'''
环境：base
'''
python /data/workspace/yaojiapeng/dangxiao/test.py
'''



