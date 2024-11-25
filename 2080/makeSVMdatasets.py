import json
from PIL import Image
import re
import numpy as np
import sklearn
import sys
sys.path.append('/data/liujiaji/action/smartcam')
from linkeaction import SinglePersonSVM
from sklearn.metrics import classification_report

# 判断点是否在多边形内
def isInterArea(testPoint,AreaPoint):#testPoint为待测点[x,y]
    LBPoint = AreaPoint[0]#AreaPoint为按顺时针顺序的4个点[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    LTPoint = AreaPoint[1]
    RTPoint = AreaPoint[2]
    RBPoint = AreaPoint[3]
    a = (LTPoint[0]-LBPoint[0])*(testPoint[1]-LBPoint[1])-(LTPoint[1]-LBPoint[1])*(testPoint[0]-LBPoint[0])
    b = (RTPoint[0]-LTPoint[0])*(testPoint[1]-LTPoint[1])-(RTPoint[1]-LTPoint[1])*(testPoint[0]-LTPoint[0])
    c = (RBPoint[0]-RTPoint[0])*(testPoint[1]-RTPoint[1])-(RBPoint[1]-RTPoint[1])*(testPoint[0]-RTPoint[0])
    d = (LBPoint[0]-RBPoint[0])*(testPoint[1]-RBPoint[1])-(LBPoint[1]-RBPoint[1])*(testPoint[0]-RBPoint[0])
    if (a>0 and b>0 and c>0 and d>0) or (a<0 and b<0 and c<0 and d<0):
        return True
    else:
        return False


datasets = []
actionLabels = []
dict = json.load(open('/data/liujiaji/action/5107_1-3-4-5-6-8-9-15.json','r',encoding='utf-8'))
for key,value in  dict.items():
    # 对应的图片
    img= Image.open('/data/liujiaji/action/kedaclassroom/5107/5107_img/'+key)
    fileprefix=re.findall(r'(.+?)\.',key)
    # 对应的骨架点
    pose = np.load('/data/liujiaji/action/kedaclassroom/5107/npy/%s.npy' %fileprefix[0] )
    testpoint = []
    allpose = []
    for i in range(pose.shape[0]):  #（44，25，3）44个人，25个关键点，3维坐标信息x,y,confidence
        x1, y1 = pose[i][1][:2]  #脖子点
        x8, y8 = pose[i][8][:2]  #骨盆点
        xc, yc = (x1+x8)*0.5, (y1+y8)*0.5 #分辨是否在 前排 中心
        testpoint.append([xc,yc]) # 该 key 图片下的 多个人的骨架点

        allpose.append(pose[i])
    # print(len(testpoint))
    ps = []
    ts = []
    for i in range(len(dict[key]['regions'])):
        tags =  dict[key]['regions'][i]['tags']   #多个标签
        ts.append(tags)

        pointxy = []
        point = dict[key]['regions'][i]['points']  # 该 key 图片下的目标框
        for j in range(len(point)):
            pointxy.append([point[j]['x'],point[j]['y']]) # 该 key 图片下的矩形目标框
        ps.append(pointxy) # 该 key 图片下的 多个目标框
        # print(pointxy)
    # print(ps)
    # print(ts)
    # print(len(ps))
    # print(len(ts))
    '''
    5107    front[[0,412],[7,704],[1920,1080],[1920,625]]
            center[[731,342],[1166,367],[751,866],[11,707]]
    '''
    for m in range(len(ps)):
        # print(ps[m])
        for n in range(len(testpoint)):
            # print(testpoint[n])

            flag = isInterArea(testpoint[n],ps[m]) #骨盆点在标注框内

            if flag == True:
                for o in range(len(ts[m])):
                    # print(allpose[n][:8,:2])  # 只要 前8个点的xy坐标

                    # # 6分类
                    # if ts[m][o] != 'headDown' and ts[m][o] != 'headUp':
                    #     datasets.append(allpose[n])
                    #     actionLabels.append(ts[m][o])

                    # 8分类
                    datasets.append(allpose[n])
                    actionLabels.append(ts[m][o])

    # print(key,fileprefix)
print(len(datasets))
print(len(actionLabels))
# print(actionLabels)
datasets_list = []
for i in range(len(datasets)):
    # print(datasets[i].reshape(75).tolist())
    datasets_list.append(datasets[i].reshape(75).tolist())

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(datasets_list, actionLabels, random_state=1, train_size=0.8)
print(np.shape(x_train)) #(1089,75)
print(np.shape(x_test)) #(273,75)
model = SinglePersonSVM()
model.train(x_train,y_train,save_path='/data/liujiaji/action/kedaclassroom/kedasvmpkl/keda8svm_weights.pkl')
model.eval(x_train,y_train,report_path='/data/liujiaji/action/kedaclassroom/kedasvmpkl/keda8svm_weights.pkl')
model.eval(x_test,y_test,report_path='/data/liujiaji/action/kedaclassroom/kedasvmpkl/keda8svm_weights.pkl')
pre_test = []
for i in range(len(x_test)):
    p_test = model.predict([x_test[i]])
    pre_test.append(p_test)

print(classification_report(y_test, pre_test))
# print(confusion_matrix(y_test, pre_test))


'''
# 自选参数
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['poly'], 'C': [1, 10, 100, 1000]}]
scores = ['precision', 'recall']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

# 调用 GridSearchCV，将 SVC(), tuned_parameters, cv=5, 还有 scoring 传递进去，
    clf = sklearn.model_selection.GridSearchCV(sklearn.svm.SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    # 用训练集训练这个学习器 clf
    clf.fit(x_train, y_train)

    print("Best parameters set found on development set:")
    print()

    # 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
    print(clf.best_params_)

    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    # 看一下具体的参数间不同数值的组合后得到的分数是多少
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(x_test)

    # 打印在测试集上的预测结果与真实值的分数
    target_names=['raiseHand', 'sleep', 'stand', 'takeNote', 'unknown', 'usePhone','headDown','headUp']
    print(classification_report(y_true, y_pred,target_names=target_names))

    print()

'''


