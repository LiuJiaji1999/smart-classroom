import sys
import sklearn
sys.path.append('/data/liujiaji/action/smartcam')
from linkeaction import SinglePersonSVM



def writeList2txt(file,data):
    '''
    将list写入txt
    :param data:
    :return:
    '''
    file.write(str(data))

def readListFromStr(str):
    '''
    str -> List
    除去冗余的方法调用
    :param str:
    :return:
    '''
    res,pos = help(str,1)
    return res

def help(str,startIndex):
    '''
    单行字符串的读取，形成list
    :param str:
    :return:
    '''
    str = str.replace(" ","") # 将所有空格删去
    res = []
    i = startIndex
    pre = startIndex
    while i <len(str):
        if str[i] == '[':
            # 将pre-i-2的字符都切片，切split
            if i-2>=pre:
                slice = str[pre:i-1].split(',')
                for element in slice:
                    res.append(float(element))
            # 递归调用 加入子list
            child,pos = help(str,i+1)
            res.append(child)
            i = pos # i移动到pos位置，也就是递归的最后一个右括号
            pre = pos + 2 # 右括号之后是, [ 有三个字符，所以要+2至少
        elif str[i] == ']':
            # 将前面的全部放入列表
            if i-1>=pre:
                slice = str[pre:i].split(',')
                for element in slice:
                    res.append(float(element))
            return res,i
        i = i + 1

    return res,i




# SVM训练
action_represent = []
file = open("/data/liujiaji/action/kedaclassroom/highallpoiang_actionDatasets.txt","r")
represent = file.read().splitlines()
for i in range(len(represent)):
    list = readListFromStr(represent[i])
    action_represent.append(list)
# print(action_represent)
print(len(action_represent))

action_label = []
file = open("/data/liujiaji/action/kedaclassroom/highallpoiang_actionDatasetsLabel.txt","r")
action_label = file.read().splitlines()
# print(action_label)
print(len(action_label))


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(action_represent, action_label, random_state=1, train_size=0.6)  #留出法

# # 循环参数，选择最佳
# for C in [0.001,0.01,0.1,1,10,100]:
#     for kernel in ['linear','poly']:
#         model = sklearn.svm.SVC(C=C,gamma='auto',decision_function_shape='ovo',kernel=kernel,degree=3)
#         model.fit(x_train, y_train)
#         print(model.score(x_train, y_train),'C:',C,'kernel:',kernel)
#         print(model.score(x_test, y_test))

model = SinglePersonSVM()
model.train(x_train,y_train,save_path='/data/liujiaji/action/kedaclassroom/kedasvmhigh_weights.pkl')
model.eval(x_train,y_train,report_path='/data/liujiaji/action/kedaclassroom/kedasvmhigh_weights.pkl')
model.eval(x_test,y_test,report_path='/data/liujiaji/action/kedaclassroom/kedasvmhigh_weights.pkl')