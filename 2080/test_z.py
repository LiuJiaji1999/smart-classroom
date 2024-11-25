
import cv2
import numpy as np
import recognition
import os
# from keras.models import load_model
from tensorflow.keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"]="2"

# minioclient = MinioClass()
# opper = skeleton.OpenPose()
# model = SinglePersonSVM(weights_path="./smartcam/weights/pose_svm_dangxiao_demo.pkl")
# model = SinglePersonSVM(weights_path="/data/liujiaji/action/svm_weights.pkl")

model_folder = '/data/liujiaji/action/Dangxiao_Action_VGG16_NCNN_Model/'
checkpoint_dir = os.path.join(os.getcwd(), model_folder, 'checkpoints')
model = load_model(os.path.join(checkpoint_dir, 'weights-best.hdf5'))

print(model)

# behavmap={1:'sleep',2:'handUp',3:'headDown'}
# seatmap={'seat1':'student1','seat2':'student2','seat3':'student3'}

idx2act = ["default", "sleep", "raiseHand", "takeNote", "usePhone", "headUp"]
samplingOrder=1

# opper = skeleton.OpenPose()
# img = '/data/liujiaji/action/keda/img/1.png'
# datum = opper.infer(cv2.imread(img))
# print(datum.poseKeypoints.shape)

def carve_box(one_skeleton, height, width):
    min_x = min_y = float('inf')
    max_x = max_y = 0
    for point in one_skeleton: #25个点
        if point[0] == 0 or point[1] == 0:
            continue
        if point[1] < min_y:
            min_y = point[1]
        if point[0] > max_x:
            max_x = point[0]
        if point[1] > max_y:
            max_y = point[1]
        if point[0] < min_x:
            min_x = point[0]
    if one_skeleton[1][1]*one_skeleton[8][1] != 0:
        H_Exp = int((one_skeleton[8][1] - one_skeleton[1][1])/2)
    else:
        H_Exp = 0
    if one_skeleton[1][0]*one_skeleton[2][0] != 0:
        W_Exp = one_skeleton[1][0] -one_skeleton[2][0]
    else:
        W_Exp = 0
    min_y = min_y - H_Exp
    max_y = max_y + H_Exp
    min_x = min_x - W_Exp
    max_x = max_x + W_Exp

    if one_skeleton[8][1] == 0:
        # continue
        pass
    max_y = one_skeleton[8][1]
    if min_x < 0:
        min_x = 0
    if max_x > width:
        max_x = width
    if min_y < 0:
        min_y = 0
    if max_y > height:
        max_y = height
    # return min_x, max_x, min_y, max_y
    return int(min_x), int(max_x), int(min_y), int(max_y)


path_list = os.listdir('/data/liujiaji/action/skeleton_16/all_datasets/') #  /data/liujiaji/action/keda/img
img_list = []
for file in path_list:
    skeleton_img = '/data/liujiaji/action/skeleton_16/all_datasets/%s.png' % (os.path.splitext(file)[0])
    temp_img = cv2.imread(skeleton_img)
    img_list.append(temp_img)

img_array = np.array(img_list)
# print(img_array.shape)
num, img_height, img_width, _ = img_array.shape #(1080, 1920, 3)
print(num, img_height, img_width, _)

for file in path_list:
    pose = np.load('/data/liujiaji/action/skeleton_16/results/%s.npy' % (os.path.splitext(file)[0]))
    for kp in pose:
        # pose: 25x3
        #label = idx2act[model.predict(kp)]
        img = []
        min_x, max_x, min_y, max_y = carve_box(kp, img_height, img_width)
        cropped_img = temp_img[min_y:max_y, min_x:max_x,::-1]
        # print(cropped_img.shape)
        resize_cropped_img = cv2.resize(cropped_img, (224,224), interpolation = cv2.INTER_AREA)
        print(resize_cropped_img.shape) #224,224,3
        img.append(resize_cropped_img[np.newaxis,:])
        c = np.concatenate(img,axis=0)
        print(c.shape)

        
        # behav = model.predict(resize_cropped_img)
        # headUpDown=recognition.HeadMetric(kp)
        # print(headUpDown)



'''
def mycallback(ch, method, properties, body):
    global samplingOrder
    ch.basic_ack(delivery_tag=method.delivery_tag)
    body = body.decode('utf-8')
    print("mycallback body:")
    print(body)
    task = json.loads(body)
    print("mycallback task:")
    print(task)

    
    start_img = time.time()
    seatCfg = dict()
    for cameraCfg in task['camerasConfig']:
        seatCfg[cameraCfg['cameraId']]=cameraCfg['seatConfig']

    for camera in task['CameraList']:
        sleep=0
        handUp=0
        takeNote=0
        usePhone=0
        
        headUp=0
        headDown=0
        inter=0
        default = 0
        #behav_rs = dict()
        output = dict()

        datum = opper.infer(cv2.imread(task['Images'][camera]))
        print(datum.poseKeypoints.shape)
        skeleton_img='/data/SmartEducationImage/skeleton/'+task['Time']+'_'+camera+'.jpg'
        opper.dump(datum,skeleton_img)
        #output['sourceImg']=skeleton_img
        #output['sourceImg']=task['Images'][camera]
        
        miniopath_source='image/source/'+task['Time'] + '_' + camera +'.jpg'
        minioclient.upload(miniopath_source,task['Images'][camera])
        output['sourceImg'] = 'school/'+miniopath_source
        
        
        miniopath_skeleton='image/skeleton/'+task['Time'] + '_' + camera +'.jpg'
        minioclient.upload(miniopath_skeleton,skeleton_img)
        output['skeletonImg'] = 'school/'+miniopath_skeleton
        
        output['faceImg'] = ''
       
        ##save
        np.save('pose',datum.poseKeypoints) 
        ##seat match
        ssdict = dict()
        if seatCfg.__contains__(camera):
            tmpdic=dict()
            tmpdic['seatConfig'] = seatCfg[camera]
            seatCfgJson = json.dumps(tmpdic)
            # print(seatCfgJson)
            ssM = seatMatch.ssMatch(seatCfgJson,datum.poseKeypoints)
            ssdict,absent = ssM.match()
            print(ssdict)

        behavList = []
        index=0
        ##behavior
        temp_img = cv2.imread(skeleton_img)
        img_height, img_width, _ = temp_img.shape
        for kp in datum.poseKeypoints:
            # pose: 25x3
            #label = idx2act[model.predict(kp)]
            min_x, max_x, min_y, max_y = carve_box(kp, img_height, img_width)
            cropped_img = temp_img[min_y:max_y, min_x:max_x,::-1]
            behav = model.predict(cropped_img)
            # behav = model.predict(kp.reshape(75))
            headUpDown=recognition.HeadMetric(kp)
            if headUpDown == 1 and behav != 2:
                behav = 5

            if ssdict.__contains__(index):
                studentBehav = dict()
                studentBehav['studentId'] = ssdict[index]
                studentBehav['behavior']= idx2act[behav]
                if behav == 4:
                    studentBehav['behavior'] = "headDown"
                elif behav == 2:
                    studentBehav['behavior'] = "inter"
                elif behav == 3:
                    # if random.randint(0,2):
                    studentBehav['behavior'] = "inter"
                behavList.append(studentBehav)
                index += 1
            else:
                index += 1
                continue
            
            #for student in absent:
            #    studentAbsent = dict()
            #    studentAbsent['studentId'] = student
            #    studentAbsent['behavior']= 'absent'
            #    behavList.append(studentAbsent)
            
            # ["default", "sleep", "raiseHand", "takeNote", "usePhone", "headUp"]
            if behav == 1:
                sleep += 1
                headDown += 1
            elif behav == 2:
                handUp += 1
                inter += 1
            elif behav == 3:
                takeNote += 1
                inter += 1
                # headDown += 1
            elif behav == 4:
                usePhone += 1
                headDown += 1
            elif behav == 5:
                headUp += 1
            elif behav == 0:
                default += 1
            else:
                pass
        # headDown = sleep+takeNote+usePhone
 
        #random output handup and inter
        # rand_num = random.randint(0, 4)
        # handUp = headUp - rand_num
                
        if handUp<0:
            handUp = 0

        output['taskType'] = task['taskType']
        output['lessonId'] = task['lessonId']
        output['cameraNum'] = task['cameraNum']
        output['cameraId'] = camera
        output['time'] = task['TimeS'] 
        output['samplingOrder'] = samplingOrder
        present = sleep+handUp+takeNote+usePhone+headUp+default
        current_rate = present/(present+len(absent))
        # print("111111111")
        
        if task['is_last'] == True or current_rate <= afterclass_rate:
            output['status'] = "finished"
        else:
            output['status'] = "continue"

        result = dict()
        
        result['present']=present
        result['absent']=len(absent)
        result['sleep']=sleep
        # result['takeNote'] = takeNote
        # result['usePhone'] = usePhone
        # result['raiseHand']=inter
        result['headUp']=headUp
        result['headDown']=headDown
        result['inter'] = inter
        #result['absent'] = len(absent) 

        output['result']=result

        # print("222222222")
        for student in absent:
            studentAbsent = dict()
            studentAbsent['studentId'] = student
            studentAbsent['behavior']= 'absent'
            behavList.append(studentAbsent)
        
        # print("3333333333")

        output['behaviorInfo'] = behavList

        
        end_img = time.time()
        running_time = end_img-start_img
        output['time cost'] = running_time

        # print("444444444")
        output_json = json.dumps(output)
        print(result)
        # print("555555555")
        with open('output.json', 'a+', encoding='utf-8') as file:
            file.write(output_json)
        
        # print("666666666")
        
        print("\n\nResultSaver\n\n")

        #myrabbitmq.RabbitPublisher.run('testAction1', str(output_json), 'ResultSaverExchange', 'ResultSaver')
        #myrabbitmq.RabbitPublisher.run('testAction1', str(output_json), 'pengEx', 'pengQueue')
        # myrabbitmq.RabbitPublisher.run('testAction1', str(output_json), '', 'ResultSaver')
        myrabbitmq.RabbitPublisher.run('testAction1', str(output_json), '', 'copy-ResultSaver')
        
    samplingOrder = samplingOrder + 1
    if task['is_last'] == True:
        samplingOrder=1

if __name__ == '__main__':
   
    start = time.time()
    
    queue = 'subtaskQueue'
    exchange = 'testExchange'
    
    print("test ***** myrabbitmq.RabbitConsumerVideo.run(mycallback,exchange, queue)")
    
    myrabbitmq.RabbitConsumerVideo.run(mycallback,exchange, queue)

    end = time.time()
    print('time cost:  %.5f sec' %(end-start))

'''