import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os

# 待预测类别
classes = ['raiseHand', 'usePhone','sleep','takeNote','listenClass']


def carve_box(one_skeleton, height, width):
    min_x = min_y = float('inf')  #正无穷
    max_x = max_y = 0 #0
    
    for point in one_skeleton: #25个点 [x,y]
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
        W_Exp = one_skeleton[1][0] - one_skeleton[2][0]
    else:
        W_Exp = 0
    min_y = min_y - H_Exp
    max_y = max_y + H_Exp
    min_x = min_x - W_Exp
    max_x = max_x + W_Exp

    if one_skeleton[8][1] == 0:
        # continue
        pass
    # max_y = one_skeleton[8][1]
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


# def process_img(img_path,cropimg_path):
def process_img(img_path,pose_path):
    img_list = []
    vis_img_point = []
    i = 0
    # pose_path = '/data/liujiaji/action/keda/npy/1.npy'
    pose = np.load(pose_path)
    print(pose.shape)
    for kp in pose:
        i = i+1
        # img = cv2.imread(img_path)
        # min_x, max_x, min_y, max_y = carve_box(kp, img.shape[0], img.shape[1])
        # cropped_img = img[min_y:max_y, min_x:max_x,::-1]
        # cv2.imwrite(cropimg_path+str(i)+'.png',cropped_img)
      
        img = Image.open(img_path)
        min_x, max_x, min_y, max_y = carve_box(kp, img.size[1], img.size[0])
        if max_x - min_x > 40 and max_y - min_y > 40 :
            # print(i,min_x, max_x, min_y, max_y)
            cropped_img = img.crop((min_x,min_y,max_x,max_y))
            
            # cropped_img.save(cropimg_path+str(i)+'.png')

            vis_xx = (min_x + max_x)//2
            vis_yy = (min_y + max_y)//2
            vis_img_point.append([vis_xx,vis_yy])
            img_list.append(cropped_img)
    # print(img_list)
    print(len(img_list))
    # print(vis_img_point)
    return img_list,vis_img_point

# img = '/data/liujiaji/action/kedaclassroom/3C102/3C102_img/3C102_1_1.jpg'
# crop_img = '/data/liujiaji/action/kedaclassroom/3C102/crop_img/'
# process_img(img,crop_img)


def predict_class(img_path, pose_path,model):
    img_list,vis_img_point =  process_img(img_path,pose_path)
    img_cls = []
    # img = Image.open(img_path)
    for i in range(len(img_list)):
        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(img_list[i]).cuda()
        img = torch.unsqueeze(img, dim=0)
        out = model(img)
        # print('i = ',i,'out = ', out)
        pre = torch.max(out, 1)[1]
        cls = classes[pre.item()]
        # print('This is {}!'.format(cls))
        img_cls.append(cls)
    return img_list,vis_img_point,img_cls


def model_struct(num_cls):
    mode1_vgg16 = torchvision.models.vgg16(pretrained=True)
    num_fc = mode1_vgg16.classifier[6].in_features
    mode1_vgg16.classifier[6] = torch.nn.Linear(num_fc, num_cls)
    for param in mode1_vgg16.parameters():
        param.requires_grad = False
    for param in mode1_vgg16.classifier[6].parameters():
        param.requires_grad = True
    mode1_vgg16.to('cuda')
    return mode1_vgg16


def main():
    device = torch.device('cuda')
    model = model_struct(5)
    model.to(device)
    model.eval()
    save = torch.load('/data/liujiaji/action/vgg16log/05-10-15h03m/model.pth') # 希望调用的权重
    model.load_state_dict(save['model'])
    pathlist = os.listdir('/data/liujiaji/action/kedaAlldata/img/')
    # for file in pathlist:
    #     img = '/data/liujiaji/action/kedaAlldata/img/'+file
    #     pose = '/data/liujiaji/action/kedaAlldata/npy/%s.npy' % (os.path.splitext(file)[0])
    img = '/data/liujiaji/action/kedaAlldata/img/5107_11.mp4#t=4.jpg'
    pose = '/data/liujiaji/action/kedaAlldata/npy/5107_11.mp4#t=4.npy'
    # crop_img = '/data/liujiaji/action/keda/crop_img/'
    _,vis_img_point,img_cls= predict_class(img,pose,model)
    print(img_cls)

    img = cv2.imread(img)
    for i in range(len(vis_img_point)):
        # print(tuple(pp[i][0]))
            # if img_cls[i] == 'listenClass':
            #     cv2.putText(img,img_cls[i],tuple(vis_img_point[i]),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            # if img_cls[i] == 'raiseHand':
            #     cv2.putText(img,img_cls[i],tuple(vis_img_point[i]),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            # if img_cls[i] == 'sleep':
            #     cv2.putText(img,img_cls[i],tuple(vis_img_point[i]),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            # if img_cls[i] == 'stand':
            #     cv2.putText(img,img_cls[i],tuple(vis_img_point[i]),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            # if img_cls[i] == 'takeNote':
            #     cv2.putText(img,img_cls[i],tuple(vis_img_point[i]),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            # if img_cls[i]== 'usePhone':
            #     cv2.putText(img,img_cls[i],tuple(vis_img_point[i]),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            # cv2.imwrite('/data/liujiaji/action/kedaAlldata/test/vggvisaction/%s.jpg' % (os.path.splitext(file)[0]),img)

        if img_cls[i] == 'listenClass':
            cv2.putText(img,img_cls[i],tuple(vis_img_point[i]),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        if img_cls[i] == 'raiseHand':
            cv2.putText(img,img_cls[i],tuple(vis_img_point[i]),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        if img_cls[i] == 'sleep':
            cv2.putText(img,img_cls[i],tuple(vis_img_point[i]),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        if img_cls[i] == 'stand':
            cv2.putText(img,img_cls[i],tuple(vis_img_point[i]),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        if img_cls[i] == 'takeNote':
            cv2.putText(img,img_cls[i],tuple(vis_img_point[i]),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        if img_cls[i]== 'usePhone':
            cv2.putText(img,img_cls[i],tuple(vis_img_point[i]),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        cv2.imwrite('/data/liujiaji/action/kedaAlldata/test/vggvisaction/5107_11.mp4#t=4.jpg',img)

if __name__ == '__main__':
    main()
