from fengzhuang_img_jid import *

# idx = 0
# image_list = os.listdir('./qianpai/')

# npy_list = os.listdir('./npy/')
# print(len(image_list))

# for image in image_list:
#     idx += 1
#     # print('{}) image : {}'.format(idx,image))
# for skeleton in npy_list:  
#     idx += 1
#     # print('{}) npy : {}'.format(idx,skeleton))

# for i in range(len(image_list)):
#     print('./qianpai/' + str(i+1) + '.jpg')
test = ssMatch('javaSeatV2.json','npy/9000.npy','qianpai/9000.jpg')
json_data = test.match()
print(json_data)