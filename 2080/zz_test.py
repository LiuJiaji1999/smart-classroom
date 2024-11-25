from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras.applications.vgg16 import VGG16
# from keras import optimizers
from tensorflow import optimizers
import pickle
# from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from keras.models import load_model
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="0"
# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
# sess = tf.compat.v1.Session(config=config)

######################### Step1:  Initialize the parameters and file paths #########################
# configure the parameters
batch_size = 20
num_classes = 8
epochs = 200
image_height = 224
image_width = 224

# model_folder = 'Dangxiao_Action_VGG16_NCNN_Model'
# base_dir = '/data/znk/still_image_test/deep_ensemble_learning/Dangxiao_Dataset/'

model_folder = 'Keda_Action_VGG16_NCNN_Model'
base_dir = '/data/liujiaji/action/kedaclassroom/vggimg/'

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

train_datagen = ImageDataGenerator(
    samplewise_center=True,
    # rescale=1./255,
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip = True,
    fill_mode='nearest'
)
datagen = ImageDataGenerator(
    samplewise_center=True,

    # rescale=1./255
)

def extract_features(directory, sample_count,Channels):
    features = np.zeros(shape=(sample_count, image_height, image_width, Channels),dtype=np.float32)
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False)
    i = 0
    for inputs_batch, labels_batch in generator:
        # features_batch = model.predict(inputs_batch)
        features_batch = inputs_batch
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            # batch_size = 
            break
    return features, labels
# [headDown headUp raiseHand sleep stand takeNote unknown usePhone]
train_data_nums = 1441+1139+26+5+124+223+44+294   #18*3+73+84+37+15
val_data_nums = 710+568+13+3+64+111+22+197     #9+12+25+8+4
train_features, train_labels = extract_features(train_dir, train_data_nums, 3)
train_labels = to_categorical(train_labels)

val_features, val_labels = extract_features(val_dir, val_data_nums, 3)
val_labels = to_categorical(val_labels)
######################### Step2:  Construct the model #########################
base_model = VGG16(weights='imagenet',
                   include_top=False,
                   input_shape=(image_height, image_width, 3))
base_model.summary()

layer_conv_base = base_model.layers

for layers_i in range(len(layer_conv_base)):
    print([layers_i,layer_conv_base[layers_i].name])

# configure the input layer
block5_pool_input = layer_conv_base[18].output
# For block5_pool
# block5_pool_norm = layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(block5_pool_input)

Branch_A = layers.Convolution2D(filters = 128, kernel_size=1,padding='same', activation='relu',strides = 1)(block5_pool_input)

Branch_B = layers.Convolution2D(filters = 128, kernel_size=1,padding='same', activation='relu')(block5_pool_input)
Branch_B = layers.Convolution2D(filters = 128, kernel_size=3,padding='same', activation='relu',strides = 1)(Branch_B)

Branch_C = layers.AveragePooling2D(pool_size=3,padding='same',strides = 1)(block5_pool_input)
Branch_C = layers.Convolution2D(filters = 128, kernel_size=3, padding='same', activation='relu')(Branch_C)

Branches_fusion = layers.concatenate([Branch_A,Branch_B,Branch_C],axis=-1)

# x = Flatten()(x)
# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(Branches_fusion)
# let's add a fully-connected layer
x = Dense(2048, activation='relu')(x)
x = Dense(2048, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(num_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
model.summary()

layer_conv_base = model.layers
for layers_i in range(len(layer_conv_base)):
    print([layers_i,layer_conv_base[layers_i].name])

# for layer in base_model.layers:
#     layer.trainable = False

for layer in model.layers[1:18]:
    layer.trainable = False

# optimizer=optimizers.SGD(lr=0.0001, momentum=0.9)
# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4,decay=1e-6),
              # optimizer=optimizer,
              metrics=['accuracy'])

# checkpoint
checkpoint_dir = os.path.join(os.getcwd(), model_folder, 'checkpoints')
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
filepath_ckp = os.path.join(checkpoint_dir, "weights-best.hdf5")

# save the best model currently
checkpoint = ModelCheckpoint(
    filepath_ckp,
    monitor='val_loss',
    verbose=2,
    save_best_only=True
    )

# fit setup
print('The traning starts!\n')

history = model.fit(
# history = model.fit_generator(
                    train_datagen.flow(train_features, train_labels, batch_size=batch_size),
                    epochs=epochs,
                    validation_data=(val_features, val_labels),
                    steps_per_epoch = train_features.shape[0] // batch_size,
                    callbacks=[checkpoint],
                    # class_weight = class_weight,
                    verbose=2
                    )
######################### Step3:  Save the history data and plots #########################
# plot the acc and loss figure and save the results
plt_dir = os.path.join(os.getcwd(), model_folder, 'plots')
if not os.path.isdir(plt_dir):
    os.makedirs(plt_dir)

print('The ploting starts!\n')
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# save the history
history_dir = os.path.join(os.getcwd(), model_folder, 'history')
if not os.path.isdir(history_dir):
    os.makedirs(history_dir)

# wb 以二进制写入
data_output = open(os.path.join(history_dir,'history_Baseline.pkl'),'wb')
pickle.dump(history.history,data_output)
data_output.close()

# rb 以二进制读取
data_input = open(os.path.join(history_dir,'history_Baseline.pkl'),'rb')
read_data = pickle.load(data_input)
data_input.close()

epochs_range = range(len(acc))
plt.plot(epochs_range, acc, 'ro', label='Training acc')
plt.plot(epochs_range, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig(os.path.join(plt_dir, 'acc.jpg'))
plt.figure()

plt.plot(epochs_range, loss, 'ro', label='Training loss')
plt.plot(epochs_range, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(os.path.join(plt_dir, 'loss.jpg'))
plt.show()
































'''

model_folder_VGG16 = 'Action_VGG16_Model'
base_dir = './Action_ForCNN'
test_dir = os.path.join('./dangxiao_imgs')

datagen = ImageDataGenerator(
    samplewise_center=True,
    # rescale=1./255
)

result_list = []

def extract_features(directory, sample_count,Channels):
    features = np.zeros(shape=(sample_count, image_height, image_width, Channels),dtype=np.float32)
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False)
    global result_list
    result_list = generator.class_indices
    i = 0
    for inputs_batch, labels_batch in generator:
        # features_batch = model.predict(inputs_batch)
        features_batch = inputs_batch
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels

test_features, test_labels = extract_features(test_dir, 2, 3)

checkpoint_dir_VGG16 = os.path.join(os.getcwd(), model_folder_VGG16, 'checkpoints')
# load the best model
# If the error appears 'Error in loading the saved optimizer', it doesn't matter!
best_model_VGG16 = load_model(os.path.join(checkpoint_dir_VGG16, "weights-best.hdf5"))

pred_prob_VGG16 = best_model_VGG16.predict(test_features)
print("pred_prob_VGG16: ", pred_prob_VGG16)
pred_labels_VGG16 = np.argmax(pred_prob_VGG16, axis=1)
print("pred_labels_VGG16: ", pred_labels_VGG16)
# print(pred_prob_VGG16)
print(result_list)
'''