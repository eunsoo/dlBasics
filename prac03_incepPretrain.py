# coding: utf-8
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 100
nb_validation_samples = 40
epochs = 5
batch_size = 16

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 2 classes
predictions = Dense(2, activation='softmax')(x)

# # this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# # first: train only the top layers (which were randomly initialized)
# # i.e. freeze all convolutional InceptionV3 layers
# ####################### Problem 1 #################################
# ############################ Your Code ############################
# #################### None 부분을 채우세요 ############################
# """
#     for 문을 이용해서 위의 base model, 즉 GoogLeNet의 상단을 트레이닝하지 않고 고정 시킵니다.
#     "None"을 채우세요.
# """
for layer in base_model.layers:
    "None"
# ############################ End of Your Code ############################

 # compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

train_datagen = ImageDataGenerator(
     rescale=1. / 255,
     shear_range=0.2,
     zoom_range=0.2,
     horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

 # 학습용 제너레이터 설정
train_generator = train_datagen.flow_from_directory(
     train_data_dir,
     target_size=(img_width, img_height),
     batch_size=batch_size)

 # 검증용 제너레이터 설정
validation_generator = test_datagen.flow_from_directory(
     validation_data_dir,
     target_size=(img_width, img_height),
     batch_size=batch_size)

 # train the model on the new data for a few epochs
model.fit_generator(
     train_generator,
     steps_per_epoch=nb_train_samples // batch_size,
     epochs=epochs,
     validation_data=validation_generator,
     validation_steps=nb_validation_samples // batch_size)
# ###################################################################
# ####################### Problem 2 #################################
# ###################################################################
# # at this point, the top layers are well trained and we can start fine-tuning
# # convolutional layers from inception V3. We will freeze the bottom N layers
# # and train the remaining top layers.

# # let's visualize layer names and layer indices to see how many layers
# # we should freeze:
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

# ###################################################################
# ############################ Your Code ############################
# #################### None 부분을 채우세요 ############################
# """
#     for 문을 이용해서 위의 base model, 즉 GoogLeNet의 상단을 트레이닝하지 않고 고정 시킵니다.
#     "None"을 채우세요.
# """
# # we chose to train the top 2 inception blocks, i.e. we will freeze
# # the first 249 layers and unfreeze the rest:
for None in model.layers["None"]:
    layer.trainable = False
for layer in model.layers["None"]:
    layer.trainable = True

# # we need to recompile the model for these modifications to take effect
# # we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

model.fit_generator(
     train_generator,
     steps_per_epoch=nb_train_samples // batch_size,
     epochs=epochs,
     validation_data=validation_generator,
     validation_steps=nb_validation_samples // batch_size) 

