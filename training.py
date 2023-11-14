import tensorflow as tf
import os
import cv2
import imghdr
from matplotlib import pyplot as plt
import numpy 
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout,Activation
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Precision,Recall,SparseCategoricalAccuracy


#os.listdir('Combat')
gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

#removing any image w weird extensions
data_dir='data'
image_exts=['jpeg','jpg','bmp','png']
dir=[]
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir,image_class)):
        path=os.path.join(data_dir,image_class,image)
        try:
            img=cv2.imread(path)
            tip=imghdr.what(path)
            if tip not in image_exts:
                os.remove(path)
        except:
            print("removing img {}".format(path))



data=tf.keras.utils.image_dataset_from_directory(data_dir) #generate data pipeline 
data_iterator=data.as_numpy_iterator() #access generator from data pipeline
batch=data_iterator.next()
#0=combat 1=building 2=fire 3=humanitarian aid
#loop to figure what label represents what class
print(batch[1])

'''
fig,ax=plt.subplots(ncols=4,figsize=(20,20))
for idx,img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
plt.show()
'''
#print(batch[0]/255)
#scaling data
data=data.map(lambda x,y:(x/255,y))
scaled_iterator=data.as_numpy_iterator().next()

print(len(data))
train_size=int(len(data)*0.7)
val_size=int(len(data)*0.2)
test_size=int(len(data)*0.1)+1 #making sure all the data adds up to 8 ie batchsize

train=data.take(train_size)
val=data.skip(train_size).take(val_size)
test=data.skip(train_size)

Model=Sequential()
#layers
Model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
Model.add(MaxPooling2D())
Model.add(Dropout(0.25))

Model.add(Conv2D(64, (3, 3), activation='relu'))
Model.add(MaxPooling2D())
Model.add(Dropout(0.25)) 

#model.add(Conv2D(16,(3, 3), activation='relu',input_shape=(256,256,3)))
#model.add(MaxPooling2D())
#model.add(Dropout(0.25)) 

Model.add(Flatten())
#dense layers, fully connected layers
Model.add(Dense(128, activation='relu'))
Model.add(Dropout(0.5))
#single output,
Model.add(Dense(5, activation='softmax'))#softmax activation

Model.compile('adam',loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

#print(model.summary())

logsdir='logs'
tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir=logsdir)
hist=Model.fit(train,epochs=10,validation_data=val,callbacks=[tensorboard_callback])

'''
fig=plt.figure()
plt.plot(hist.history['accuracy'],color='teal',label='loss')
plt.plot(hist.history['val_accuracy'],color='orange',label='val_loss')
fig.suptitle('Loss',fontsize=20)
plt.legend(loc='upper left')
plt.show()
'''

precision=Precision()
re=Recall()
acc=SparseCategoricalAccuracy()
#new_model = Sequential()
#new_model.compile('adam',loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

'''
for batch in test.as_numpy_iterator():
    x,y=batch
    yhat=new_model.predict(x)
    #precision.update_state(y,yhat)
    y_onehot = tf.one_hot(y, depth=5)
    y_onehot = tf.reshape(y_onehot, (-1, 5))
    precision.update_state(y_onehot, yhat)
    re.update_state(y,yhat)
    acc.update_state(y,yhat)
'''
accuracy = SparseCategoricalAccuracy()
for batch in test.as_numpy_iterator():
    x, y = batch
    y_onehot = tf.one_hot(y, depth=5)
    y_onehot = tf.reshape(y_onehot, (-1, 5))
    yhat = Model.predict(x)
    #print(yhat)
    accuracy.update_state(y, y_onehot)

# Calculate and print accuracy
accuracy_result = accuracy.result()
print("Accuracy:", accuracy_result)
print("precision:", precision.result().numpy())
print("recall:", re.result().numpy())
print("accuracy:", acc.result().numpy())

target_labels = numpy.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
testimg = cv2.imread('fire1.jpeg')
resize = tf.image.resize(testimg, (256, 256))
resize = numpy.expand_dims(resize, 0)  # Add an extra dimension
resize = resize / 255  # Normalize the image

yhat = Model.predict(resize,0)
predicted_class_index = numpy.argmax(yhat)
class_labels = ['combat', 'destroyedbuildings', 'fire', 'humanitarianaid', 'military']
#print(predicted_class_index)
#print(yhat)
#print(yhat.shape,resize.shape)
if 0 <= predicted_class_index < len(class_labels):
    predicted_class_label = class_labels[predicted_class_index]
    print("Predicted Class:", predicted_class_label)
else:
    print("Predicted class index out of range")

Model.save(os.path.join('models','event_classifier.h5'))
new=load_model(os.path.join('models','event_classifier.h5'))
yhatnew=new.predict(resize,0)
print(yhatnew)
