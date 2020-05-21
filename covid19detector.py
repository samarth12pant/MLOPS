#!/usr/bin/env python
# coding: utf-8

# In[1]:

''''@author: Samarth Pant'''
from keras.layers import Convolution2D


# In[2]:


from keras.layers import MaxPool2D
from keras.layers import Flatten  
from keras.layers import Dense
from keras.optimizers import Adam ,RMSprop ,SGD ,Nadam ,Adamax
from keras.models import Sequential


# In[3]:


model = Sequential()


# In[4]:


import random


# In[5]:

#adds a convolution layer and picks random kernl size each time code runs.
model.add(Convolution2D(filters=random.randint(40,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6),(7,7))),
                        activation='relu',
                        input_shape=(130,130,3)
                       ))


# In[6]:

#adds a MaxPool layer and picks random kernl size each time code runs.
model.add(MaxPool2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6),(7,7)))))


# In[8]:


#this is a function for randomly selecting an architecture for the model from available architectures.
def architecture(option):
    if option == 1:
        model.add(Convolution2D(filters=random.randint(40,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6),(7,7))),
                        activation='relu'
                       ))
    elif option == 2:
        model.add(Convolution2D(filters=random.randint(40,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6),(7,7))),
                        activation='relu'
                       ))
        model.add(MaxPool2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6),(7,7)))))
        
    elif option == 3:
        #two convolutional and 2 max pooling layers
        model.add(Convolution2D(filters=random.randint(40,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6),(7,7))),
                        activation='relu'
                       ))
        model.add(MaxPool2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6),(7,7)))))
        
        model.add(Convolution2D(filters=random.randint(40,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6),(7,7))),
                        activation='relu'
                       ))
        model.add(MaxPool2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6),(7,7)))))
    elif option == 4:
        model.add(Convolution2D(filters=random.randint(40,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6),(7,7))),
                        activation='relu'
                       ))
        model.add(Convolution2D(filters=random.randint(40,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6),(7,7))),
                        activation='relu'
                       ))
        model.add(MaxPool2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6),(7,7)))),padding='same')
    
    else:
        model.add(Convolution2D(filters=random.randint(40,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6),(7,7))),
                        activation='relu'
                       ))
        model.add(MaxPool2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6),(7,7)))))
        
        model.add(Convolution2D(filters=random.randint(40,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6),(7,7))),
                        activation='relu'
                       ))
        model.add(MaxPool2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6),(7,7)))))


# In[9]:


#above function is called to add a architecture of layer to the model
architecture(random.randint(1,4))


# In[10]:

#adds a convolution and MaxPool2D layer and picks random kernl size each time code runs.
model.add(Convolution2D(filters=random.randint(40,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6),(7,7))),
                        activation='relu'
                       ))
model.add(MaxPool2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6),(7,7)))))


# In[11]:

#adds flattening layer
model.add(Flatten())


# In[12]:


#function for addition of fully connected network at the end of the model
def fclayer(option):
    if option == 1:
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
    elif option == 2:
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
    elif option == 3:
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
    elif option == 4:
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        
    else:
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))


# In[14]:


#each time code runs, it randomly selects the layers from available above 5 options of layers
fclayer(random.randint(1,5))


# In[15]:

#adds dense layer and choosses random activation function for the model.
model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','','softmax','sigmoid'))))


# In[16]:


#output layer(only 1 neuron as it is binary classification.Either a person will have COVID-19 or will not have)
model.add(Dense(units=1,activation='sigmoid'))


# In[17]:

#prints summary of the model
print(model.summary())


# In[18]:


#random choice for initializer
model.compile(optimizer=random.choice((RMSprop(lr=.0001),Adam(lr=.0001),SGD(lr=.001),Nadam(lr=.001),Adamax(lr=.001))),loss='binary_crossentropy',metrics=['accuracy'])


# In[19]:


from keras.preprocessing.image import ImageDataGenerator


# In[22]:

#importing training and testing data nd convertuing acording to parameters specified while making the model. 
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'E:/covid_dataset/training_set/',
        target_size=(130,130),
        batch_size=40,
        class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
        'E:/covid_dataset/testing_set/',
        target_size=(130,130),
        batch_size=40,
        class_mode='binary')


# In[24]:

#performing epochs on model
result = model.fit(
        train_generator,
        steps_per_epoch=50,
        epochs=random.randint(1,50),
        validation_data=validation_generator,
        validation_steps=28)


# In[25]:

#saving the model.
#model.save("covid.h5")


# In[26]:

#shows loss and accuracy on both training and testing data
result.history


# In[27]:


print(result.history['accuracy'][0])


# In[29]:


modlayrs =str(model.layers)
accuracy = str(result.history['accuracy'][0])


# In[32]:


#sending mail if accuracy greater than or is equal to 78 percent
if result.history['accuracy'][0] >= .78:
    import smtplib
    # creates SMTP session 
    sm = smtplib.SMTP('smtp.gmail.com', 587)
    # start TLS for security 
    sm.starttls()


# In[34]:

#for user to login
sm.login("sender_emailaddress", "senderpassword")


# In[36]:


message1 = accuracy
message2 = modlayrs


# In[37]:


# sending email to developer 
sm.sendmail("sender_emailaddress", "receivers_emailaddress", message1)
sm.sendmail("sender_emailaddress", "receivers_emailaddress", message2)
print("Mail sent successfully..")


# In[39]:


#closing the session
sm.quit()


# In[ ]:




