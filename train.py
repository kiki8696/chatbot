import numpy as np
import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD

lemmatizer =WordNetLemmatizer()
#Load file json
with open("intents.json","r") as f:
    intents = json.load(f)

words=[]
classes =[]
documents =[]
stop_words =['.',',','?','!']
for intent in intents['intents']:
   for pattern in intent['patterns']:
     list_word =nltk.word_tokenize(pattern)
     words.extend(list_word)
     documents.append((list_word,intent['tag'])) #documents contains each sentence with corresponding class
     if intent['tag'] not in classes:  
        classes.append(intent['tag']) #classify the classes of words in a variable classes

words =[lemmatizer.lemmatize(word) for word in words if word not in stop_words] #lemmatize the words
words=sorted(set(words)) #remove dublicate words 
classes=sorted(set(classes))
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

training =[]
output_empty =[0] * len(classes) #initialise a vector null with len of classes exist

for document in documents :
   bag =[]

   word_patterns = document[0] #get the sentences from the documents
   word_patterns =[lemmatizer.lemmatize(word.lower()) for word in word_patterns]
   for word in words: 
       if word in word_patterns:
        bag.append(1)  
       else:
          bag.append(0)

   output_row =list(output_empty)
   output_row[classes.index(document[1])] =1 #put 1 for the corespondant class
   training.append([bag,output_row])

random.shuffle(training)
training=np.asarray(training,dtype=object)
x_train = list(training[:,0]) #split data to x_train and y_train
y_train = list(training[:,1])  

model = Sequential()   
model.add(Dense(128,input_shape=(len(x_train[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]),activation='softmax'))
sgd=SGD(learning_rate=0.01,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
model_fit=model.fit(np.array(x_train),np.array(y_train),epochs=200,batch_size=5,verbose=1)
model.save('chatbot_model.h5',model_fit)
print('Done')



