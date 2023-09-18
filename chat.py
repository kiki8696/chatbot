import numpy as np
import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model


lemmatizer=WordNetLemmatizer()
with open("intents.json","r") as f:
    intents = json.load(f)

words=pickle.load(open('words.pkl','rb'))
classes=pickle.load(open('classes.pkl','rb'))
model =load_model('chatbot_model.h5')

def clean_sentence(sentence):
    sentence_words=nltk.word_tokenize(sentence)
    sentence_words=[lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
def bag_of_words(sentence):
    sentence_words=clean_sentence(sentence)
    bag= [0] * len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word==w:
                bag[i]=1
    return np.array(bag)
def prediction(sentence):
    bow =bag_of_words(sentence)
    response=model.predict(np.array([bow]))[0]
    ERROR=0.25
    result=[[i,r] for i,r in enumerate(response) if r > ERROR] 
    result.sort(key=lambda x:x[1],reverse=True)
    liste_result=[]
    for r in result:
        liste_result.append({'intents':classes[r[0]],'probability':str(r[1])})
    return liste_result

def get_response(intents_list,intents_json):
    tag=intents_list[0]['intents']
    list_of_intents =intents_json['intents']
    for i in list_of_intents :
        if i['tag']==tag:
            result = random.choice(i['responses'])
            break
    return result
print("GO! Bot is running")
print("If you want to leave the chat please write (quit ot exit)")
while True:
    message=input("The User:")
    if message != "quit" and message !="exit":
       ints=prediction (message)
       res=get_response(ints,intents)
       print(res)
    else:
        print("goodbye!  glade to talk with you")
        break







