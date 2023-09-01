'''This program is a rule based chatbot. Input is an audio which is converted to text.
The text is then one hot encoded in order to predict a response using the bag of words model. The 
training data is a file which contains 'tags'(classes) to which each user input might fall into,
'patterns'- probable questions asked and 'responses'- suitable responses for the patterns. A DNN
network is used to learn and predict the probability of the user input belonging to a particular
class.
'''

import os
import time
import speech_recognition as sr
import pickle
import random
import tflearn
import tensorflow
import pyttsx3
import nltk
nltk.download('punkt')
import json
import numpy
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
from gensim.parsing.preprocessing import remove_stopwords
import pyjokes

#text to speech
def speak(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.say(text)
    engine.runAndWait()

#speech to text
def get_audio():
    r=sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        said = ""

        try:
            said=r.recognize_google(audio)
            print(said)
        except Exception as e:
            print("Exception: " + str(e))
    return said

#get the intents file
with open("intents.json") as file:
    data = json.load(file)


words = []
labels = []
inp_x = []
inp_y = []

#processing training data
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        inp_x.append(wrds)
        inp_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

# Create a dictionary using the tokenized words
for x, doc in enumerate(inp_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(inp_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)

#DNN network
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 16)
net = tflearn.fully_connected(net, 16)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

# Initialize the model
model = tflearn.DNN(net)

# Change this to false after one run to avoid training again and again
not_saved = True

if not_saved:
    model.fit(training, output, n_epoch=500, batch_size=10, show_metric=True)
    model.save("model.tflearn")

model.load("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for k in s_words:
        for i, j in enumerate(words):
            if j == k:
                bag[i] = 1

    return numpy.array(bag)

class chatbot_ai():
    def __init__(self,name):
        self.name=name
    def driver_function(self):
        print("Start talking, say quit once to exit")
        while True:
            inp = get_audio()
            print("you:",inp)
            if inp.lower() == "quit":
                break

            results = model.predict([bag_of_words(inp, words)])
            maximum= numpy.amax(results)
            result_2=results
            result_2= numpy.delete(result_2, numpy.where(result_2 == maximum))
            sec_max=numpy.amax(result_2)
            print("Probability of the response being right :",maximum)
            if 'joke' in inp:
                a=pyjokes.get_joke()
                print(self.name,':',a)
                speak(a)
            elif maximum>=1.1*sec_max:
                results_index = numpy.argmax(results)
                tag = labels[results_index]

                for tg in data["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['responses']

                x=random.choice(responses)
                print(self.name,':',x)
                speak(x)
            else:
                y= "I don't get you, can you rephrase your question"
                print(self.name,':',y)
                speak(y)

p=chatbot_ai("Tim")
print("Listening...")
speak("Powering On....start speaking, if you want to stop say quit")
p.driver_function()


