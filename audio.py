# Libraries
import speech_recognition as sr
from deep_translator import GoogleTranslator

# Initialize recognizer
r = sr.Recognizer()

def record_text():
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            print("Listening in Nepali...---.....")
            audio = r.listen(source)
            mytext = r.recognize_google(audio, language="ne-NP")
            print(f"Recognized: {mytext}")
            return mytext
    except sr.RequestError as e:
        print(f"Error: {e}")
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
    return None

def nepali_trans(nepali_text):
    if nepali_text:
        translated_text = GoogleTranslator(source='ne', target='en').translate(nepali_text)
        print(f"Translated text: {translated_text}")
        return translated_text
    return None

def output_text(text):
    if text:  
        with open("output.txt", "a") as f:
            f.write(text + "\n")
        print("Text saved to output.txt")
nepali_texts = [ ]

try:
    while True:
        nepali_text = record_text()  
        if nepali_text:  
            nepali_texts.append(nepali_text)
except KeyboardInterrupt:
    print("User interrupted. Exiting....")

for nepali_text in nepali_texts:
        english_text = nepali_trans(nepali_text) 
        output_text(english_text) 

# Reading content from output.txt file
content = open("output.txt",'r')
print(content.read())




