import speech_recognition as speech
import pyttsx3
import sys




def speak(text):
    engine=pyttsx3.init()
    engine.setProperty('rate', 130)
    engine.say(text)
    engine.runAndWait()


def listen():
    r=speech.Recognizer()
    with speech.Microphone() as mic:
        print("speak now....")
        audio=r.listen(mic)

        try:
            text=r.recognize_google(audio)
            print(text)
            return text

        except speech.RequestError:
            speak("network error")
            return None 

        except speech.UnknownValueError:
            speak("no Audio")
            r=speech.Recognizer()
            return None

        except speech.WaitTimeoutError:
            print("⏳ No speech detected in 10 seconds. Exiting...")
            speak("No input detected. Closing program.")
            sys.exit() 
            
        

if __name__ == "__main__":
    speak("Speak now...")
    command = listen()
    if command:
       if "stop" in command:
        speak("closing the program...")
        sys.exit()
       else:
             speak(command)