from voice import speak,listen
import vision
import cv2

def main():
    speak("Hello this is multi model assistant!!.Say a command")

    while True:
        command=listen().lower()
        print("You Said",command)

        if "start vision" in command:
            speak("starting vision")
            vision.start_camera()
            if cv2.waitKey(1)== ord("s"):
                object=vision.object_detect
                speak(object)

        elif cv2.waitKey(1)== ord("q"):
            speak("exiting vision")
            break

        else:
            speak("not understood!!speak again")

if __name__=="__main__":
    main()