from voice import speak,listen
import vision
import cv2

def main():
    speak("Hello this is multi model assistant!!.Say a command")

    while True:
        command=listen().lower()
        print("You Said",command)

        if "face detection" in command:
            speak("starting recogntion ")
            vision.face_detection()

        elif "object detection":
            speak("starting detection")
            vision.object_detect()
            break

        else:
            speak("not understood!!speak again")

if __name__=="__main__":
    main()