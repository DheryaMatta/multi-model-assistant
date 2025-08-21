import cv2

capture=cv2.VideoCapture("http://10.139.44.102:8080/video")

while(True):

    ret,frame= capture.read()

    if not ret:
        break

    cv2.imshow("phone webcam",frame)

    if cv2.waitKey(1)==ord("q"):
        break

capture.release()
cv2.destroyAllWindows()