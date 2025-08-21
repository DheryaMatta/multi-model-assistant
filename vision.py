import cv2
from deepface import DeepFace
from ultralytics import YOLO

cap=cv2.VideoCapture(0) #replace 0 with ip address to get feed from mobile camera
model=YOLO("yolov8n.pt")
count=0
while(True):

    ret,frame=cap.read()
    count+=1
    if not ret:
        break
    
    try:
        face_det=DeepFace.extract_faces(img_path=frame,enforce_detection=False)

        for face in face_det:
             area=face["facial_area"]
             x, y, w, h = area["x"],area["y"],area["w"],area["h"]
             cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 5)

    except Exception as e:
        print("face error",e)
    results=model(frame,imgsz=320)
    Annotated_frame=results[0].plot()  #used for detecting
    combined = cv2.addWeighted(frame, 0.7, Annotated_frame, 0.7, 0)

    # ---------- Show ----------
    cv2.imshow("Phone Stream", combined)


        
    if cv2.waitKey(1)==ord("q"):
        break



cap.release()
cv2.destroyAllWindows()













