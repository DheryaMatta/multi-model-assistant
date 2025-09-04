import cv2
import wikipedia
from deepface import DeepFace
from ultralytics import YOLO



def face_detection(frame):
    cap=cv2.VideoCapture(0) #replace 0 with ip address to get feed from mobile camera
    model=YOLO("yolov8n.pt")
    while(True):

        ret,frame=cap.read()

        if not ret:
            break
        try:
            face_det = DeepFace.extract_faces(img_path=frame, enforce_detection=False)
            for face in face_det:
                area = face["facial_area"]
                x, y, w, h = area["x"], area["y"], area["w"], area["h"]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

                # Face recognition
                recog = frame[y:y+h, x:x+w]
                result = DeepFace.find(img_path=recog, db_path="faces/", enforce_detection=False)
                if len(result[0]) > 0:
                    identity = result[0].iloc[0]['identity'].split("/")[-1].split(".")[0]
                    cv2.putText(frame, identity, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                else:
                    cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        except Exception as e:
            print("face error:", e)

        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    def object_detect():
        cap=cv2.VideoCapture(0) #replace 0 with ip address to get feed from mobile camera
        model=YOLO("yolov8n.pt")
        while (True):
            ret,frame=cap.read()

            results = model(frame, imgsz=320)
            annotated_frame = results[0].plot()

            cv2.imshow("Object Detection", annotated_frame)

            if not ret:
                break
            for box in results[0].boxes:
                cls_id=int(box.cls[0])# get class id of detected object
                obj_name = results[0].names[cls_id]# get name
                print("detected object",obj_name)
                return obj_name
            try:
                summary=wikipedia.summary(obj_name,sentences=2)
                print(f"{obj_name} summary:{summary}")
                cv2.putText(annotated_frame, f"{obj_name}: {summary[:50]}..." , (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            except Exception as e:
                    print(f"Error for {obj_name}: {e}")

                
            
            if cv2.waitKey(1)==ord("q"): #for closing
                break



    cap.release()
    cv2.destroyAllWindows()













