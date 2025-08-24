import cv2
import wikipedia
from deepface import DeepFace
from ultralytics import YOLO


def start_camera():
    cap=cv2.VideoCapture(0) #replace 0 with ip address to get feed from mobile camera
    model=YOLO("yolov8n.pt")
    count=0
    while(True):

        ret,frame=cap.read()
        count+=1
        if not ret:
            break
        
        try:#face detection by deepface
            face_det=DeepFace.extract_faces(img_path=frame,enforce_detection=False)# face detection frame if true will cuase error when no face found
            frame_h, frame_w, _ = frame.shape

            for face in face_det:
                    area=face["facial_area"]
                    x, y, w, h = area["x"],area["y"],area["w"],area["h"]
                    if w > 30 and h > 30:     
                        if w < frame_w * 0.8 and h < frame_h * 0.8:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        except Exception as e:
            print("face error",e)
    
        object_det=model(frame,imgsz=320)# object detion img size low for fatser detection
        Annotated_frame=object_det[0].plot()  #used for frames plotting
        try: #face recogniton
            recog= frame[y:y+h, x:x+w]   # crop face
            result=DeepFace.find(img_path=recog,db_path="faces/",enforce_detection=False)#recognize/compares img

            if len(result[0])>0:
                identity = result[0].iloc[0]['identity'].split("/")[-1].split(".")[0]#if found remove .jpg and gives name to identity
                cv2.putText(frame, identity, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        except:
            pass

        combined = cv2.addWeighted(frame, 0.7, Annotated_frame, 0.7, 0) #both frames

        # ---------- Show ----------
        cv2.imshow("Phone Stream", combined)

        if cv2.waitKey(1)== ord("d"): #object information
            for box in object_det[0].boxes:
                cls_id=int(box.cls[0])# get class id of detected object
                obj_name = object_det[0].names[cls_id]# get name
                print("detected object",obj_name)
            try:
                summary=wikipedia.summary(obj_name,sentences=2)
                print(f"{obj_name} summary:{summary}")
                cv2.putText(Annotated_frame, f"{obj_name}: {summary[:50]}..." , (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            except Exception as e:
                print(f"Error for {obj_name}: {e}")

                
            
        if cv2.waitKey(1)==ord("q"): #for closing
            break



    cap.release()
    cv2.destroyAllWindows()













