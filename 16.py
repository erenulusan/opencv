# %% Tracking Algoritmaları : Seçilen roide tekli nesne takibi
import cv2

# farkli algoritmaları gözlemleyebilmek için bir dict oluşturalım

tracking_algorithms= {"csrt"      : cv2.legacy.TrackerCSRT_create,
                      "kcf"       : cv2.legacy.TrackerKCF_create,
                      "mil"       : cv2.legacy.TrackerMIL_create,
                      "boosting"  : cv2.legacy.TrackerBoosting_create,
                      "mosse"     : cv2.legacy.TrackerMOSSE_create,
                      "medianflow": cv2.legacy.TrackerMedianFlow_create}


#algoritma seç
selected_algorithm= "mosse"

cap=cv2.VideoCapture("akantrafik.mp4")

ret, frame= cap.read()
if not ret:
    print("video ya da kamera yok")

#roi seçimi : takip için
bbox = cv2.selectROI("isaretle", frame)

# kullanacağımız trackerı oluşturalım
tracker= tracking_algorithms[selected_algorithm]()
tracker.init(frame,bbox)


#döngüyü kuralım
while True:
    
    ret, frame= cap.read()
    if not ret: break
    
    
    success, box= tracker.update(frame)
    
    if success:
        x,y,w,h= [int(i) for i in box]
        cv2.rectangle(frame,(x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"takip algoritmasi: {selected_algorithm}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
    else: 
        cv2.putText(frame, "takip yok", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
    cv2.imshow("takip", frame)
    if cv2.waitKey(30) & 0xFF == ord("q"):break

cap.release()
cv2.destroyAllWindows()    
    


    
    