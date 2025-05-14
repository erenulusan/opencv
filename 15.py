# %% Meanshift - Camshift ile Yüz Tanıma ve Takip etme
"""
haarcascade kullanılarak yüz tespiti kamera üzerinden yüz tespiti
sonra yapılan yüz tespitinde meanshift ve camshift tracking algoritması kullanılacak "c" basınca camshift "m" basınca meanshit şeklinde
"""
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# default mode
aktif_mode= "c"

while True:
    ret, frame= cap.read()
    if not ret:
        print("kamera görüntüsü yok")
        break
    
    faces= face_cascade.detectMultiScale(frame)
    if len(faces) > 0:
        (x,y,w,h)= faces[0]
        track_window= (x,y,w,h)
        
        roi= frame[y:y+h, x:x+w]
        hsv_roi= cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        break
    
#termination ve criteria
term_crit= (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    
#takip
while True:
    ret, frame= cap.read()
    if not ret:break
    
    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bp= cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)
    
    #meanshift algoritmasını uygula
    if aktif_mode == "m":
        _, track_window= cv2.meanShift(bp, track_window, term_crit)
        x,y,w,h= track_window
        ms_frame= cv2.rectangle(frame.copy(), (x,y), (x+w, y+h), (255,0,255), 2)
        cv2.imshow("Meanshift",ms_frame)
        
    #camshift uygula
    elif aktif_mode == "c":
        ret, track_window= cv2.CamShift(bp, track_window, term_crit)
        pts=cv2.boxPoints(ret)
        pts=np.int0(pts)
        cs_frame= cv2.polylines(frame.copy(), [pts], True, (0,255,0),2)
        cv2.imshow("camshift", cs_frame)
        
        
    key= cv2.waitKey(1) & 0xFF
    if key== ord("m"):
        aktif_mode="m"
        cv2.destroyAllWindows()
    elif key == ord("c"):
        aktif_mode="c"
        cv2.destroyAllWindows() 
        
    elif key == ord("q"):break
cap.release()
cv2.destroyAllWindows()