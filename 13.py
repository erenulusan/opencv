# %% Yüz Tespiti : Face Detection
"""
kullanılan xml dosyaları için:
https://github.com/opencv/opencv/tree/master/data/haarcascades
""" 
import cv2
img1= cv2.imread("sablon.jpg", 0)
img2= cv2.imread("faces.jpg", 0)

face_cascade= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face= face_cascade.detectMultiScale(img1) # bi tane yüz tespiti için
faces= face_cascade.detectMultiScale(img2, minNeighbors=8) # çoklu yüz tespiti için

for (x,y,w,h) in face:
    cv2.rectangle(img1, (x,y), (x+w, y+h), (255,255,255), 10)
    
for (x,y,w,h) in faces:
    cv2.rectangle(img2, (x,y), (x+w, y+h), (255,255,255), 10)
    
cv2.imshow("img1", img1)
cv2.imshow("img2", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# %% Yüz tespiti video 
import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap= cv2.VideoCapture("faces_video.mp4") 
#kamerayla test etmek için : cap= cv2.VideoCapture(0)

while True:
    ret, frame= cap.read()
    
    if not ret:break
    
    frame= cv2.resize(frame, (720,720))
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray= cv2.equalizeHist(gray)
    
    face= face_cascade.detectMultiScale(gray)
    
    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y), (x+w, y+h), (255,255,255), 2)
        
    cv2.imshow("yüz tespit", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):break

cap.release()
cv2.destroyAllWindows()

#%% gülümseme tespiti
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

cap= cv2.VideoCapture(0)

while True:
    ret, frame= cap.read()
    
    if not ret:break
    
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #önce ön yüzü tespit edelim
    face= face_cascade.detectMultiScale(gray)
    
    for (x,y,w,h) in face:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255),2)
    
        # yüz için roi
        roi_gray= gray[y:y+h, x:x+w]
        roi_color= frame[y:y+h, x:x+w]
    
        #gülümsemeyi aramak için
        smiles= smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=10)
        print("Gülümseme sayısı:", len(smiles))
    #eğer gülümseme varsa
        if len(smiles) > 0: 
            cv2.putText(frame, "Smile Detected :)", (x, y+15),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)    
    cv2.imshow("gulumseme tespit", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):break
    
cap.release()
cv2.destroyAllWindows()

# %% göz tespit
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

cap= cv2.VideoCapture(0)

while True:
    ret, frame= cap.read()
    if not ret:break
    
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #yüzü bul
    face= face_cascade.detectMultiScale(gray)
    
    for (x,y,w,h) in face:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255), 2)
        
        #saptanılan yüz bölgesinde göz arayalım
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        
        #göz tespit
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
        
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (255,0,0), 2)
          
    cv2.imshow("yuz ve goz detect", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):break

cap.release()
cv2.destroyAllWindows()

