# %% video kamera ve video kaydetme
import cv2

#capture
cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # kameranın genişliğini al kaydetmek için lazım manuel olarak da belirlenebilir
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # kameranın yüksekliğini al kaydetmek için lazım manuel belirlenebilir
fourcc = cv2.VideoWriter_fourcc(*"DIVX") # encode etme biçimi
writer= cv2.VideoWriter("video.mp4", fourcc, 20, (width,height)) 


while True: 
    
    ret, frame= cap.read()
    cv2.flip(frame, 1) #ayna görüntüsü eklemek için
    cv2.imshow("video",frame)
    
    # save
    writer.write(frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
writer.release()
cv2.destroyAllWindows()


# %% kaydedilen videoyu okuma ve açma
import cv2 
import time

video = "video.mp4"

cap= cv2.VideoCapture(video)
print(f"Videonun Genişliği: {cap.get(3)}")
print(f"Videonun Yüksekliği: {cap.get(4)}")

if cap.isOpened() == False: 
    print("Hata")
    
while True:
    ret, frame = cap.read()
    
    if ret == True: 
        time.sleep(0.01) #videoyu yavaşlatmak için
        cv2.imshow("video", frame)
    else: break

    if cv2.waitKey(1) & 0xFF == ord("q") : break

cap.release()
cv2.destroyAllWindows()