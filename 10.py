# %% Renk filtresi kullanılarak şekil tespiti
"""
1- kameradan görüntü al
2- bulanıklaştır (gaussian blur): gürültüyü azaltıyoruz düzgün kenar tespiti için
3- griye çevir: net hatları yakalayabilmek için
4- kontur bul: nesnelerin dış hatları
5- konturu basitleştir : kaç köşesinin olduğunu bul ve köşe sayısına göre şekli sınıflandır
6- şekili etiketle ve adını yaz
7- görüntüyü göster
"""

import cv2
import numpy as np

cap= cv2.VideoCapture(0)

#beyaz renk filtresi uygulayalım yani videoda beyaz renkteki nesneleri maskeleyip şekillerini tespit edeceğiz
# beyaz renk için hsv 
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 50, 255])

while True:
    ret, frame= cap.read()
    if not ret:break
    
    #alınan framei hsvye çevir
    hsv= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #beyaz rengi filtrele
    mask= cv2.inRange(hsv, lower_white, upper_white)
    
    #gürültü azalt
    mask= cv2.erode(mask, None, iterations=2)
    mask= cv2.dilate(mask, None, iterations=2)
    
    #kontur 
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        area= cv2.contourArea(c)
        if area < 1500:continue
        
        epsilon= 0.02* cv2.arcLength(c, True)
        approx= cv2.approxPolyDP(c, epsilon, True)
        corner_counts= len(approx)
        
        shape="Tanimsiz"
        if corner_counts==3:
            shape= "ucgen"
            
        elif corner_counts== 4:
            (x,y,w,h)= cv2.boundingRect(approx)
            ar= float(w) / h
            shape= "kare" if 0.95 <= ar <= 1.05 else "dikdortgen"
            
            
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
        x, y = approx[0][0]
        cv2.putText(frame, shape + " (Beyaz)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Beyaz Sekil Tespiti", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()    