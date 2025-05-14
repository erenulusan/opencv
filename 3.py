# %%  tuval oluşturma çizim yapma
import cv2
import numpy as np

#512x512 boyutunda siyah bir resim oluştur
canvas= np.zeros((512,512,3), dtype=np.uint8)
#512x512 boyutunda beyaz bir resim oluşturmak için:
canvas_white= np.zeros((512,512,3), dtype=np.uint8) + 255

#çizgi 
cv2.line(canvas, (100,100), (100,300), (255,0,0), 3) 
cv2.line(canvas, (0,512), (512,0), (0,255,0), 5)

#dikdörtgen
cv2.rectangle(canvas, (200,200), (360,360), (0,255,255), -1)
#çember 
cv2.circle(canvas, (300,300), 45, (0,0,255), cv2.FILLED)


#metin
cv2.putText(canvas, "Resim", (100,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))


cv2.imshow("image",canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

