# %% morfolojik işlemler
import cv2
import numpy as np

img= np.zeros((512,512,3),dtype=np.uint8)

# önce siyah resime yazı ekleyelim
text= "open cv egzersiz"
font= cv2.FONT_HERSHEY_SIMPLEX
scale= 1.75
thickness = 4
color = (255, 255, 255)

# yazının genişlik ve yüksekliği
(text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)

#şimdi yazıyı ortalamak için koordinatları hesapla
x= (img.shape[1] - text_w) // 2
y= (img.shape[0] - text_h) // 2


# yazıyı ekleyelim
cv2.putText(img, text, (x,y), font, scale, color,  thickness, cv2.LINE_AA)
cv2.imshow("Yazı", img)
cv2.imwrite("opencv_egzersiz.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()



# %% morfolojik işlemlere başlama ->
import cv2
import numpy as np


img = cv2.imread("opencv_egzersiz.jpg", 0)  # grayscale

kernel = np.ones((5, 5), dtype=np.uint8)

# 1. Erozyon (aşındırma)
erode_img = cv2.erode(img, kernel, iterations=1)

# 2. Genişleme (dilate)
dilate_img = cv2.dilate(img, kernel, iterations=1)

# 3. Beyaz gürültü (white noise) + Açılma (opening)
whiteNoise = (np.random.randint(0, 2, img.shape) * 255).astype(np.uint8)
noise_img_white = cv2.add(img, whiteNoise)
opening = cv2.morphologyEx(noise_img_white, cv2.MORPH_OPEN, kernel)

# 4. Siyah gürültü (black noise) + Kapatma (closing)
blackNoise = (np.random.randint(0, 2, img.shape) * 255).astype(np.uint8)
noise_img_black = cv2.subtract(img, blackNoise)
closing = cv2.morphologyEx(noise_img_black, cv2.MORPH_CLOSE, kernel)

# 5. Gradyan (dilate - erode)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

#  Görüntüleri göster
cv2.imshow("Orijinal", img)
cv2.imshow("Erozyon", erode_img)
cv2.imshow("Genişleme", dilate_img)
cv2.imshow("Beyaz Gürültü", noise_img_white)
cv2.imshow("Açılma", opening)
cv2.imshow("Siyah Gürültü", noise_img_black)
cv2.imshow("Kapatma", closing)
cv2.imshow("Gradyan", gradient)

cv2.waitKey(0)
cv2.destroyAllWindows()


