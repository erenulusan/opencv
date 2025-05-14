# %% kenar algılama :edge detection
import cv2
import numpy as np

img= cv2.imread("farklibmw.jpg", 0)

#her şeyi kenar olarak tespit eder 
edges= cv2.Canny(image= img, threshold1=0, threshold2=255)

#medyan temelli threshold belirleme
median_val= np.median(img)

low= int(max(0, (1 - 0.33)* median_val)) #medyan alt sınırı -> 81
high= int(min(255, (1 + 0.33)*median_val)) #medyan üst sınırı -> 160

median_edges= cv2.Canny(image= img, threshold1=low, threshold2=high)

#gürültüler kenar olarak algılanabilir, gürültü azaltıp kenar bulma
blurred= cv2.blur(img, ksize=(5,5))
blurred_median_val= np.median(blurred)

blurred_low= int(max(0, (1 - 0.33)* blurred_median_val))
blurred_high= int(min(255, (1 + 0.33)* blurred_median_val))

blurred_edges= cv2.Canny(img, threshold1=blurred_low, threshold2= blurred_high)

cv2.imshow("orijinal", img)
cv2.imshow("kenar algilama", edges)
cv2.imshow("medyan kenar", median_edges)
cv2.imshow("bulanıklastır + medyan", blurred_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %% köşe algılama : corner detection
import cv2
import numpy as np

# Harris Corner Detection
img = cv2.imread("santranc.jpg", 0)
color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Görüntüyü renkliye çevir
hc = cv2.cornerHarris(img, blockSize=2, ksize=3, k=0.04)
hc = cv2.dilate(hc, None)
color_img[hc > 0.01 * hc.max()] = [0, 0, 255]  # Kırmızıyla boya

cv2.imshow("Harris Corner Detection", color_img)

# Shi-Tomasi Detection
img2 = cv2.imread("santranc.jpg")
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
corners = np.int0(corners)

for corner in corners:
    x, y = corner.ravel()
    cv2.circle(img2, (x, y), 4, (0, 255, 0), -1)

cv2.imshow("Shi-Tomasi Corner Detection", img2)

cv2.waitKey(0)
cv2.destroyAllWindows()

# %% Gerçek Zamanlı Kameradan  Kenar Tespiti: 3 Aşamlı canny uygulayacağız yukarıdaki gibi
import cv2
import numpy as np
import time

cap= cv2.VideoCapture(0)

while True:
    ret, frame= cap.read()
    if not ret:
        break
    
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #sabit threshold
    edges= cv2.Canny(image= gray, threshold1=50, threshold2=150)
    
    # medyan temelli threshold 
    median_val = np.median(gray)
    low = int(max(0, (1 - 0.33) * median_val))
    high = int(min(255, (1 + 0.33) * median_val))
    canny_median = cv2.Canny(gray, low, high)    
    
    # gürültülerin kenar olarak algılanmaması için blurlayıp kenar medyan threshold ile kenar bulalım
    blurred= cv2.blur(gray, (5,5))
    blurred_median_val= np.median(blurred)
    blur_low = int(max(0, (1 - 0.33) * blurred_median_val))
    blur_high = int(min(255, (1 + 0.33) * blurred_median_val))
    canny_blur= cv2.Canny(blurred, blur_low, blur_high)
    
    cv2.imshow("1 - Sabit Threshold", edges)
    cv2.imshow("2 - Medyan Threshold", canny_median)
    cv2.imshow("3 - Blur + Medyan", canny_blur)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()    


# %% kontur algılama : contour detection
import cv2
import matplotlib.pyplot as plt
import numpy as np

#kontur için siyah bir resim oluştur
canvas= np.zeros((600,600), dtype= np.uint8)

# şekil çiz
cv2.rectangle(canvas, (150,100), (450,200), 255, -1) # dikdörtgen 
cv2.circle(canvas, (300,150), 40, 0, -1) # daire çiz

#üçgen çiz
pts= np.array([[300,300], [400,500], [200,500]], np.int32)
pts= pts.reshape((-1,1,2))
cv2.fillPoly(canvas, [pts], 255)

cv2.circle(canvas, (100,400), 40, 255, -1)
cv2.circle(canvas, (100,400), 20, 0, -1)

#bu oluşturduğumuz görseli kontur olarak kaydedilim
cv2.imwrite("kontur.jpg", canvas)


# şimdi bu resmi içeri aktarıp kontur arama yapalım
img= cv2.imread("kontur.jpg", 0)

#kontur bulma
contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE )

external = np.zeros_like(img)
internal= np.zeros_like(img)

#konturları ayır
for i in range(len(contours)):
    if hierarchy [0][i][3] == -1: #dış kontur
        cv2.drawContours(external, contours, i , 255, -1)
    else:
        cv2.drawContours(internal ,contours, i, 255, -1)
        
#görselleştir
cv2.imshow("orijinal",img)
cv2.imshow("dis konturlar", external)
cv2.imshow("ic konturlar", internal)
cv2.waitKey(0)
cv2.destroyAllWindows()