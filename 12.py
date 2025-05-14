# %% Şablon Eşleştirme
import cv2
import numpy as np

img= cv2.imread("sablon.jpg", 0) 
template= cv2.imread("template.jpg", 0)

h, w = template.shape

res= cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc= cv2.minMaxLoc(res)
top_left= max_loc


img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.rectangle(img_color, top_left, (top_left[0] + w, top_left[1] + h), (0, 255, 0), 2)


cv2.putText(img_color, f"Skor: {max_val:.2f}", (top_left[0], top_left[1] + 20 ),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 6. Göster
cv2.imshow("orijinal", img)
cv2.imshow("aranacak sablon",template)
cv2.imshow("Tespit Sonucu", img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()



# %% Özellik Eşleştirme : Feature Matching
import cv2

img1= cv2.imread("chocolates.jpg", 0) #aranacak
img2= cv2.imread ("nestle.jpg", 0) #aranılan 

#orb nesnesini oluştur (Oriented Fast and Rotated BRIEF)
orb= cv2.ORB_create()

# anahtar nokta ve descriptor
key1, ds1= orb.detectAndCompute(img1, None)
key2, ds2= orb.detectAndCompute(img2, None)

#BFMatcher ile eşleştirme
bf= cv2.BFMatcher(cv2.NORM_HAMMING) 
matches= bf.match(ds1, ds2)

matches= sorted(matches, key=lambda x: x.distance) #mesafeye göre sırala

img_matches = cv2.drawMatches(img1, key1, img2, key2, matches[:20], None, flags=2)


#sift
sift = cv2.SIFT_create()
 
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()

#knn 
matches = bf.knnMatch(des1, des2, k=2)

#  iyi eşleşmeleri filtrele
good_matches = []

for m, n in matches:
    if m.distance < 0.6 * n.distance:
        good_matches.append([m])

img_sift_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)

cv2.imshow("img1", img1)
cv2.imshow("img2", img2)
cv2.imshow("eslestirme",img_matches)
cv2.imshow("sift",img_sift_matches)
cv2.waitKey()
cv2.destroyAllWindows()





