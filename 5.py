# %% Görüntü birleştirme 
import cv2
import numpy as np

#resimleri içe aktar
img1= cv2.imread("bmw.jpg")
img2= cv2.imread("farklibmw.jpg")

#birleştirmek için görüntülerin aynı boyutta olması gerekir görüntüleri resize etme
img1=cv2.resize(img1, (512,512))
img2= cv2.resize(img2, (512,512))

#yatay birleştirme
yatay= np.hstack((img1,img2))

#dikey
dikey= np.vstack((img1,img2))


cv2.imshow("img1",img1)
cv2.imshow("img2",img2)
cv2.imshow("yatay birlestirme", yatay)
cv2.imshow("dikey birlestirme", dikey)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %% blending görüntü karıştırma 
import cv2
import numpy as np

#resimleri içe aktar
img1= cv2.imread("bmw.jpg")
img2= cv2.imread("farklibmw.jpg")

#birleştirmek için görüntülerin aynı boyutta olması gerekir görüntüleri resize etme
img1=cv2.resize(img1, (512,512))
img2= cv2.resize(img2, (512,512))


#blending
blend = cv2.addWeighted(src1=img1, alpha=0.4, src2=img2, beta=0.6, gamma=0)#gamma parlaklık eklemek için


cv2.imshow("img1",img1)
cv2.imshow("img2",img2)
cv2.imshow("birlestirme",blend)
cv2.waitKey(0)
cv2.destroyAllWindows()




# %%  Rotate işlemleri
import cv2
img= cv2.imread("bmw.jpg")
img= cv2.resize(img,(512,512))

img_ayna= cv2.flip(img, 1) 

rotate_90= cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

cv2.imshow("ayna goruntu", img_ayna)
cv2.imshow("90 derece saat", rotate_90)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %% perspektif düzeltme : warp perspective
import cv2

img=cv2.imread("warp.jpg")

pts1= np.float32([[163,90], [623,91], [163,435], [630, 434]])
pts2= np.float32([[4,4], [750,10], [4,530], [750,530]])

m=cv2.getPerspectiveTransform(pts1, pts2)
imgOutput= cv2.warpPerspective(img, m, (750,530))

cv2.imshow("original", img)
cv2.imshow("son",imgOutput)
cv2.waitKey()
cv2.destroyAllWindows()


# %% Görüntü eşikleme thresholding
import cv2

img= cv2.imread("bmw.jpg")
img_resize= cv2.resize(img, (600,600))
img_resize= cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

# eşikleme
_, thresh_img = cv2.threshold(img_resize, thresh = 60, maxval = 255, type = cv2.THRESH_BINARY)

#adaptif eşikleme
thresh_img2 = cv2.adaptiveThreshold(img_resize, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)

cv2.imshow("orijinal", img_resize)
cv2.imshow("Esikleme", thresh_img)
cv2.imshow("Adaptif esikleme", thresh_img2)
cv2.waitKey()
cv2.destroyAllWindows()



