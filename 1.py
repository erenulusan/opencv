# %% Resmi İçe Aktarma
import cv2

#resmi içe aktar
img= cv2.imread("bmw.jpg")
cv2.imshow("image", img)

# gri olarak oku ve kaydet
img_gray= cv2.imread("bmw.jpg", 0) # cv2.imread("bmw.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imwrite("bmw_copy.jpg", img_gray)
cv2.imshow("gri", img_gray)

cv2.waitKey(0)
cv2.destroyAllWindows()

# %% 
import cv2 

img= cv2.imread("bmw.jpg")
print(f"Resim Boyutu: {img.shape}")

# yeniden boyutlandır
img_resized= cv2.resize(img, (600, 480))
print("Resized Resim boyutu: {img_resized.shape}")

#kırpma
"""
sol üst - (x,y) - (99,215)
sağ üst - (x,y) - (1778,218)
sol alt - (x,y) - (100,1010)
sağ alt - (x,y) - (1774, 1006)

x1- 99 , x2- 1778, y1-215, y2-1010

"""

img_cropped= img[215:1010, 99:1778]# width height -> height width

cv2.imshow("orijinal resim", img)
cv2.imshow("yeniden boyutlanmis resim", img_resized)
cv2.imshow("Kirpma uygulanan resim", img_cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
