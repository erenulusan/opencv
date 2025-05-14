# %% blurring işlemleri
"""
1- ortalama blur: Seçilen pencere içindeki tüm piksellerin ortalamasını alır ve merkez pikselin yerine yazar. 
    * gürültüyü hafifletir ama keskin kenarları bozabilir
    
2- gauss blur: pencere içindeki piksellerin gauss dağılımına göre ağırlıklı ortalamasını alır merkeze yakın olanlar daha etkili olur.
    * gürültüyü daha doğal şekilde azaltır
    * görüntü daha az bulanık olur
    * genelde edge detection öncesi kullanılır
    * sigmaX parametresi: bulanıklık seviyesini kontrol eder

3- medyan blur: pencere içindeki piksellerin medyanını alır ve merkez piksele yazar
    * salt pepper için en iyi filtre
    * kenarları korur, daha az detay kaybı yaşanır.
"""
import cv2
import numpy as np

#resmi içeri aktaralım
img = cv2.imread("farklibmw.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ortalam bulanıklaştırma
mean_blurring= cv2.blur(img, ksize= (3,3))

# gaussian blur
gb= cv2.GaussianBlur(img, ksize = (3,3), sigmaX = 7)

# medyan blur
mb= cv2.medianBlur(img, ksize=3)


cv2.imshow("image",img)
cv2.imshow("ortalama blur", mean_blurring)
cv2.imshow("gaussian blur", gb)
cv2.imshow("medyan blur", mb)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %% gürültü ekleme ve blurring
import cv2
import numpy as np

img= cv2.imread("farklibmw.jpg")
img_rgb= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_norm= img_rgb / 255.0 

# gaussian noise
def gaussian_noise(image, mean=0 , var= 0.01):
    row, col, ch= image.shape
    sigma= var**0.5
    gauss= np.random.normal(mean, sigma, (row,col,ch))
    noisy= image + gauss
    noisy= np.clip(noisy, 0, 1)
    return noisy

noisy_gauss= gaussian_noise(img_norm)
denoised_gauss = cv2.GaussianBlur((noisy_gauss * 255).astype(np.uint8), (3, 3), 7)

# salt pepper gürültü ekleme
def salt_pepper_noise(image, amount=0.005, s_vs_p=0.5):
    row, col, ch = image.shape
    noisy = np.copy(image)

    # Salt (beyaz)
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[tuple(coords)] = 1

    # Pepper (siyah)
    num_pepper = np.ceil(amount * image.size * (1 - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[tuple(coords)] = 0

    return noisy

noisy_sp = salt_pepper_noise(img_norm)
denoised_sp = cv2.medianBlur((noisy_sp * 255).astype(np.uint8), 3)

# poisson gürültü
def poisson_noise(image):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    noisy = np.clip(noisy, 0, 1)
    return noisy

noisy_poisson = poisson_noise(img_norm)
denoised_poisson = cv2.GaussianBlur((noisy_poisson * 255).astype(np.uint8), (3, 3), 7)


cv2.imshow("Orijinal", img)

cv2.imshow("Gaussian Noisy", (noisy_gauss * 255).astype(np.uint8))
cv2.imshow("Gaussian Denoised", denoised_gauss)

cv2.imshow("Salt-Pepper Noisy", (noisy_sp * 255).astype(np.uint8))
cv2.imshow("Salt-Pepper Denoised", denoised_sp)

cv2.imshow("Poisson Noisy", (noisy_poisson * 255).astype(np.uint8))
cv2.imshow("Poisson Denoised", denoised_poisson)

cv2.waitKey(0)
cv2.destroyAllWindows()



# %% gradyan
import cv2

img = cv2.imread("santranc.jpg", 0)

# dikey kenarları yakalamak için
dikey = cv2.Sobel(img, ddepth=cv2.CV_16S, dx=1, dy=0, ksize=5)
dikey = cv2.convertScaleAbs(dikey)

# yatay kenarları yakalamak için
yatay = cv2.Sobel(img, ddepth=cv2.CV_16S, dx=0, dy=1, ksize=5)
yatay = cv2.convertScaleAbs(yatay)

# hem dikey hem yatay kenarları yakalamak için 
laplacian = cv2.Laplacian(img, ddepth=cv2.CV_16S)
laplacian = cv2.convertScaleAbs(laplacian)

# göster
cv2.imshow("Orijinal", img)
cv2.imshow("Dikey", dikey)
cv2.imshow("Yatay", yatay)
cv2.imshow("Laplacian", laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()


# %% histogram hesaplamaları: piksel yoğunluklarının dağılımı
import cv2
import matplotlib.pyplot as plt


img = cv2.imread("sablon.jpg", 0)
#histogram hesapla
hist_gray = cv2.calcHist([img], [0], None, [256], [0,256])

#histogram eşitleme ve eşitlenen histogramı hesaplama
equalized= cv2.equalizeHist(img)
hist_eq = cv2.calcHist([equalized], [0], None, [256], [0,256])


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Orijinal")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(equalized, cmap='gray')
plt.title("histogram eşitleme")
plt.axis("off")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(hist_gray, color='gray')
plt.title(" orijinal görüntünün histogramı")
plt.xlabel("px"), plt.ylabel("yoğunluk")

plt.subplot(1, 2, 2)
plt.plot(hist_eq, color='black')
plt.title("eşitlenmiş histogram")
plt.xlabel("px"), plt.ylabel("yoğunluk")

plt.tight_layout()
plt.show()


