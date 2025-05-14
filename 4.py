import cv2
import numpy as np

def nothing(x):
    pass

# Pencere oluştur
cv2.namedWindow("image")

# R, G, B için trackbar ekle
cv2.createTrackbar("R", "image", 0, 255, nothing)
cv2.createTrackbar("G", "image", 0, 255, nothing)
cv2.createTrackbar("B", "image", 0, 255, nothing)

# On/Off switch trackbar
cv2.createTrackbar("0 : OFF \n1 : ON", "image", 0, 1, nothing)

while True:
    img = np.zeros((300, 512, 3), np.uint8)

    # Trackbar'dan değerleri al
    r = cv2.getTrackbarPos("R", "image")
    g = cv2.getTrackbarPos("G", "image")
    b = cv2.getTrackbarPos("B", "image")
    s = cv2.getTrackbarPos("0 : OFF \n1 : ON", "image")

    if s == 0:
        img[:] = 0  # Kapalıysa siyah yap
    else:
        img[:] = [b, g, r]  # RGB değerlerini uygula

    cv2.imshow("image", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()


