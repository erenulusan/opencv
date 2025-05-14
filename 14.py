# %% plaka yakalama ve plaka yazdırma
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"D:\Tesseract\tesseract.exe"

img= cv2.imread("plaka.jpg")

gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#gürültüleri azaltalım
blurred= cv2.bilateralFilter(gray, 11, 17, 17)
edged= cv2.Canny(blurred, 30, 200)

#kontur
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20] #büyük konturları filtrele

plate_img = None

for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        plate_img = gray[y:y+h, x:x+w]
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        break


if plate_img is not None:
    text = pytesseract.image_to_string(plate_img, config="--psm 7")
    print("Plaka:", text.strip())
    cv2.imshow("Plaka", plate_img)
else:
    print("Plaka yok")



cv2.imshow("Sonuç", img)
cv2.waitKey(0)
cv2.destroyAllWindows()