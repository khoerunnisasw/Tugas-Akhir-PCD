import cv2 #library opencv
img=cv2.imread('foto1.jpg') #membaca file gambar
cv2.imshow('image',img) #menampilkan gambar
cv2.waitKey()
cv2.destroyAllWindows()
