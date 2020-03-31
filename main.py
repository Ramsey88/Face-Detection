import cv2
from facedetector import FaceDetector

image = cv2.imread("goat1.jpg", 1)
cv2.imshow("hi", image)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
p = FaceDetector(faceCascadePath="haarcascade_frontalface_default.xml")
face = p.detect(image=gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
print("I found {} face(s)".format(len(face)))
for (x, y, w, h) in face:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Faces", image)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
