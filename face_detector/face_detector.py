import cv2 as cv

# loading pre-trained data from opencv library
trained_face_data = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load an image to detect faces
# img = cv.imread('rdj.jpg')
img = cv.imread('acm.png')
grayscaled_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
for face in face_coordinates:
    (x, y, w, h) = face
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv.imshow('Image show method', img)
cv.waitKey()
print('Code completed')
