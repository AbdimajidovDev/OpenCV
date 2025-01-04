import cv2
import numpy as np
import face_recognition

imgUser = face_recognition.load_image_file('imagesBasic/To\'lqinjon.jpg')
imgUser = cv2.cvtColor(imgUser, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('imagesBasic/Doston.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgUser)[0]
encodeUser = face_recognition.face_encodings(imgUser)[0]
cv2.rectangle(imgUser, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

result = face_recognition.compare_faces([encodeUser], encodeTest)
faceDis = face_recognition.face_distance([encodeUser], encodeTest)
print(result, faceDis)
cv2.putText(imgTest, f'{result} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("Abdimajidov To'lqinjon", imgUser)
cv2.imshow("Eshto'xtarov Dostonbek ", imgTest)
cv2.waitKey(0)
