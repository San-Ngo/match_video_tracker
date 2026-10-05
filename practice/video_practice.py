import cv2

cap = cv2.VideoCapture('/Users/baonamsanngo/Pictures/Photos Library.photoslibrary/resources/derivatives/C/CE7A60AD-1DD7-4A58-A1E3-942FC3DE1809_1_102_o.jpeg')
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img = cap.read()
    if not success:          
        break
    cv2.imshow("Video", img)
    if cv2.waitKey(5000) & 0xFF == ord("q"):   
        break

