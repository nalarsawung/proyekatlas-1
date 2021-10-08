import os
import cv2

def get_id(folder='dataset'):
	alldata = [i for i in os.listdir(folder)]
	print(alldata)
	alldata.sort()
	ids = 0
	if alldata:
		for i in alldata:
			tempid = i.split('.')[1]
			ids = int(tempid) + 1
	return ids

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_id = get_id()
print(f'Saving into id {face_id}')
count = 0
while(True):
	ret, img = cam.read()
	# img = cv2.flip(img, -1) # flip video image vertically
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_detector.detectMultiScale(gray, 1.5, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)	 
		count += 1
		cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
		cv2.imshow('Training Face', img)
	k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
	if k == 27:
		break
	elif count >= 20:
		 break
cam.release()
cv2.destroyAllWindows()