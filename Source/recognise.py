import cv2
import numpy as np
import config
import sys

recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer = cv2.face.EigenFaceRecognizer_create()
#recognizer = cv2.face.FisherFaceRecognizer_create()

recognizer.read('cvdata/train.yml')
faceCascade = cv2.CascadeClassifier("cvdata/haarcascade_frontalface_default.xml");
font = cv2.FONT_HERSHEY_SIMPLEX
def recognise(img_dir, uid_dir):
	img=cv2.imread(img_dir)
	f=open(uid_dir)
	uid=f.readlines()
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces=faceCascade.detectMultiScale(gray, 1.2, 5)
	for(x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(225,0,0),2)
		fixed = cv2.resize(gray[y:y+h,x:x+w], (config.FIXED_SIZE, config.FIXED_SIZE))
		Id, conf = recognizer.predict(fixed)
		if(conf <= 50):
			Id=uid[Id]
		else:
			Id="Unknown"
		cv2.putText(img,str(Id),(x,y-10), font, 2,(255,255,255),2,cv2.LINE_AA)
	cv2.imshow('image',img)
	if cv2.waitKey(10000):
		cv2.destroyAllWindows()


if __name__ == '__main__':
#	f=open('test.txt')
#	lines=f.readlines()
#	for line in lines:
#		img_dir,ID=line.split("  ")
#		recognise(img_dir,'uid.txt')
	recognise(sys.argv[1],'uid.txt')