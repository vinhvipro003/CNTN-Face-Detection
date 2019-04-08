import cv2
import numpy as np 
import config

recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer = cv2.face.EigenFaceRecognizer_create()
#recognizer = cv2.face.FisherFaceRecognizer_create()
def Accurate(validate_dir):
	train_dir = 'cvdata/train.yml'
	recognizer.read(train_dir)
	f=open(validate_dir)
	lines=f.readlines()
	testCount = 0
	check = 0
	for line in lines:
		img_dir, realID = line.split("  ")
		realID=int(realID)
		img = cv2.imread(img_dir)
		gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		fixed = cv2.resize(gray, (config.FIXED_SIZE, config.FIXED_SIZE))
		Id, conf = recognizer.predict(fixed)
		testCount = testCount + 1
		if (Id == realID):
			check = check + 1
	return float(check)/testCount*100
