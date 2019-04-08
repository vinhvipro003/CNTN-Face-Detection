import os
import numpy as np
import sys
import cv2

import config

def GetImgAndLabel(train_dir):
	Ids=[]
	faces=[]
	f=open(train_dir)
	lines=f.readlines()
	for line in lines:
		img_dir,ID=line.split("  ")
		img=cv2.imread(img_dir)
		gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		fixed = cv2.resize(gray, (config.FIXED_SIZE, config.FIXED_SIZE))
		imageNp=np.array(fixed,'uint8')
		faces.append(imageNp)
		Ids.append(int(ID))
	return faces,Ids
