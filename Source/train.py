import cv2
import numpy as np
from dataset import GetImgAndLabel
from eval import Accurate

recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer = cv2.face.EigenFaceRecognizer_create()
#recognizer = cv2.face.FisherFaceRecognizer_create()

if __name__ == '__main__':
	faces,Ids=GetImgAndLabel('train.txt')
	recognizer.train(faces,np.array(Ids))
	recognizer.save('cvdata/train.yml')
	print('Training complete with accurate: '+ str(Accurate('validate.txt'))+'%')