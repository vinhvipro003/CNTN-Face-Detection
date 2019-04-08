import glob
import random
import os
import numpy as np
import sys

import config

def init_dataset_meta(data_dir, ptrain = config.PTRAIN, pvalidate = config.PVAL, ptest = config.PTEST):
	if not os.path.exists(data_dir):
		data_dir = config.DATASET_DIR
	uid = os.listdir(data_dir)
	uid.sort()
	f1 = open('uid.txt', 'w')
	for i in range(0, len(uid)):
		f1.write(uid[i] + '\n')
	f1.close()

	path = []
	for i in range(0,len(uid)):
		path1 = []
		imgPath = os.listdir(data_dir + '\\' + uid[i])
		for j in imgPath:
			path1.append(data_dir + '\\' + uid[i] + '\\' + j+'  '+str(i))
		random.shuffle(path1)
		path.append(path1)
	f2 = open('train.txt', 'w')
	f3 = open('validate.txt', 'w')
	f4 = open('test.txt', 'w')
	for i in range(0, len(path)):
		for j in range(0, ptrain):
			f2.write(path[i][j] + '\n')

		for j in range(ptrain, ptrain + pvalidate):
			f3.write(path[i][j] + '\n')
		

		for j in range(ptrain + pvalidate, 9):
			f4.write(path[i][j] + '\n')
	f2.close()
	f3.close()
	f4.close()


if __name__ == '__main__':
	init_dataset_meta(sys.argv[1])
