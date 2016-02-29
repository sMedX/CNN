import os 
import itertools
import sys
import random
import glob 


class ClassData:
	def __init__(self, path, fileList, testIter, trainIter):
		self.path = path
		self.fileList = fileList
		self.testIter = testIter
		self.trainIter = trainIter

def main():

	classCount = 4
	testf = open('./test_newtum_all_4.txt', 'w')
	trainf = open('./train_newtum_all_4.txt', 'w')
	#l = range(20)
	#print >> testf, l[3:]
	#return
	classPathes = []

	classPathes.append('D:/alex/tiles/livertumors/64x64/sampling-0782-ada2/*/TN/*.png')
	classPathes.append('D:/alex/tiles/livertumors/64x64/sampling-0782-ada2/*/FP/*.png')
	classPathes.append('D:/alex/tiles/livertumors/64x64/sampling-0782-ada2/*/FN/*.png')
	classPathes.append('D:/alex/tiles/livertumors/64x64/sampling-0782-ada2/*/TP/*.png')

	classes = [];

	minCount = 10000000
	testCount = 5000
	for path in classPathes:
		#fileList = os.listdir(path)
		fileList = glob.glob(path)
		
		count = len(fileList)
		fileList = random.sample(fileList, count)
		classData = ClassData(path, fileList, iter(fileList[ : testCount]), iter(fileList[testCount : ])); 
		classes.append(classData)
		
		if count < minCount:
			minCount = count;
		print(path)		
		print(count)		

	print(testCount)
	for file0 in classes[0].fileList[ : testCount]:
		print >>testf, file0 + ' 0'		
		#print >>trainf, file0 + ' 0'		
		for i in range(1, classCount):
			fileI = next(classes[i].testIter, None)	
			if fileI != None:
				print >>testf, fileI + ' ' + str(i)
				#print >>trainf, classes[i].path + '/' + fileI + ' ' + str(i)

	#trainCount = minCount
	trainCount = 160000

	for file0 in classes[0].fileList[ : trainCount]:
		print >>trainf, file0 + ' 0'		
		for i in range(1, classCount):
			fileI = next(classes[i].trainIter, None)	
			if fileI != None:
				print >>trainf, fileI + ' ' + str(i)


if __name__ == "__main__":
    main()