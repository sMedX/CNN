#from __future__ import division

import os 
import itertools
import sys
import random
import glob 


class ClassData:
    def __init__(self, pathTiles, fileList, testIter, trainIter):
        self.pathTiles = pathTiles
        self.fileList = fileList
        self.testIter = testIter
        self.trainIter = trainIter

def main(pathTiles, pathOut, pathImages, k, n):
    if (k >= n):
        print 'error. k must be less than n'
        return
    
    classCount = 3
    preset = pathOut.replace('\\', '/').split('/')[-2]
    print 'preset: ', preset

    #make n parts
    allDirs=os.listdir(pathTiles)
    dirsInPart = int(len(allDirs) / n)
    parts = []
    for i in range(0,n):
        if (i!=n-1):
            partI = allDirs[i*dirsInPart:(i+1)*dirsInPart]
            parts.append(partI)
        else: # rest elements goes to the last part
            partI = allDirs[i*dirsInPart:]
            parts.append(partI)
    print 'parts:', parts
    
    testDirs=parts[k]
    trainDirs=[]
    for i in range(0,n):
        if (i!=k):
            trainDirs.extend(parts[i])
    print 'testDirs', testDirs
    print 'trainDirs', trainDirs
    
    sampleTestf = open(os.path.join(pathOut, 'test-cv-'+str(k)+'-'+str(n)+'.txt'), 'w')
    sampleTrainf = open(os.path.join(pathOut, 'train-cv-'+str(k)+'-'+str(n)+'.txt'), 'w')
    for dir in testDirs:
        print >>sampleTestf, os.path.join(pathImages, preset, dir)
    for dir in trainDirs:
        print >>sampleTrainf, os.path.join(pathImages, preset, dir)
        
    classes = [];

    minCount = 10000000
    testCount = 5000
    for iClass in range(0,classCount):
        pathTilesI = os.path.join(pathTiles, '*', str(iClass), '*.png')
        
        print 'class:', str(iClass)
        print 'pathTiles:', pathTilesI
        
        #fileList = [file for file in glob.glob(pathTilesI) if any(name in file for name in trainDirs)] #filter for
        fileList = glob.glob(pathTilesI)
        
        print 'filelist count:', str(len(fileList))
        
        count = len(fileList)
        fileList = random.sample(fileList, count)
        classData = ClassData(pathTilesI, fileList, iter(fileList[ : testCount]), iter(fileList[testCount : ]))
        classes.append(classData)
        
        if count < minCount:
            minCount = count;        

    testf = open(os.path.join(pathOut, 'tileList-test-cv-'+str(k)+'-'+str(n)+'.txt'), 'w')
    trainf = open(os.path.join(pathOut, 'tileList-train-cv-'+str(k)+'-'+str(n)+'.txt'), 'w')
    
    for file0 in classes[0].fileList[ : testCount]:
        print >>testf, file0 + ' 0'        
        #print >>trainf, file0 + ' 0'        
        for i in range(1, classCount):
            fileI = next(classes[i].testIter, None)    
            if fileI != None:
                print >>testf, fileI + ' ' + str(i)
                #print >>trainf, classes[i].pathTiles + '/' + fileI + ' ' + str(i)

    #trainCount = minCount
    trainCount = 2000000

    for file0 in classes[0].fileList[ : trainCount]:
        print >>trainf, file0 + ' 0'        
        for i in range(1, classCount):
            fileI = next(classes[i].trainIter, None)    
            if fileI != None:
                print >>trainf, fileI + ' ' + str(i)

    return

    it = iter(notum[testCount :])
    for tumfile in tum[testCount :]:
        print >>trainf,tumfile + ' 1'        
        notumfile = next(it)    
        print >>trainf, notumfile + ' 0'
        notumfile = next(it)    
        print >>trainf, notumfile + ' 0'
        notumfile = next(it)    
        #print >>trainf, notumfile + ' 0'
        #notumfile = next(it)    
        #print >>trainf, notumfile + ' 0'
                
    return


if __name__ == "__main__":
    print 'Number of arguments:', len(sys.argv), 'arguments.'
    print 'Argument List:', str(sys.argv)
    
    pathTiles = sys.argv[1]
    pathOut = sys.argv[2]
    pathImages = sys.argv[3]
        
    print 'pathTiles:', pathTiles
    print 'pathOut:', pathOut
    print 'pathImages:', pathImages
    
    if (len(sys.argv)>5):
        k = int(sys.argv[4])
        n = int(sys.argv[5])
    else:
        k=0
        n=1
        
    main(pathTiles, pathOut, pathImages, k, n)