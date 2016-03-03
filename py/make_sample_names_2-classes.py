#from __future__ import division

import os 
import itertools
import sys
import random

def main(pathTiles, pathOut, pathImages, k, n):
    ###
    #numbers present for each class
    itemCount = 2000000 # all sample count. i.e. pos + negPosRate*neg
    testTrainRate = 0.1
    negPosRate = 3 # n negative for each positive sample
    classCount = 2 #0 and 1
    ###
    
    if (k >= n):
        print 'error. k must be less than n'
        return

    preset = pathOut.replace('\\', '/').split('/')[-2]
    print 'preset: ', preset
    
    testCount = int(itemCount * testTrainRate/classCount)
    print 'testCount:'+str(testCount)
    
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
    
    negpathTileses = []
    pospathTileses = []
    
    for sample in trainDirs:
        negpathTileses.append(os.path.join(pathTiles,sample,'0'))
        pospathTileses.append(os.path.join(pathTiles,sample,'1'))
    
    posList = []
    negList = []
    
    for pathTiles in pospathTileses:
        for file in os.listdir(pathTiles):
            posList.append(os.path.join(pathTiles,file))
    for pathTiles in negpathTileses:
        for file in os.listdir(pathTiles):
            negList.append(os.path.join(pathTiles,file))
     
    print posList[0]
    posCount = len(posList)
    negCount = len(negList)
    
    print 'all pos:'+str(posCount)
    print 'all neg:'+str(negCount)

    posCount = int(itemCount * 1 / (negPosRate + 1))
    negCount = int(itemCount*negPosRate / (negPosRate + 1))
    
    print 'used pos:'+str(posCount)
    print 'used neg:'+str(negCount)

    pos = random.sample(posList, posCount + testCount)
    neg = random.sample(negList, negCount + testCount)
    
    testf = open(os.path.join(pathOut, 'tileList-test-cv-'+str(k)+'-'+str(n)+'.txt'), 'w')
    trainf = open(os.path.join(pathOut, 'tileList-train-cv-'+str(k)+'-'+str(n)+'.txt'), 'w')
    
    it = iter(neg[ : testCount])
    for posfile in pos[ : testCount]:
        negfile = next(it)    
        print >>testf, posfile + ' 1'        
        print >>testf, negfile + ' 0'

    it = iter(neg[testCount :])
    for posfile in pos[testCount :]:
        print >>trainf, posfile + ' 1'
        for i in range(negPosRate):
            negfile = next(it)    
            print >>trainf, negfile + ' 0'         
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