#from __future__ import division

import os 
import itertools
import sys
import random

def main(pathTiles, pathOut, pathImages, samplesListFile, n):
    ###
    #numbers present for each class
    testTrainRate = 0.1
    negPosRate = 3 # n negative for each positive sample
    classCount = 2 #0 and 1
    ###
    
    preset = pathOut.replace('\\', '/').split('/')[-2]
    print 'preset: ', preset
    
 
    #make n parts
    requestedSamples = []
    with open(samplesListFile) as f:
        for line in f:
            line=line.replace('\n','').replace('\\','/').split('/')[-1]
            requestedSamples.append(line)
    print 'requestedSamples:', requestedSamples
    
    print 'existing dirs:', os.listdir(pathTiles) 
    if (any([dir not in os.listdir(pathTiles) for dir in requestedSamples])):
        print 'at least one of requested samples epsent in tile folder'
        return False
   
    dirsInPart = int(len(requestedSamples) / n)
    parts = []
    for i in range(0,n):
        if (i!=n-1):
            partI = requestedSamples[i*dirsInPart:(i+1)*dirsInPart]
            parts.append(partI)
        else: # rest elements goes to the last part
            partI = requestedSamples[i*dirsInPart:]
            parts.append(partI)
    print 'parts:', parts
    
    for k in range(2, n):
        print '====================================='
        print 'part #', k
        testDirs=parts[k]
        trainDirs=[]
        for i in range(0,n):
            if (i!=k):
                trainDirs.extend(parts[i])
        print 'testDirs', testDirs
        print 'trainDirs', trainDirs
        
        with open(os.path.join(pathOut, 'test-cv-'+str(k)+'-'+str(n)+'.txt'), 'w') as sampleTestf:
            for dir in testDirs:
                print >>sampleTestf, os.path.join(pathImages, preset, dir)
        with open(os.path.join(pathOut, 'train-cv-'+str(k)+'-'+str(n)+'.txt'), 'w') as sampleTrainf:
            for dir in trainDirs:
                print >>sampleTrainf, os.path.join(pathImages, preset, dir)
        
        negPathTiles = []
        posPathTiles = []
        
        for sample in trainDirs:
            negPathTiles.append(os.path.join(pathTiles,sample,'0'))
            posPathTiles.append(os.path.join(pathTiles,sample,'1'))
        
        posList = []
        negList = []
        
        for path in posPathTiles:
            for file in os.listdir(path):
                posList.append(os.path.join(path,file))
        for path in negPathTiles:
            for file in os.listdir(path):
                negList.append(os.path.join(path,file))

        posCount = len(posList)
        negCount = len(negList)
        allCount = posCount + negCount
        
        print 'all:'+str(allCount)
        print 'all pos:'+str(posCount)
        print 'all neg:'+str(negCount)

        posCount = min([int(allCount * 1 / (negPosRate + 1)), posCount])
        negCount = min([int(allCount*negPosRate / (negPosRate + 1)), negCount])
        
        print 'used pos:'+str(posCount)
        print 'used neg:'+str(negCount)

        pos = random.sample(posList, posCount)
        neg = random.sample(negList, negCount)
       
        testCount = int(allCount * testTrainRate/classCount)
        print 'testCount:'+str(testCount)
        
        if pos < testCount or neg < testCount:
            print 'to few elements. Count must be at least more than testCount'
            continue

        kn=str(k)+'-'+str(n)
        #test on train
        with open(os.path.join(pathOut, 'tileList-train-test-cv-'+kn+'.txt'), 'w') as testf:    
            it = iter(neg[ : testCount])
            for posfile in pos[ : testCount]:
                negfile = next(it)    
                print >>testf, posfile + ' 1'        
                print >>testf, negfile + ' 0'
        #train
        with open(os.path.join(pathOut, 'tileList-train-cv-'+kn+'.txt'), 'w') as trainf:
            it = iter(neg[testCount :])
            for posfile in pos[testCount :]:
                print >>trainf, posfile + ' 1'
                for i in range(negPosRate):
                    negfile = next(it)    
                    print >>trainf, negfile + ' 0'
        #test on test
        print 'test: '+str(testCount*classCount)
        negPathTiles = []
        posPathTiles = []
        
        for sample in testDirs:
            negPathTiles.append(os.path.join(pathTiles,sample,'0'))
            posPathTiles.append(os.path.join(pathTiles,sample,'1'))
        
        posList = []
        negList = []
        
        for path in posPathTiles:
            for file in os.listdir(path):
                posList.append(os.path.join(path,file))
        for path in negPathTiles:
            for file in os.listdir(path):
                negList.append(os.path.join(path,file))

        pos = random.sample(posList, min([testCount, len(posList)])) 
        neg = random.sample(negList, min([testCount, len(negList)]))
        
        print 'used test pos: ' + str(len(pos))
        print 'used test neg: ' + str(len(neg))
        
        with open(os.path.join(pathOut, 'tileList-test-test-cv-'+kn+'.txt'), 'w') as testf:
            it = iter(neg[ : testCount])
            for posfile in pos[ : testCount]:
                negfile = next(it)    
                print >>testf, posfile + ' 1'        
                print >>testf, negfile + ' 0'
    return True

if __name__ == "__main__":
    print 'Number of arguments:', len(sys.argv), 'arguments.'
    print 'Argument List:', str(sys.argv)
    
    pathTiles = sys.argv[1]
    pathOut = sys.argv[2]
    pathImages = sys.argv[3]
    samplesListFile = sys.argv[4]
    n = int(sys.argv[5])
    
    print 'pathTiles:', pathTiles
    print 'pathOut:', pathOut
    print 'pathImages:', pathImages
    print 'samplesListFile:', samplesListFile
    print 'n:', n
        
    success = main(pathTiles, pathOut, pathImages,samplesListFile, n)
    if not success:
        sys.exit(1)