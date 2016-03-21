import os
import subprocess
from subprocess import call
import errno
import itertools
import sys
import random

classCount = '2'
cut = 'D:\\alex\\CNN-build\\utils\\Release\\CutImageByTiles.exe'
makeTileLists = ['python.exe', 'make_sample_names_' + classCount + '-classes.py']
train = 'C:\\caffe\\bin\\caffe_cc35.exe'
clas = 'D:\\alex\\nets_pancreas\\caffe-win-1029\\bin\\class_new.exe'
postproc = 'D:\\alex\\CNN-build\\postprocessing\\Release\\postprocessing.exe'
valid = 'D:\\alex\\CNN-build\\validation\\Release\\Validation.exe'

preset = 'livertumors'
ver = '25'
spacing = '0.782'
spacingStr = 'orig' if spacing=='0' else spacing

deviceId = '1'
iters = '250000'

dir=os.path.join('C:/caffe', preset, ver)
print dir

r = '32'
size = int(r) * 2
label = preset + '.nrrd'
mask = 'liver.nrrd'
patient = 'patient.nrrd'
  
imagesPath = 'D:\\alex\\images'
tilesFolder = os.path.join('D:\\alex\\tiles', preset, str(size) + 'x' + str(size), 'sampling-' + spacingStr)
samplesList = os.path.join(imagesPath, preset, 'samplesListGood.txt')

deploy = os.path.join('C:\\caffe', preset, ver, 'deploy.prototxt')
start = [0, 0, 0]
size = [512, 512, 1000]
batchLength = '1024'
groupX = '3'
groupY = '3'
print preset,',v-', ver, ',spacing-', spacing

snaphotPrefix='D:\\Artem\\caffe\\snap'
sigma = '4.000000'


n = 4  # number of parts for cross-validation

def main():

    #retcode = subprocess.call([cut, '-listFile', samplesList, '-imageName', patient, '-labelName1', label,
    #                 '-labelName2', 'livertumors_dark.nrrd', '-maskName', mask, '-radius', r, '-preset', preset,
    #                 '-stride 0 0 0', '-spacingXY', spacing, spacing, '-strideNegative 1', '-folder', tilesFolder])
    #
    #if (retcode != 0):
    #    print 'error. ', cut, ' exit with ', retcode
    #    return

    # make lists
    success = makeSampleNames2Classes(tilesFolder, dir, imagesPath,samplesList, n)
    if not success:
        print 'error. ', cut, ' exit with ', retcode
        return
        
    for k in range(2, n):
        kn = str(k) + '-' + str(n)
        # create folder for snapshots
        snapshotFolder = os.path.join(snaphotPrefix, preset, ver, kn)
        try:
            os.makedirs(snapshotFolder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise e
            pass
                
        model = os.path.join(snapshotFolder, '_iter_' + iters + '.caffemodel')

    
        sampleTrainListK = os.path.join(dir, 'train-cv-' + kn + '.txt')
        sampleTestListK = os.path.join(dir, 'test-cv-' + kn + '.txt')

        tileTrainListK = os.path.join(dir, 'tileList-train-cv-' + kn + '.txt')
        tileTrainTestListK = os.path.join(dir, 'tileList-train-test-cv-' + kn + '.txt')
        tileTestTest5050ListK = os.path.join(dir, 'tileList-test-test-50-50-cv-' + kn + '.txt')
        tileTestTestListK = os.path.join(dir, 'tileList-test-test-cv-' + kn + '.txt')

        # make net and solver
        solver = createNetAndSolver(kn, tileTrainTestListK, tileTrainListK, tileTestTest5050ListK, tileTestTestListK, iters, snapshotFolder)
        
        subprocess.call([train, 'train', '--solver', solver, '--gpu', deviceId], stdout = open('log'+kn+'.txt', 'w')) #not work on windows

        suffix = '-v' + ver + '-g' + groupX + 'x' + groupY + '-c' + classCount + '-s' + spacing + '-cv' + kn + '.nrrd'
        outputCNN = preset + '-cnn' + suffix

        with open(samplesList) as f:
            for line in f:
                line=line.replace('\n','')
                args=[clas, deploy, model, str(start[0]), str(start[1]), str(start[2]), str(size[0]), str(size[1]), str(size[2]), r,
                preset, spacing, batchLength, groupX, groupY, classCount, os.path.join(line, patient),
                os.path.join(line, mask), os.path.join(line, outputCNN), deviceId]
                print args
                subprocess.call(args)
        
        #validate(sampleTrainListK, sampleTestListK, label, outputCNN, suffix)          

        with open(samplesList) as f:
            for line in f:
                line = line.replace('\n','')
                subprocess.call([postproc, '-image', os.path.join(line, outputCNN), '-gaussianVariance', sigma])
        
        suffix = suffix + '-gaussian' + sigma + '.nrrd'
        outputCNN = preset + '-cnn' + suffix

        validate(sampleTrainListK, sampleTestListK, label, outputCNN, suffix)
    return

def makeSampleNames2Classes(pathTiles, pathOut, pathImages, samplesListFile, n):
    #numbers present for each class
    testTrainRate = 0.1
    negPosRate = 3 # n negative for each positive sample
    classCount = 2 #0 and 1
    ###
    
    print 'pathTiles:', pathTiles
    print 'pathOut:', pathOut
    print 'pathImages:', pathImages
    print 'samplesListFile:', samplesListFile
    print 'n:', n
    
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
    
    for k in range(0, n):
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
     
        testCount = int(allCount * testTrainRate/classCount)
        print 'testCount:'+str(testCount)
        
        if posCount < testCount or negCount < testCount:
            print 'too few elements. Count must be at least more than testCount'
            return False
   
        print 'used pos:'+str(posCount)
        print 'used neg:'+str(negCount)

        pos = random.sample(posList, posCount)
        neg = random.sample(negList, negCount)
       
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

def validate(samplesTrainListK, samplesTestListK, label, outputCNN, suffix):
    with open(samplesTrainListK) as f:
        for line in f:
            line = line.replace('\n','')
            subprocess.call([valid, '-testImage', os.path.join(line, outputCNN), '-label', os.path.join(line, label),
                             '-report', 'report-cnn-train-' + preset + '-' + suffix + '.csv'])
    with open(samplesTestListK) as f:
        for line in f:
            line = line.replace('\n','')
            subprocess.call([valid, '-testImage', os.path.join(line, outputCNN), '-label', os.path.join(line, label),
                             '-report', 'report-cnn-test-'  + preset + '-'+ suffix + '.csv'])
    return valid

def createNetAndSolver(kn, tileTrainTestListK, tileTrainListK, tileTestTest5050ListK, tileTestTestListK, iters, snapshotFolder):
    netTemplate = os.path.join(dir, 'netTemplate.prototxt')
    net = os.path.join(dir, 'net-cv-' + kn + '.prototxt')
    fileData = None
    with open(netTemplate, 'r') as file:
        fileData = file.read()
    fileData = fileData \
        .replace('%trainTilesList%', tileTrainListK.replace('\\','/')) \
        .replace('%trainTestTilesList%', tileTrainTestListK.replace('\\','/')) \
        .replace('%testTest50-50TilesList%', tileTestTest5050ListK.replace('\\','/')) \
        .replace('%testTestTilesList%', tileTestTestListK.replace('\\','/'))

    with open(net, 'w') as file:
        file.write(fileData)
    print 'net ' + net + ' has been created'
    
    solverTemplate = os.path.join(dir, 'solverTemplate.prototxt')
    solver = os.path.join(dir, 'solver-cv-' + kn + '.prototxt')
    fileData = None
    with open(solverTemplate, 'r') as file:
        fileData = file.read()
    fileData = fileData \
        .replace('%net%', net.replace('\\','/')) \
        .replace('%iters%', iters) \
        .replace('%snapshotFolder%', snapshotFolder.replace('\\','/')+'/')
    with open(solver, 'w') as file:
        file.write(fileData)
    print 'solver ' + solver + ' has been created'

    return solver

if __name__ == "__main__":
    main()