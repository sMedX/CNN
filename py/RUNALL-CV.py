import os
import subprocess
from subprocess import call
import errno
import itertools
import sys
import random

classCount = '3'
cut = 'D:\\alex\\CNN-build\\utils\\Release\\CutImageByTiles.exe'
makeTileLists = ['python.exe', 'make_sample_names_' + classCount + '-classes.py']
train = 'C:\\caffe\\bin\\caffe_cc35.exe'
clas = 'D:/alex/caffe-ms/caffe-master/Build/x64/Release/caffe.exe'
postproc = 'D:\\alex\\CNN-build\\postprocessing\\Release\\postprocessing.exe'
valid = 'D:\\alex\\CNN-build\\validation\\Release\\Validation.exe'

preset = 'livertumors'
ver = '28'
spacing = '0.782'
spacingStr = 'orig' if spacing=='0' else spacing
tilesParam = '-3class-3dircad'

deviceId = '1'
iters = '45000'

dir=os.path.join('C:/caffe', preset, ver)
print dir

r = '32'
size = int(r) * 2
label = preset + '.nrrd'
mask = 'liver.nrrd'
patient = 'patient.nrrd'
  
imagesPath = 'D:\\alex\\images'
tilesFolder = os.path.join('D:\\alex\\tiles', preset, str(size) + 'x' + str(size), 'sampling-' + spacingStr + tilesParam)
samplesList = os.path.join(imagesPath, preset, 'samplesList.txt')

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
    #success = makeSampleNames(int(classCount), tilesFolder, dir, imagesPath,samplesList, n)
    #if not success:
    #    print 'error. ', cut, ' exit with ', retcode
    #    return
        
    for k in range(0, n):
        kn = str(k) + '-' + str(n)
        # create folder for snapshots
        snapshotFolder = os.path.join(snaphotPrefix, preset, ver)###, kn)
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
        tileTestTestListK = os.path.join(dir, 'tileList-test-test-cv-' + kn + '.txt')

        #make net and solver
        solver = createNetAndSolver(kn, tileTrainTestListK, tileTrainListK, tileTestTestListK, iters, snapshotFolder)
        with open('log'+kn+'.txt', 'w') as log:
            subprocess.call([train, 'train', '--solver', solver, '--gpu', deviceId], stdout = log, stderr = log)

        suffix = '-v' + ver + '-g' + groupX + 'x' + groupY + '-c' + classCount + '-s' + spacing + '-cv' + kn + '.nrrd'
        outputCNN = preset + '-cnn' + suffix

        with open(samplesList) as f:
            for line in f:
                line=line.replace('\n','')
                args=[clas, deploy, model, str(start[0]), str(start[1]), str(start[2]), str(size[0]), str(size[1]), str(size[2]), r,
                preset, spacing, batchLength, groupX, groupY, classCount, os.path.join(line, patient),
                os.path.join(line, mask), os.path.join(line, outputCNN), deviceId]
                print args
                #subprocess.call(args)

        #validate(sampleTrainListK, sampleTestListK, label, outputCNN, suffix)
        
        with open(samplesList) as f:
            for line in f:
                line = line.replace('\n','')
                #subprocess.call([postproc, '-image', os.path.join(line, outputCNN), '-gaussianVariance', sigma, '-preset', preset])
        
        suffix = suffix + '-gaussian' + sigma + '.nrrd'
        outputCNN = preset + '-cnn' + suffix

        #validate(sampleTrainListK, sampleTestListK, label, outputCNN, suffix)
    return

def makeSampleNames(classCount, pathTiles, pathOut, pathImages, samplesListFile, n):
    #numbers present for each class
    testTrainRate = 0.1
    stride = [4, 1, 1]
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
       
        kn=str(k)+'-'+str(n)       
        with open(os.path.join(pathOut, 'test-cv-'+kn+'.txt'), 'w') as sampleTestf:
            for dir in testDirs:
                print >>sampleTestf, os.path.join(pathImages, preset, dir)
        with open(os.path.join(pathOut, 'train-cv-'+kn+'.txt'), 'w') as sampleTrainf:
            for dir in trainDirs:
                print >>sampleTrainf, os.path.join(pathImages, preset, dir)
        
        allPathTiles = []
        list = []
        count = []
        allCount = 0
        for i in range(0, classCount):
            iPathTiles = [] #list of dirs for current class
            for sample in trainDirs:
                iPathTiles.append(os.path.join(pathTiles,sample,str(i)))
            allPathTiles.append(iPathTiles)
        
            iList = [] #list of files for current class
            for path in iPathTiles:
                for file in os.listdir(path):
                   iList.append(os.path.join(path,file))
            list.append(iList)
            
            iCount = len(iList)
            print 'class ', str(i), ' count: ', str(iCount)
            iCount /= stride[i] 
            print 'used only ', str(iCount)
            count.append(iCount)
            allCount += iCount
       
        print 'used summary count:', str(allCount)
     
        testCount = int(allCount * testTrainRate)
        print 'testCount:', str(testCount)
        iTestCount = testCount / classCount
        
        for i in range(0, classCount):
            if count[i] < iTestCount:
                print 'too few elements in class ', str(i)
                print 'Count must be more than iTestCount'
                return False
   
            iSample = random.sample(list[i], count[i])

            #test on train
            with open(os.path.join(pathOut, 'tileList-train-test-cv-'+kn+'.txt'), 'w') as testf:    
                for iClassFile in iSample[ : iTestCount]:
                    print >>testf, iClassFile + ' ' + str(i) 
            #train
            with open(os.path.join(pathOut, 'tileList-train-cv-'+kn+'.txt'), 'w') as trainf:
                for iClassFile in iSample[iTestCount :]:
                    print >>trainf, iClassFile  + ' ' + str(i)
                    
            #test on test
            print 'test: '+str(iTestCount)
            #testPathTiles = []
            iPathTiles = []
        
            for sample in testDirs:
                iPathTiles.append(os.path.join(pathTiles,sample,str(i)))
        
            iList = []
            for path in iPathTiles:
                for file in os.listdir(path):
                    iList.append(os.path.join(path,file))

            iSample = random.sample(iList, min([iTestCount, len(iList)])) 
            
            print 'used test of ', str(i), ' class:' + str(len(iSample))
            
            with open(os.path.join(pathOut, 'tileList-test-test-cv-'+kn+'.txt'), 'w') as testf:
                for iClassFile in iSample:
                    print >>testf, iClassFile  + ' ' + str(i)
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

def createNetAndSolver(kn, tileTrainTestListK, tileTrainListK, tileTestTestListK, iters, snapshotFolder):
    netTemplate = os.path.join(dir, 'netTemplate.prototxt')
    net = os.path.join(dir, 'net-cv-' + kn + '.prototxt')
    fileData = None
    with open(netTemplate, 'r') as file:
        fileData = file.read()
    fileData = fileData \
        .replace('%trainTilesList%', tileTrainListK.replace('\\','/')) \
        .replace('%trainTestTilesList%', tileTrainTestListK.replace('\\','/')) \
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