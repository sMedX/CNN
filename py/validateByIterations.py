from __future__ import division
from subprocess import cal
import os
from os import path

# params
ver='2'
preset='liver'
sampleListName='test-cv-3-4.txt'
spacing='1.5'
classCount='2'
mask = preset+'.nrrd-boundingRec-r5.nrrd'
sigma='4.000000' #format is important for cpp code
forceOverwrite = False

#dir='D:/alex/images'
samplesList=os.path.join('C:/caffe/',preset,ver,sampleListName)
exeClass='D:/alex/nets_pancreas/caffe-win-1029/bin/class_new.exe'
postproc = 'D:/alex/CNN-build/postprocessing/Release/postprocessing.exe'
exeValid='D:/alex/CNN-build/validation/Release/validationRetVOE.exe'
deploy='C:/caffe/'+preset+'/'+ver+'/deploy.prototxt'
startX='0'
startY='0'
startZ='0'
sizeX='512'
sizeY='512'
sizeZ='1000'
r='32'
batchLength='1024'
groupX='3'
groupY='3'
deviceId='0'
suffix='-v'+ver+'-g'+groupX+'x'+groupY+'-c'+classCount+'-s'+spacing
snapshotPrefix='D:/artem/caffe/snap'

unbuffered=0
outFile = open('VOEByIters-'+preset+suffix+'.csv', 'w', unbuffered)

outFile.write('iter; avg VOE class; avg VOE largest Object, avg VOE Smoothed;\n')

for iter in range(5000,150000,5000):
    print iter
    model=os.path.join(snapshotPrefix, preset, ver,'_iter_'+str(iter)+'.caffemodel')
    sumVOE=[0, 0, 0] ###
    count=0
    with open(samplesList) as f:
        for line in f:
            path = line.replace('\n','')
            outputImage=os.path.join(path,preset+suffix+'.nrrd')
            if forceOverwrite or not isfile(outputImage):
                args=[exeClass,deploy,model,startX,startY,startZ,sizeX,sizeY,sizeZ,r,preset,spacing,batchLength,groupX,groupY,classCount,os.path.join(path,'patient.nrrd'),os.path.join(path,mask),outputImage,deviceId]
                #print args
                call(args)

            postprocessedImage1 = outputImage+'-largestObject.nrrd'
            postprocessedImage2 = outputImage+'-gaussian'+sigma+'.nrrd'
            if forceOverwrite or not isfile(outputImage) or not isfile(outputImage):
                call([postproc, '-image', os.path.join(line, outputImage), '-gaussianVariance', sigma, '-preset', preset])  
            
            voe=[0,0,0]
            voe[0]=call([exeValid,  '-testImage', outputImage, '-label', os.path.join(path,preset+'.nrrd')])
            voe[1]=call([exeValid,  '-testImage', postprocessedImage1, '-label', os.path.join(path,preset+'.nrrd')])
            voe[2]=call([exeValid,  '-testImage', postprocessedImage2, '-label', os.path.join(path,preset+'.nrrd')])
          
            print 'voe: ', voe
            sumVOE+=voe
            count+=1
    outFile.write(str(iter)+';'+str(sumVOE[0]/count)+';'+str(sumVOE[1]/count)+';'+str(sumVOE[2]/count)+'\n')