from __future__ import division
from subprocess import call
import os
from os import path

ver='25'
preset='livertumors'
dir='D:\\alex\\images'
#samplesList=os.path.join(dir,preset,'samplesList.txt')###
samplesList='C:\\caffe\\livertumors\\25\\test-cv-3-4.txt' ### test-cv-3-4.txt
exeClass='D:/alex/nets_pancreas/caffe-win-1029/bin/class_new.exe'
exeValid='D:/alex/CNN-build/validation/Release/validationRetVOE.exe'
deploy='C:/caffe/'+preset+'/'+ver+'/deploy.prototxt'
startX='0'
startY='0'
startZ='0'
sizeX='512'
sizeY='512'
sizeZ='1000'
r='32'
spacing='0.782'###
batchLength='1024'
groupX='3'
groupY='3'
classCount='2'###
deviceId='0'
suffix='-v'+ver+'-g'+groupX+'x'+groupY+'-c'+classCount+'-s'+spacing

mask = 'liver.nrrd'#preset+'.nrrd-boundingRec-r5.nrrd'

unbuffered=0
outFile = open('iterationsVOETrain'+preset+'-test'+suffix+'.txt', 'w', unbuffered)### todo remove 'test'
snapshotPrefix='D:/artem/caffe/snap'

for iter in range(5000,150000,5000):
    print iter
    model=os.path.join(snapshotPrefix, preset, ver,'_iter_'+str(iter)+'.caffemodel')
    sumVOE=0
    countVOE=0
    with open(samplesList) as f:
        for line in f:
             path = line.replace('\n','')
             outputImage=os.path.join(path,preset+suffix+'.nrrd')
             args=[exeClass,deploy,model,startX,startY,startZ,sizeX,sizeY,sizeZ,r,preset,spacing,batchLength,groupX,groupY,classCount,os.path.join(path,'patient.nrrd'),os.path.join(path,mask),outputImage,deviceId]
             #print args
             call(args)
             voe=call([exeValid,  '-testImage', outputImage, '-label', os.path.join(path,preset+'.nrrd')])
             print 'voe: '+str(voe)
             sumVOE+=voe
             countVOE+=1
    outFile.write(str(iter)+' '+str(sumVOE/countVOE)+'\n')