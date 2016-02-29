from __future__ import division
from subprocess import call
import os
from os import path

ver='6'###
preset='pancreas'###
dir='D:\\alex\\images'
exeClass='D:/alex/nets_pancreas/caffe-win-1029/bin/class_new.exe'
exeValid='D:/alex/CNN-build/validation/Release/validationRetVOE.exe'
deploy='C:/caffe/'+preset+'/'+ver+'/deploy.prototxt'
startX='0'
startY='0'
startZ='0'
sizeX='511'
sizeY='511'
sizeZ='1000'
r='32'
spacing='orig'###
batchLength='1024'
groupX='3'
groupY='3'
classCount='2'###
deviceId='0'
suffix='-cnn-'+preset+'-v'+ver+'-g'+groupX+'x'+groupY+'-c'+classCount+'-s'+spacing

mask = preset+'.nrrd-boundingRec-r5.nrrd'

unbuffered=0
outFile = open('iterationsVOETest-'+suffix+'.txt', 'w', unbuffered)

for iter in range(5000,450000,25000):
    print iter
    model=os.path.join('D:/artem/caffe/snap', preset, ver,'_iter_'+str(iter)+'.caffemodel')
    sumVOE=0
    countVOE=0
    with open(os.path.join(dir,preset,'samplesList.txt')) as f:
        for line in f:
             path = line[:-1]
             outputImage=os.path.join(path,preset+suffix+'.nrrd')
             args=[exeClass,deploy,model,startX,startY,startZ,sizeX,sizeY,sizeZ,r,preset,spacing,batchLength,groupX,groupY,classCount,os.path.join(path,'patient.nrrd'),os.path.join(path,mask),outputImage,deviceId]
             #print args
             call(args)
             voe=call([exeValid,  '-testImage', outputImage, '-label', path+'/'+preset+'.nrrd'])
             print 'voe: '+str(voe)
             sumVOE+=voe
             countVOE+=1
    outFile.write(str(iter)+' '+str(sumVOE/countVOE)+'\n')