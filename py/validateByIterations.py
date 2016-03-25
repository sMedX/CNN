from __future__ import division
from subprocess import call
import os
from os import path

# params
ver='2'
preset='liver'
sampleListName='test-cv-3-4.txt'
spacing='1.5'
classCount='3'
mask = preset+'.nrrd-dilate-r64.nrrd'
sigma='4.000000' #format is important for cpp code
forceOverwrite = False
kn='3-4'

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
deviceId='1'
suffix='-v'+ver+'-g'+groupX+'x'+groupY+'-c'+classCount+'-s'+spacing
snapshotPrefix='D:/artem/caffe/snap'

unbuffered=0
outFile = open('VOEByIters-'+preset+suffix+'-'+sampleListName.replace('.txt','')+'.csv', 'w', unbuffered)

outFile.write('iter; avg VOE class; avg VOE largest Object, avg VOE Smoothed;\n')

for iter in range(5000,80000,5000):
    print iter
    snapshotFolder = os.path.join(snapshotPrefix, preset, ver, kn)
    model = os.path.join(snapshotFolder, '_iter_' + str(iter) + '.caffemodel')

    sumVOE=[0, 0, 0] ###
    count=0
    with open(samplesList) as f:
        for line in f:
            path = line.replace('\n','')
            outputImage=os.path.join(path,preset+suffix+'-it'+str(iter)+'.nrrd')
            if forceOverwrite or not os.path.isfile(outputImage):
                args=[exeClass,deploy,model,startX,startY,startZ,sizeX,sizeY,sizeZ,r,preset,spacing,batchLength,groupX,groupY,classCount,os.path.join(path,'patient.nrrd'),os.path.join(path,mask),outputImage,deviceId]
                #print args
                call(args)

            postprocessedImage1 = outputImage+'-largestObject.nrrd'
            postprocessedImage2 = outputImage+'-gaussian'+sigma+'.nrrd'
            if forceOverwrite or not os.path.isfile(postprocessedImage1) or not os.path.isfile(postprocessedImage2):
                args = [postproc, '-image', os.path.join(line, outputImage), '-gaussianVariance', sigma, '-preset', preset]
                print args
                call(args)  
            
            voe=[0,0,0]
            voe[0]=call([exeValid,  '-testImage', outputImage, '-label', os.path.join(path,preset+'.nrrd')])
            voe[1]=call([exeValid,  '-testImage', postprocessedImage1, '-label', os.path.join(path,preset+'.nrrd')])
            voe[2]=call([exeValid,  '-testImage', postprocessedImage2, '-label', os.path.join(path,preset+'.nrrd')])
          
            print 'voe: ', voe
            for i in range(0, 3):
                sumVOE[i]+=voe[i]
            count+=1
            
            print 'avg voe', sumVOE[0] / count,  sumVOE[1] / count,  sumVOE[2] / count
    outFile.write(str(iter)+';'+str(sumVOE[0]/count)+';'+str(sumVOE[1]/count)+';'+str(sumVOE[2]/count)+'\n')