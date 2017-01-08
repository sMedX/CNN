from __future__ import division
import os
import subprocess
from subprocess import call
import errno
import itertools
import sys
import random
import time

# environ
clas = os.environ['clas']
postproc = os.environ['postproc']
valid = os.environ['valid']
python = os.environ['python']

imagesDir = os.environ['images']

def main():
    preset = 'livertumors'
    deviceId = '0'##

    classCount = '2'#3
    r = '64'#'32'
    size = int(r) * 2
    label = preset + '.nrrd'
    mask = 'liver.nrrd'
    patient = 'patient.nrrd'

    samplesList = os.path.join(imagesDir, preset, 'samplesListRelative.txt')

    netDir = 'D:/alex/downloads/20161123-010518-4093_epoch_0.65.tar'
    deploy = os.path.join(netDir, 'deploy.prototxt')
    model = os.path.join(netDir, 'snapshot_iter_67613.caffemodel')
    start = [0, 0, 0]
    size = [512, 512, 1000]
    batchLength = '64'
    groupX = str(3)#*224/64)
    groupY = str(3)#*224/64)
    spacing = str(1.6)#/(224/64))

    sigma = '4.000000'
    sampleTrainList = 'C:/Users/alkor/Documents/test-cv-0-2-rel.txt'
    sampleTestList =  'C:/Users/alkor/Documents/train-cv-0-2-rel.txt'

    #apply
    outputCNN=preset+'_'+netDir.split('/')[-2]+'.nrrd'
    with open(samplesList) as f:
        for line in f:
            line=line.replace('\n','')
            args=[clas, deploy, model, str(start[0]), str(start[1]), str(start[2]), str(size[0]), str(size[1]), str(size[2]), r,
            preset, spacing, batchLength, groupX, groupY, classCount, os.path.join(imagesDir, preset, line, patient),
            os.path.join(imagesDir, preset, line, mask), os.path.join(imagesDir, preset, line, outputCNN), deviceId]
            print args
            subprocess.call(args)
    
    validate(sampleTrainList, sampleTestList, label, outputCNN, preset, '')

    
    #postprocess
    with open(samplesList) as f:
        for line in f:
            line = line.replace('\n','')
            print line
            subprocess.call([postproc, '--image', os.path.join(imagesDir, preset, line, outputCNN), '--gaussianVariance', sigma, '--preset', preset])
    

    validate(sampleTrainList, sampleTestList, label, outputCNN+'-gaussian'+sigma+'.nrrd', preset, '')


def validate(samplesTrainListK, samplesTestListK, label, outputCNN, preset, suffix):
    with open(samplesTrainListK) as f:
        for line in f:
            line = line.replace('\n','')
            subprocess.call([valid, '--testImage', os.path.join(imagesDir, preset, line, outputCNN), '--label', os.path.join(imagesDir, preset, line, label),
                             '--report', 'report-cnn-train-' + preset + '-' + suffix + '.csv'])
    with open(samplesTestListK) as f:
        for line in f:
            line = line.replace('\n','')
            subprocess.call([valid, '--testImage', os.path.join(imagesDir, preset, line, outputCNN), '--label', os.path.join(imagesDir, preset, line, label),
                             '--report', 'report-cnn-test-' + preset + '-'+ suffix + '.csv'])
    
if __name__ == "__main__":
    main()
