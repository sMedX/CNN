import os
import subprocess
from subprocess import call
import errno

classCount = '2'
cut = 'D:\\alex\\CNN-build\\utils\\Release\\CutImageByTiles.exe'
makeTileLists = ['python.exe', 'make_sample_names_' + classCount + '-classes.py']
clas = 'D:\\alex\\nets_pancreas\\caffe-win-1029\\bin\\class_new.exe'
postproc = 'D:\\alex\\CNN-build\\postprocessing\\Release\\postprocessing.exe'
valid = 'D:\\alex\\CNN-build\\validation\\Release\\Validation.exe'

preset = 'liver'
ver = '2_'
dir=os.path.join('C:/caffe', preset, ver)
print dir

def main():

    spacing = '1.5'
    #
    r = '32'
    size = int(r) * 2
    label = preset + '.nrrd'
    mask = preset+'.nrrd-boundingRec-r5.nrrd'
    patient = 'patient.nrrd'
      

    tilesFolder = os.path.join('D:\\alex\\tiles', preset, str(size) + 'x' + str(size), 'sampling-' + spacing)
    samplesList = os.path.join('D:\\alex\\images', preset, 'samplesList.txt')

    print preset,',v-', ver, ',spacing-', spacing
    
    #retcode = subprocess.call([cut, '-listFile', samplesList, '-imageName', patient, '-labelName1', label,
    #                 '-labelName2', 'livertumors.nrrd', '-maskName', mask, '-radius', str(r), '-preset', preset,
    #                 '-stride 0 0 0', '-spacingXY', spacing, spacing, '-strideNegative 1', '-folder', tilesFolder])
    
    #if (retcode != 0):
    #    print 'error. ', cut, ' exit with ', retcode
    #    return
  
    snapshotFolder = os.path.join('D:\\Artem\\caffe\\snap', preset, ver)
    # create folder for snapshots
    try:
        os.makedirs(snapshotFolder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e
        pass
    deviceId = '1'
    iters = '80000'

    n = 4  # number of parts for cross-validation

    for k in range(0, n):
        kn = str(k) + '-' + str(n)
        sampleTrainListK = os.path.join(dir, 'tileList-train-cv-' + kn + '.txt')
        sampleTestListK = os.path.join(dir, 'tileList-test-cv-' + kn + '.txt')

        tileTrainListK = os.path.join(dir, 'train-cv-' + kn + '.txt')
        tileTestListK = os.path.join(dir, 'test-cv-' + kn + '.txt')

        solver = createNetAndSolver(kn, tileTestListK, tileTrainListK,iters, snapshotFolder)

        if (k!=0):
            subprocess.call([makeTileLists[0], makeTileLists[1], tilesFolder, dir, str(k), str(n)])

        train = 'C:\\caffe\\bin\\caffe_cc35.exe'
        subprocess.call([train, 'train', '--solver', solver, '--gpu', deviceId])

        deploy = os.path.join('C:\\caffe', preset, ver, 'deploy.prototxt')
        model = os.path.join('D:/artem/caffe/snap', preset, ver, '_iter_' + iters + '.caffemodel')
        start = [0, 0, 0]
        size = [512, 512, 1000]
        batchLength = '1024'
        groupX = '3'
        groupY = '3'
        suffix = '-v' + ver + '-g' + groupX + 'x' + groupY + '-c' + classCount + '-s' + spacing + '-cv' + kn + '.nrrd'
        outputCNN = preset + '-cnn-' + suffix

        with open(samplesList) as f:
            for line in f:
                line=line[:-1]
                args=[clas, deploy, model, str(start[0]), str(start[1]), str(start[2]), str(size[0]), str(size[1]), str(size[2]), r,
                preset, '0' if spacing=='orig' else spacing, batchLength, groupX, groupY, classCount, os.path.join(line, patient),
                os.path.join(line, mask), os.path.join(line, outputCNN), deviceId]
                print args
                if (k!=0):
                    subprocess.call(args)

        validate(sampleTrainListK, sampleTestListK, label, outputCNN, suffix)          

        sigma = '4.000000'

        with open(samplesList) as f:
            for line in f:
                subprocess.call([postproc, '-image', os.path.join(line, outputCNN), '-gaussianVariance', sigma])

        suffix = suffix + '-gaussian' + sigma + '.nrrd'
        outputCNN = preset + '-cnn-' + suffix

        validate(label, outputCNN, suffix)
    return


def validate(samplesTrainListK, samplesTestListK, label, outputCNN, suffix):
    with open(samplesTrainListK) as f:
        for line in f:
            subprocess.call([valid, '-testImage', os.path.join(line, outputCNN), '-label', os.path.join(line, label),
                             '-report report-cnn-train-' + suffix + '.csv'])
    with open(samplesTestListK) as f:
        for line in f:
            subprocess.call([valid, '-testImage', os.path.join(line, outputCNN), '-label', os.path.join(line, label),
                             '-report report-cnn-test-' + suffix + '.csv'])
    return valid


def createNetAndSolver(kn, tileTestListK, tileTrainListK, iters, snapshotFolder):
    netTemplate = os.path.join(dir, 'netTemplate.prototxt')
    net = os.path.join(dir, 'net-cv-' + kn + '.prototxt')
    fileData = None
    with open(netTemplate, 'r') as file:
        fileData = file.read()
    fileData = fileData \
        .replace('%trainTilesList%', tileTrainListK.replace('\\','/')) \
        .replace('%testTilesList%', tileTestListK.replace('\\','/'))
    with open(net, 'w') as file:
        file.write(fileData)
        
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
    return solver

if __name__ == "__main__":
    main()