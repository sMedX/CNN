import os
import os.path
import shutil

exportSource = "../livertumors"
exportTarget = "../livertumors_export"
dirListFile = exportTarget + '/samplesListRelative.txt'

files = ['patient.nrrd', 'livertumors.nrrd', 'liver.nrrd']

os.makedirs(exportTarget)

# create list
with open(dirListFile, 'w') as file:
    for dir in os.listdir(exportSource):
        if os.path.isdir(os.path.join(exportSource, dir)):
            file.write(dir + '\n')

# actually coping
dirList = [line.rstrip('\n') for line in open(dirListFile)]
for dir in dirList:
    dst = os.path.join(exportTarget, dir)
    os.mkdir(dst)
    for file in files:
        src = os.path.join(exportSource, dir, file)
        shutil.copy(src, dst)
        print(src)
print('done')
