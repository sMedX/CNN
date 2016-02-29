import os
from os import path, rename

preset='pancreas'###
dir='D:\\alex\\images'
with open(os.path.join(dir,preset,'samplesList.txt')) as f:
    for line in f:
        path = line[:-1]
        print path
        names=['Venous_phase.nrrd', 'Arterial_phase.nrrd']
        new=os.path.join(path,'patient.nrrd')
        for name in names:
            old=os.path.join(path,name)
            print old,'->',new
            if (os.path.isfile(old)):
                try:
                    os.rename(old, new)
                    break
                except Exception as e:
                    print e
                    pass
                 