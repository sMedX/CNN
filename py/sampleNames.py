import os

#pathTiles = 'D:/alex/tiles-usi/64x64'
pathTiles = '/root/host/tles-usi/224x224'

with open(os.path.join(pathTiles, 'tileNamesPartCV1-2.txt'), 'w') as sampleTestf:
        with open(os.path.join(pathTiles, 'tileNamesPartCV2-2.txt'), 'w') as sampleTrainf:
            for label in os.listdir(pathTiles):
                print label
                pathTilesForLabel = os.path.join(pathTiles, label)
                if not os.path.isdir(pathTilesForLabel):
                    continue

                for file in os.listdir(pathTilesForLabel):
                    num = int(file.split('_')[0][1:]) #extract num '37' from filename n37_70-226.png
                    string = os.path.join(pathTilesForLabel, file) + ' ' + label
                    if num < 17:
                        print >>sampleTrainf, string
                    else:
                        print >>sampleTestf, string



