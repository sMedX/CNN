import os

def makeSampleNames(pathTiles, label):
    print 'pathTiles:', pathTiles
    with open(('tileList.txt'), 'w') as f:
        for file in os.listdir(pathTiles):
            print >>f, os.path.join(pathTiles,file) + ' ' + label

if __name__ == "__main__":
    makeSampleNames('D:/alex/tiles/liver/64x64/sampling-1.5/001/0', '0')