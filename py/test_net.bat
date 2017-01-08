set "caffe=D:/alex/ms-caffe-rep-copy1/Build/x64/Release/caffe.exe"

set "nets=D:/alex/caffe-nets"
set "snaps=D:/Artem/caffe/snap"

set "dir=C:/Users/alkor/Downloads/20161014-130117-db42_epoch_0.1.tar"

%caffe% test -model %dir%/test.prototxt -weights C:/Users/alkor/Downloads/20161016-200508-e138_epoch_0.4.tar/snapshot_iter_11848.caffemodel -gpu all

pause