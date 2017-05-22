

Модель тренируется в darknet, это экспериментальная среда для глубокого обучения,
написана на C. Код местами выглядит страшновато, но он вполне рабочий.

Далее надо будет сконвертировать модель в формат TensorFlow, это можно сделать с
помощью DarkFlow (реализация DarkNet в TensorFlow).

После всего этого мы получаем граф для TensorFlow, который мы вставляем в демо
для Android из TensorFlow, и запускаем полученный процесс.


*** Как натренировать модель с помощью darknet

1. Забираем darknet

$ git clone https://github.com/pjreddie/darknet


2. Редактируем Makefile чтобы включить CUDA и OpenCV:

$ cd darknet

$ git diff Makefile
diff --git a/Makefile b/Makefile
index 029d02c..8e70093 100644
--- a/Makefile
+++ b/Makefile
@@ -1,6 +1,6 @@
-GPU=0
-CUDNN=0
-OPENCV=0
+GPU=1
+CUDNN=1
+OPENCV=1
 DEBUG=0


3. Собираем darknet

$ make

Посе сборки должен получиться исполняемый файл ․/darknet. Код экспериментальный, поэтому
могут быть небольшие проблемы при сборке.


4. Создаем файл с настройками набора данных

$ cat ./cfg/retail.data
classes = 1
train = /large/datasets/retail_yolo/train.txt
valid = /large/datasets/retail_yolo/valid.txt
names = /large/datasets/retail_yolo/names.txt
backup = backup_voc/


5. Генерируем метки для darknet.

Для каждого изображения X.jpg нужно создать файл X.txt следующего формата։
<x> <y> <width> <height> <class>
<x> <y> <width> <height> <class>
<x> <y> <width> <height> <class>
...

Позиция и размеры должны быть относительно изображения, пример такого файла։
0.48643410852713176 0.4547803617571059 0.16279069767441862 0.7364341085271318 0
0.6526162790697674 0.4754521963824289 0.17732558139534882 0.7416020671834626 0
0.8231589147286822 0.4683462532299742 0.1928294573643411 0.7583979328165374 0
0.9554263565891472 0.45542635658914726 0.08914728682170547 0.7454780361757106 0

Скрпит которым я генерировал такие описания называется make_yolo_annotations.py


6. Забираем претренированные веса։

$ wget https://pjreddie.com/media/files/tiny-yolo-voc.weights


7. Запускаем обучение։

$ ./darknet detector train cfg/retail.data cfg/tiny-yolo-voc.cfg tiny-yolo-voc.weights -clear

По умолчанию он сделает 50000 циклов обучения, заняло 10-12 часов на моей машине.



*** Как сконвертировать модель в формат TensorFlow

1. Забираем darkflow

$ git clone https://github.com/thtrieu/darkflow.git


2. Устанавливаем darkflow

$ cd darkflow

$ pip3 install .


3. Конвертируем модель

$ flow --model cfg/tiny-yolo-voc.cfg --load ~/darknet/backup_voc/tiny-yolo-voc_final.weights --savepb --verbalise=True


Конвертер может начать ругаться на то, что количество строк в файле labels.txt не соответствует
количеству классов, тогда просто генерируем файл с названиями классов, он ни на что не влияет

$ seq 1 20 >> labels.txt


4. В результате должен получиться файл c весами порядка 60 Мб։

$ ls -la graph-tiny-yolo-voc.pb
-rw-rw-r-- 1 mel mel 63481148 май 22 09:18 graph-tiny-yolo-voc.pb



*** Как собрать демо TensorFlow для Android с нашей моделью

1. Клонируем TensorFlow

$ git clone --recurse-submodules https://github.com/tensorflow/tensorflow.git


2. Скачиваем и устанавливаем bazel, систему сборки для TensorFlow с
https://github.com/bazelbuild/bazel/releases. Вообще есть пакет для Ubuntu, но он у меня
не заработал.


3. Скачиваем Anroid SDK (можно вместе с Android Studio) и Android NDK, прописываем
пути к ним в файле WORKSPACE:

$ git diff WORKSPACE 
diff --git a/WORKSPACE b/WORKSPACE
index b2d6fb5..92e020b 100644
--- a/WORKSPACE
+++ b/WORKSPACE
@@ -17,23 +17,23 @@ closure_repositories()
 load("//tensorflow:workspace.bzl", "tf_workspace")
 
 # Uncomment and update the paths in these entries to build the Android demo.
-#android_sdk_repository(
-#    name = "androidsdk",
-#    api_level = 23,
-#    # Ensure that you have the build_tools_version below installed in the 
-#    # SDK manager as it updates periodically.
-#    build_tools_version = "25.0.2",
-#    # Replace with path to Android SDK on your system
-#    path = "<PATH_TO_SDK>",
-#)
-#
+android_sdk_repository(
+   name = "androidsdk",
+   api_level = 23,
+   # Ensure that you have the build_tools_version below installed in the 
+   # SDK manager as it updates periodically.
+   build_tools_version = "25.0.2",
+   # Replace with path to Android SDK on your system
+   path = "/large/build/Android/",
+)
+
 # Android NDK r12b is recommended (higher may cause issues with Bazel)
-#android_ndk_repository(
-#    name="androidndk",
-#    path="<PATH_TO_NDK>",
-#    # This needs to be 14 or higher to compile TensorFlow. 
-#    # Note that the NDK version is not the API level.
-#    api_level=14)
+android_ndk_repository(
+   name="androidndk",
+   path="/large/build/android-ndk-r12b/",
+   # This needs to be 14 or higher to compile TensorFlow. 
+   # Note that the NDK version is not the API level.
+   api_level=14)
 
 # Please add all new TensorFlow dependencies in workspace.bzl.
 tf_workspace()


4. Собираем код на C++ для демо։

$ bazel build -c opt //tensorflow/examples/android:tensorflow_demo


5. Кладем файл graph-tiny-yolo-voc.pb в tensorflow/examples/android/assets


6. Дальше открываем проект tensorflow/examples/android в Android Studio,
прописываем путь к нашему файлу и указываем что надо использовать YOLO.

$ git diff tensorflow/examples/android/src/org/tensorflow/demo/DetectorActivity.java 
diff --git a/tensorflow/examples/android/src/org/tensorflow/demo/DetectorActivity.java b/tensorflow/examples/android/src/org/tensorflow/demo/DetectorActivity.java
index 5800f80..2ed454e 100644
--- a/tensorflow/examples/android/src/org/tensorflow/demo/DetectorActivity.java
+++ b/tensorflow/examples/android/src/org/tensorflow/demo/DetectorActivity.java
@@ -74,7 +74,7 @@ public class DetectorActivity extends CameraActivity implements OnImageAvailable
   private static final int YOLO_BLOCK_SIZE = 32;
 
   // Default to the included multibox model.
-  private static final boolean USE_YOLO = false;
+  private static final boolean USE_YOLO = true;
 
   private static final int CROP_SIZE = USE_YOLO ? YOLO_INPUT_SIZE : MB_INPUT_SIZE;


7. Запускаем проект на телефоне с помощью USB debugging (в эмуляторе работать не будет). Как включить
отладку на телефоне и настроить ее для своей ОС лучше погуглить.


8. Обрабатываем напильником и радуемся жизни.
