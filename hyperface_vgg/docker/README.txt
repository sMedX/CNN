1. Создание docker-образа

    Из родительской директории (hyperface_vgg) выполните команду:
    nvidia-docker build -t emotions -f docker/Dockerfile .

2. Подготовка данных

    Директория с данными должна иметь следующую структуру
    .
    ├── images
    │   ├── 300W
    │   │   ├── 01_Indoor
    │   │   ├── 02_Outdoor
    │   │   └── 300W
    │   └── MENPO
    │       ├── test
    │       │   ├── profile
    │       │   └── semifrontal
    │       └── train
    ├── models
    └── result

    Директория ./models содержит файлы модели model_epoch_115 и snapshot_epoch_115
    Директория ./result пустая
    Директории ./images/300W/01_Indoor, ./images/300W/02_Outdoor, ./images/MENPO/test/profile, 
        ./images/MENPO/test/semifrontal, ./images/MENPO/train содержит обучающую информацию

    Директория с данными также содержит файл config.json следующего содержания:

    {
        "dataset": "menpo",
        "aflw_sqlite_path": "/large/datasets/aflw/aflw/data/aflw.sqlite",
        "aflw_imgdir_path": "/large/datasets/aflw/aflw/data/flickr",
        "aflw_cache_path": "./aflw_cache.npz",
        "aflw_test_rate": 0.041010499,
        "menpo_train_paths": [
            "./data/images/300W/01_Indoor",
            "./data/images/300W/02_Outdoor",
            "./data/images/MENPO/train"
        ],
        "menpo_test_rate": 0.05,
        "menpo_cache_path": "./data/menpo_cache.npz",
        "vgg_caffemodel_path": "/large/datasets/VGG/VGG_ILSVRC_16_layers.caffemodel",
        "gpu": 0,
        "n_loaders_train": 7,
        "n_loaders_test": 1,
        "n_epoch": 200,
        "batchsize": 10,
        "port_face": 8888,
        "port_weight": 8889,
        "port_lossgraph": 8890,
        "port_evaluate": 8891,
        "port_demo": 9000,
        "outdir": "./data/result",
        "outdir_pretrain": "result_pretrain"
    }
    
    Директория с данными подготовлена на сервере:
    /home/egorov/Projects/Emotions/DockerData

3. Создание и запуск контейнера

    Из директории с данными выполните команду:
    nvidia-docker run -d \
        --name=emotions \
        -p 8888:8888 \
        -p 8890:8890 \
        --mount type=bind,source="$(pwd)",target=/app/data \
        emotions
            
4. После запуска прогресс обучения можно наблюдать на

    http://localhost:8888/
    http://localhost:8890/

    Если контейнер запускается на удаленной машине, до нее необходимо настроить ssh-туннели:
    ssh -L 8888:localhost:8888 -L 8890:localhost:8890 46.0.198.105
    где 46.0.198.105 - адрес сервера
