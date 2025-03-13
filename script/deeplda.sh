cd ../

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name deeplda \
    ++data.version=1n-v1

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name deeplda \
    ++data.version=10n-v1

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name deeplda \
    ++data.version=250n-v1
