cd ../

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name deeptica \
    ++data.version=timelag-1n-v1 \
    ++steeredmd.repeat=0

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name deeptica \
    ++data.version=timelag-10n-v1 \
    ++steeredmd.repeat=0

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name deeptica \
    ++data.version=timelag-250n-v1 \
    ++steeredmd.repeat=0
