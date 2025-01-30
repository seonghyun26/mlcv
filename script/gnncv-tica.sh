cd ../

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name gnncv-tica \
    ++data.version=graph-1n-v1

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name gnncv-tica \
    ++data.version=graph-10n-v1

# CUDA_VISIBLE_DEVICES=$1 python main.py \
#     --config-name gnncv-tica \
#     ++data.version=graph-250n-v1
