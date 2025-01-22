cd ../

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name timelagged-autoencoder \
    ++data.version=timelag-1n-v1

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name timelagged-autoencoder \
    ++data.version=timelag-10n-v1

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name timelagged-autoencoder \
    ++data.version=timelag-250n-v1
