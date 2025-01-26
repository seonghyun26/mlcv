cd ../

echo "Timelag 1n"
CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name vde \
    ++data.version=timelag-1n-v1 &

sleep 1

echo "Timelag 10n"
CUDA_VISIBLE_DEVICES=$2 python main.py \
    --config-name vde \
    ++data.version=timelag-10n-v1 &

sleep 1

echo "Timelag 250n"
CUDA_VISIBLE_DEVICES=$3 python main.py \
    --config-name vde \
    ++data.version=timelag-250n-v1 &
