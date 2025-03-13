cd ../

for k in 100 200 300 400;
do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name deeplda \
        ++data.version=10n-v1 \
        ++steered.simulation.k=$k
    sleep 1
done