cd ../

for k in 100 200 300 400;
do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name deeptda \
        ++data.version=10n-v1 \
        ++steeredmd.simulation.k=$k
    sleep 1
done