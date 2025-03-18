cd ../

for k in 100 200 300 400 500 600 700;
do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name tbgcv \
        ++data.version=timelag-250n-v1 \
        ++steeredmd.simulation.k=$k
    sleep 1
done