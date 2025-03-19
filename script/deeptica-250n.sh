cd ../

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name deeptica \
    ++data.version=itimelag-250n-v1 \
    ++steeredmd.simulation.k=$k


for k in 200 400 600 800 1000;
do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name deeptica \
        ++model.checkpoint=True \
        ++model.checkpoint_name=final \
        ++data.version=timelag-250n-v1 \
        ++steeredmd.simulation.k=$k
    sleep 1
done