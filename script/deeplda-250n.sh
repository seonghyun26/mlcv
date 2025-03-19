cd ../

# for k in 100 200 300 400;
for k in 500 600 700;
do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name deeplda \
        ++model.checkpoint=True \
        ++model.checkpoint_name=final \
        ++data.version=250n-v1 \
        ++steeredmd.simulation.k=$k
    sleep 1
done