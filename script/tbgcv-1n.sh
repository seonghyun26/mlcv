cd ../

# for k in 100 200 300 400;
# do
#     CUDA_VISIBLE_DEVICES=$1 python main.py \
#         --config-name tbgcv \
#         ++data.version=timelag-1n-v1 \
#         ++steeredmd.simulation.k=$k
#     sleep 1
# done

for k in 200 300 400;
do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name tbgcv-both \
        ++data.version=timelag-1n-v1 \
        ++steeredmd.simulation.k=$k
    sleep 1
done