# cd ../

# for k in 200 300 400;
# do
#     CUDA_VISIBLE_DEVICES=$1 python main.py \
#         --config-name tbgcv-both \
#         ++data.version=timelag-10n-v1 \
#         ++steeredmd.simulation.k=$k
#     sleep 1
# done