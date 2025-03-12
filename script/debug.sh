cd ../

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name debug \
    hydra.run.dir=outputs/_debug