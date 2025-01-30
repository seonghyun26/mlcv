cd ../

# Time independent models
dataset_list=(da-1n-v1 da-10n-v1 da-250n-v1)
model_list=(deeplda deeptda autoencoder)

for i in "${!dataset_list[@]}"; do
    for j in "${!model_list[@]}"; do
        CUDA_VISIBLE_DEVICES=$1 python main.py \
            --config-name ${model_list[$j]}  \
            ++data.version=${dataset_list[$i]} 
        sleep 1
    done
done