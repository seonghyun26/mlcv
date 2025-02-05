cd ../model

model_list=(deeplda deeptda autoencoder deeptica timelagged-autoencoder vde)

for i in "${!model_list[@]}"; do
    cd ./"${model_list[$i]}"
    pwd
    jit_file=$(find . -name "*.pt")
    jit_file=${jit_file:2}
    echo $jit_file
    if [ -n "$jit_f ile" ]; then
        target_dir="../../script/zip/${model_list[i]}/"
        mkdir -p $target_dir
        cp $jit_file $target_dir
    fi
    cd ..
done