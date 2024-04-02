files_path='/gscratch/simondu/ypwang61/files' # Path to the files (input/output)
dataset_path="/gscratch/simondu/datasets" # Path to the datasets. Metadata should be in this folder



########################### DataComp-Small ###########################
datasets_scale='datacomp_small' # Dictionary datacomp_small or datacomp_medium

fraction=0.45 # fraction for CLIP score filtering
fraction_vas=0.3 # fraction for VAS filtering

######### VAS(ImageNet-1k) #########
# {VAS(ImageNet-1k) 20% fraction} intersect {CLIP score (L/14) 30% fraction}
target_variance_name='imagenet-1k' # target variance name for VAS(target proxy)

python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_f${fraction}_fvas${fraction_vas}_final.npy \
        --files_path ${files_path} --name vas --arch l14 --fraction ${fraction} --fraction_vas ${fraction_vas} --target_variance_name ${target_variance_name}


######### VAS-D(DataComp-Small) #########
# {VAS-D(DataComp-Small) 20% fraction} intersect {CLIP score (L/14) 30% fraction}
num_iters=168 # number of iterations for VAS-D
batch_size=500000 # batch size for calculating target variance in VAS-D
batch_size_vass=200000 # batch size for calculating VAS score in VAS-D

python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_d_f${fraction}_fvas${fraction_vas}_${num_iters}_final.npy \
        --files_path ${files_path} --name vas_d --arch l14 --fraction ${fraction} --fraction_vas ${fraction_vas} --num_iters ${num_iters} --batch_size ${batch_size} --batch_size_vass ${batch_size_vass}
   