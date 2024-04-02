################################# Start of Arguments #################################
num_gpus=8 # number of gpus
files_path='/homes/gws/ypwang61/nobackup/files' # Path to the files
is_overwrite=0 # 0 -> NO overwrite the existing shards, 1-> overwrite the existing shards

####### DataComp-medium ######
scale="medium" # small or medium
datasets_scale='datacomp_medium' # datacomp_small or datacomp_medium

# ####### DataComp-small ######
# scale="small" 
# datasets_scale='datacomp_small' 


# define a list of the names of filters, for filter in a given list, echo the name of the filter
filter_list=(
    # example for baseline: 
    # no_filter
    # clip_score_l14_30_percent
    # image_based_intersect_clip_score_l14_30_percent

    # example for VAS/VAS-D on datacomp_medium:
    # vas_f0.3_fvas0.2_final
    # vas_d_f0.3_fvas0.2_500_final

    # example for VAS/VAS-D on datacomp_small:
    # vas_f0.45_fvas0.3_final
    # vas_d_f0.45_fvas0.3_168_final
)

# list of seeds
seed_list=( 0 )

################################# End of Arguments #################################





########## reshard the training data ##########
for filter in "${filter_list[@]}"
do      
        if [ $filter == 'no_filter' ]
        then
            continue
        else
            # sharder
            echo "resharder begin for ${filter}"
            mkdir /local1/datasets/${datasets_scale}/${filter}
            if [ $is_overwrite -eq 1 ]
            then
                python resharder.py -i /local1/datasets/${datasets_scale}/shards -o /local1/datasets/${datasets_scale}/${filter} -s ${files_path}/${datasets_scale}/uids/${filter}.npy --overwrite
            else
                python resharder.py -i /local1/datasets/${datasets_scale}/shards -o /local1/datasets/${datasets_scale}/${filter} -s ${files_path}/${datasets_scale}/uids/${filter}.npy
            fi

            echo "resharder done for ${filter}"
        fi
done

########## train ##########
for seed in "${seed_list[@]}"
do
    for filter in "${filter_list[@]}" 
    do  
        exp_name="${filter}_${scale}_seed_${seed}"

        if [ $filter == 'no_filter' ]
        then
            data_dir="/local1/datasets/${datasets_scale}/shards"
        else
            data_dir="/local1/datasets/${datasets_scale}/${filter}"
        fi
        # run 
        torchrun --rdzv_backend c10d --rdzv_endpoint localhost:29499 --nproc_per_node $num_gpus \
                train.py --scale $scale --data_dir $data_dir --output_dir ${files_path}/${datasets_scale}/output/ --exp_name ${exp_name} 

        echo "training done for ${exp_name}, data_dir = ${data_dir}"
    done
done

########## evaluate on 38 evaluation sets ##########
for seed in "${seed_list[@]}"
do
    for filter in "${filter_list[@]}"
    do
        exp_name="${filter}_${scale}_seed_${seed}"

        if [ $filter == 'no_filter' ]
        then
            data_dir="/local1/datasets/${datasets_scale}/shards"
        else
            data_dir="/local1/datasets/${datasets_scale}/${filter}"
        fi
        
        python evaluate.py  --train_output_dir ${files_path}/${datasets_scale}/output/${exp_name}/ --data_dir /local1/datasets/datacomp_eval/
        echo "evaluation done for ${exp_name}, data_dir = ${data_dir}"
    done
done
