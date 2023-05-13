
## DBAT: D_ood = D_test

for heads in 3 4 
do
    for seed in 0 1 2
    do 
    python ./src/main.py --config ./configs/waterbirds_cc_np.yml --epochs 300 --ensemble_size $heads --group CC_WaterBirds_np_h$heads --seed $seed --alpha 0.1 --perturb_type ood_is_test  --results_base_folder heads_$heads 

    done


done
