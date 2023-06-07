
## DBAT: D_ood = D_test

for heads in 3 4 
do
    for seed in 0 1 2
    do 
    python ./src/main.py --config ./configs/waterbirds_cc_np.yml --epochs 300 --ensemble_size $heads --group CC_WaterBirds_np_h$heads --seed $seed --alpha 0.1 --perturb_type ood_is_test  --results_base_folder heads_$heads 

    done


done

#    python ./src/main.py --config ./configs/waterbirds_cc_np.yml --epochs 300 --ensemble_size 3 --group CC_WaterBirds_np_h3 --seed 0 --alpha 0.1 --perturb_type ood_is_test  --results_base_folder  alpha=0.1_train/grey/h3 --no_wandb --resume


    python ./src/main.py --config ./configs/waterbirds_cc_np.yml --epochs 100 --ensemble_size 4 --group CC_WaterBirds_np_h4 --seed 2 --alpha 0.1 --perturb_type ood_is_test  --results_base_folder heads_4 --no_wandb
