
## DBAT: D_ood = D_test

for heads in 64 
do
    for seed in 0 1 2
    do 
    python ./src/main.py --config ./configs/waterbirds_cc_np.yml --model resnet18 --batch_size_train 256 --batch_size_eval 256 --epochs 30 --ensemble_size $heads --group CC_WaterBirds_np_h$heads --seed $seed --alpha 0.1 --perturb_type ood_is_test  --results_base_folder mult_heads/heads_$heads --no_wandb

    done


done
