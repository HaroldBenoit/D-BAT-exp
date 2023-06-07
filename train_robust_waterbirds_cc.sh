
## DBAT: D_ood = D_test

for seed in 0 1 2 
    do

    python ./src/main.py --config ./configs/waterbirds_cc.yml --epochs 300 --batch_size_train 64 --batch_size_eval 256 --no_wandb --model robust_resnet50 --seed $seed --alpha 0.1 --perturb_type ood_is_test  --results_base_folder varying_pretraining 

    done
