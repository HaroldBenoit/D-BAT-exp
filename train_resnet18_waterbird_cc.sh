for epochs in 30 
do
for alpha in 0.1
do
    for seed in 0 1 2 

    do 
    python ./src/main.py --config ./configs/waterbirds_cc.yml --epochs $epochs --no_wandb --model resnet18 --seed $seed --alpha $alpha --perturb_type ood_is_test  --results_base_folder varying_pretraining 
    done
done
done