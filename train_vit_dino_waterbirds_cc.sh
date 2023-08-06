
# D-BAT
for epochs in 30 100
do
for alpha in 0.1
do
for seed in 0 1 2
do
python ./src/main.py  --config ./configs/waterbirds_cc.yml --model vit_dino --batch_size_train 64 --batch_size_eval 64 --seed $seed --epochs $epochs  --alpha $alpha --perturb_type ood_is_test  --results_base_folder varying_pretraining --no_wandb

done
done
done


