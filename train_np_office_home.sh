
# D-BAT
for epochs in 50 100
do
for alpha in 0.00001 0.1
do
for seed in 0 1 2
do
python ./src/main.py  --config ./configs/office_home_np.yml --seed $seed --epochs $epochs  --alpha $alpha --perturb_type ood_is_test  --results_base_folder office_home --no_wandb

done
done
done
