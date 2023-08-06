
# D-BAT
for epochs in 30
do
for alpha in 0.00001
do
for seed in 0 1 2
do
python ../src/main.py  --config ../configs/office_home.yml --model vit_b_16 --batch_size_train 48 --batch_size_eval 48 --seed $seed --epochs $epochs  --alpha $alpha --perturb_type ood_is_test  --results_base_folder ../office_home --no_wandb

done
done
done

