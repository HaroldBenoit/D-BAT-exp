
for epochs in 300
    do
for seed in 0 1 2 
    do 
    python src/main.py --config configs/waterbirds_cc_np.yml --seed $seed --alpha 0.1 --inverse_correlation --perturb_type ood_is_test --epochs $epochs --root_dir  datasets --results_base_folder inverse_correlated --no_wandb --resume
    python src/main.py --config configs/waterbirds_cc.yml --seed $seed --alpha 0.1 --inverse_correlation --perturb_type ood_is_test --epochs $epochs --root_dir  datasets --results_base_folder inverse_correlated --no_wandb --resume

    done
done


