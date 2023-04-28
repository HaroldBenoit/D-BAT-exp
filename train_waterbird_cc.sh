
## DBAT: D_ood = D_test
python ./src/main.py --config ./configs/waterbirds_cc.yml --seed 0 --alpha 0.0001 --perturb_type ood_is_test  --results_base_folder results_reproduction 
python ./src/main.py --config ./configs/waterbirds_cc.yml --seed 1 --alpha 0.0001 --perturb_type ood_is_test  --results_base_folder results_reproduction
python ./src/main.py --config ./configs/waterbirds_cc.yml --seed 2 --alpha 0.0001 --perturb_type ood_is_test  --results_base_folder results_reproduction

## ERM
python ./src/main.py --config ./configs/waterbirds_cc.yml --seed 0 --alpha 0  --results_base_folder results_reproduction --no_diversity
python ./src/main.py --config ./configs/waterbirds_cc.yml --seed 1 --alpha 0  --results_base_folder results_reproduction --no_diversity
python ./src/main.py --config ./configs/waterbirds_cc.yml --seed 2 --alpha 0  --results_base_folder results_reproduction --no_diversity


