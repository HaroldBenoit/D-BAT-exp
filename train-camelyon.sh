

# one of each first

# DBAT: D_ood = D_test
python ./src/main.py --config ./camelyon17_paper.yml --seed 0 --alpha 0.000001 --perturb_type ood_is_test --results_base_folder results_reproduction 
# DBAT: D_ood != D_test
python ./src/main.py --config ./camelyon17_paper.yml --seed 0 --alpha 0.000001 --perturb_type ood_is_not_test --results_base_folder results_reproduction

#ERM
python ./src/main.py --config ./camelyon17_paper.yml --seed 0  --no_diversity --alpha 0.0  --results_base_folder results_reproduction





# DBAT: D_ood = D_test
python ./src/main.py --config ./camelyon17_paper.yml --seed 1 --alpha 0.000001 --perturb_type ood_is_test --results_base_folder results_reproduction
python ./src/main.py --config ./camelyon17_paper.yml --seed 2 --alpha 0.000001 --perturb_type ood_is_test --results_base_folder results_reproduction


# DBAT: D_ood != D_test
python ./src/main.py --config ./camelyon17_paper.yml --seed 1 --alpha 0.000001 --perturb_type ood_is_not_test --results_base_folder results_reproduction
python ./src/main.py --config ./camelyon17_paper.yml --seed 2 --alpha 0.000001 --perturb_type ood_is_not_test --results_base_folder results_reproduction

# ERM
python ./src/main.py --config ./camelyon17_paper.yml --seed 1  --no_diversity --alpha 0.0  --results_base_folder results_reproduction
python ./src/main.py --config ./camelyon17_paper.yml --seed 2  --no_diversity --alpha 0.0  --results_base_folder results_reproduction