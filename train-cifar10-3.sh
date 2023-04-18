



for alpha in  1
    do 
        for epochs in 25 75 150 200
            do
                for ensemble_size in 2 3 4 5 
                    do 
                        python ./src/main.py --ensemble_size $ensemble_size  --config ./cifar.yml --split_semantic_task_idx 0 --split_spurious_task_idx 4 --epochs $epochs --alpha $alpha --group D-BAT-v2 

                    done

            done

    done 