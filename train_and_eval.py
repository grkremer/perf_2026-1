from functions import run_experiment
import pandas as pd
import numpy as np
import pickle, itertools, time, sys

def shuffled_tests(models, datasets, times = 5, shuffle = True):
    models_and_datasets = [
        {"model": model, "dataset": dataset} for model, dataset in list(itertools.product(models, datasets)) * times
    ]
    if shuffle:
        np.random.shuffle(models_and_datasets)
    return models_and_datasets

#recebe parametro como nome do arquivo de output

if __name__== "__main__":
    output_dir = 'results/' + str(sys.argv[1]) + '.csv' if len(sys.argv) > 1 else 'results/'+str(time.time())+'.csv'
    model_names = ["ConvE", "ComplEx","ConvKB","DistMult","TransE","RotatE"]

    dataset_names = ["Nations", "DBpedia50", "FB15k-237", "WN18RR", "YAGO3-10"]
    final_df = None

    tests = shuffled_tests(model_names, dataset_names, times = 2, shuffle=True)

    for test in tests:
        print(f"Running {test['model']} on {test['dataset']}")
        df = run_experiment(model_name=test['model'], dataset_name=test['dataset'], batch_size=128, test_batch_size=128, 
                            epochs=50, device="cuda", seed=1, n_tests = 1, inference_batch_size=1, slice_size=32)
        final_df = pd.concat([final_df, df], ignore_index=True)
        final_df.to_csv(output_dir, index=False)
    
    
