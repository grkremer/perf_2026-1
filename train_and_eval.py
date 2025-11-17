from train_and_eval_functions import run_experiment
import pandas as pd
import numpy as np
import pickle
import itertools

def shuffled_tests(models, datasets, times = 5, shuffle = True):
    models_and_datasets = [
        {"model": model, "dataset": dataset} for model, dataset in list(itertools.product(models, datasets)) * times
    ]
    if shuffle:
        np.random.shuffle(models_and_datasets)
    return models_and_datasets

if __name__ == "__main__":
    model_names = ["ConvE", "ComplEx","ConvKB","DistMult","TransE","RotatE"]

    dataset_names = ["Nations", "DBpedia50", "FB15k-237", "WN18RR", "YAGO3-10"]
    final_df = None
    output_dir = 'results/'
    version = 'v16'

    tests = shuffled_tests(model_names, dataset_names, times = 1, shuffle=False)

    for test in tests:
        print(f"Running {test['model']} on {test['dataset']}")
        df = run_experiment(model_name=test['model'], dataset_name=test['dataset'], batch_size=128, test_batch_size=128, 
                            epochs=50, device="cuda", seed=1, n_tests = 1, inference_batch_size=1, slice_size=32)
        final_df = pd.concat([final_df, df], ignore_index=True)
        final_df.to_csv(output_dir + 'results_' + version + '.csv', index=False)
    
