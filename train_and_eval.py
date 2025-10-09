from train_and_eval_functions import run_experiment
import pandas as pd
import pickle

if __name__ == "__main__":
    model_names = ["ConvE","ComplEx","ConvKB","DistMult","TransE","RotatE","R-GCN"]

    dataset_names = [ "Nations"]
    final_df = None
    #models = []
    output_dir = 'results/'
    version = 'v07'
    for dataset in dataset_names:
        for model_name in model_names:
            print(f"Running {model_name} on {dataset}")
            df = run_experiment(model_name=model_name, dataset_name=dataset, batch_size=2048, test_batch_size=32, epochs=10, device="cuda", seed=1, n_tests = 1, inference_batch_size=1)
            final_df = pd.concat([final_df, df], ignore_index=True)
            #with open(output_dir + 'model_'+model_name+'_'+dataset+'.pkl', 'wb') as f:
            #    pickle.dump(model, f)
            #del model
            final_df.to_csv(output_dir + 'results_' + version + '.csv', index=False)