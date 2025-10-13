import os
import time
import pickle
from tqdm.notebook import tqdm
import pandas as pd
import logging
import gc

import torch
import numpy as np

import threading
import pynvml

from pykeen.pipeline import pipeline
from pykeen.datasets import get_dataset, dataset_resolver

#logging.getLogger("pykeen").setLevel(logging.ERROR)

def gpu_monitor(stop_event, interval=1.0, device_index=0, stats=None):
    """Thread para monitorar consumo de GPU durante o treino."""
    
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    while not stop_event.is_set():
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # W
        stats["mem_used"].append(mem.used / 1024**2)  # MB
        stats["util"].append(util.gpu)  # %
        stats["power"].append(power)  # W
        time.sleep(interval)

def run_experiment(model_name: str, dataset_name: str, epochs: int = 100,
                   batch_size: int = 256, test_batch_size = 256, device: str = "cuda",
                   inference_batch_size: int = 1, seed: int = 42,
                   n_tests: int = 1, verbose: bool = True, slice_size = None,
                   gpu_index: int = 0, monitor_interval: float = 0.1) -> pd.DataFrame:

    metrics = []
    models = []
    pynvml.nvmlInit()
    for i in range(n_tests):
        stats = {"mem_used": [], "util": [], "power": []}
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=gpu_monitor, args=(stop_event, monitor_interval, gpu_index, stats)
        )
        seed = seed + 1
        np.random.seed(seed)

        # --- iniciar monitoramento GPU
        monitor_thread.start()

        dataset = get_dataset(dataset=dataset_name)

        # --- Treino + avaliação do PyKEEN
        result = pipeline(
            model=model_name,
            dataset=dataset_name,
            epochs=epochs,
            device=device,
            random_seed=seed, 
            training_kwargs=dict(batch_size=batch_size, use_tqdm_batch=False,), # sampler="schlichtkrull" ISSO AQUI FAZ O NEGOCIO RODAR 10 VEZES MAIS LENTO!!!
            negative_sampler = "basic",
            negative_sampler_kwargs=dict(
            filtered=True),
            use_tqdm=verbose,
            evaluation_kwargs=dict(slice_size = slice_size)
        )

        # --- parar monitoramento
        stop_event.set()
        monitor_thread.join()

        # --- estatísticas GPU
        avg_mem = sum(stats["mem_used"]) / len(stats["mem_used"]) if stats["mem_used"] else None
        peak_mem = max(stats["mem_used"]) if stats["mem_used"] else None
        avg_util = sum(stats["util"]) / len(stats["util"]) if stats["util"] else None
        peak_util = max(stats["util"]) if stats["util"] else None
        avg_power = sum(stats["power"]) / len(stats["power"]) if stats["power"] else None
        peak_power = max(stats["power"]) if stats["power"] else None
        total_energy_wh = sum(p * monitor_interval for p in stats["power"]) / 3600 if stats["power"] else None

        # extrair tempos do pipeline
        train_time = getattr(result, "train_seconds", None)
        eval_time = getattr(result, "evaluate_seconds", None)

        # métricas do teste
        mrr = result.metric_results.get_metric('both.realistic.inverse_harmonic_mean_rank')
        hits1 = result.metric_results.get_metric('both.realistic.hits_at_1')
        hits3 = result.metric_results.get_metric('both.realistic.hits_at_3')
        hits5 = result.metric_results.get_metric('both.realistic.hits_at_5')
        hits10 = result.metric_results.get_metric('both.realistic.hits_at_10')

         # --- Tempo de inferência pura
        dataset = dataset_resolver.lookup(dataset_name)()
        triples = dataset.testing.mapped_triples
        n_test = int(triples.shape[0]) if hasattr(triples, "shape") else len(triples)

        model = result.model
        device_torch = model.device
        # CONVERSÃO ROBUSTA:
        if isinstance(triples, torch.Tensor):
            triples_tensor = triples.to(device=device_torch, dtype=torch.long)
        else:
            # as_tensor evita cópia se já for tensor; em numpy -> cria tensor
            triples_tensor = torch.as_tensor(triples, dtype=torch.long)
            triples_tensor = triples_tensor.to(device=device_torch)

        # inferência
        with torch.inference_mode():
            infer_t0 = time.perf_counter()
            for j in range(0, n_test, inference_batch_size):   # use 'j' para não conflitar
                batch = triples_tensor[j:j+inference_batch_size]
                _ = model.score_hrt(batch)
            # sincroniza só se CUDA
            if device_torch.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(device_torch)
            infer_t1 = time.perf_counter()

        infer_time = (infer_t1 - infer_t0) / n_test if n_test else None
        
        # após terminar este experimento (antes do próximo loop):
        del batch
        del triples_tensor
        del model
        del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # sincroniza para garantir operações pendentes concluidas
            torch.cuda.synchronize()
        # --- montar resultado em forma de DataFrame (1 linha)
        df = pd.DataFrame([{
            "model": model_name,
            "dataset": dataset_name,
            "seed": seed,
            "epochs": epochs,
            "train_time": train_time,
            "eval_time": eval_time,
            "inference_time": infer_time,
            "mrr": mrr,
            "hits@1": hits1,
            "hits@3": hits3,
            "hits@5": hits5,
            "hits@10": hits10,
            "gpu_mem_avg_MB": avg_mem,
            "gpu_mem_peak_MB": peak_mem,
            "gpu_util_avg_%": avg_util,
            "gpu_util_peak_%": peak_util,
            "gpu_power_avg_W": avg_power,
            "gpu_power_peak_W": peak_power,
            "gpu_energy_Wh": total_energy_wh,
        }])
        metrics.append(df)
    pynvml.nvmlShutdown()
        
        
    return pd.concat(metrics, ignore_index=True)