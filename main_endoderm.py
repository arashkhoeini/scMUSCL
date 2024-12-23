import warnings
warnings.filterwarnings("ignore")
from tabulate import tabulate
import anndata as ann
from configs.init_configs import init_config, show_config
import pickle
from pathlib import Path
from data.experiment import Experiment
import numpy as np
import torch
from trainer import Trainer
import sys
import pandas as pd
from datetime import datetime
import os



def init_seed(seed):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def store_result(config, result):
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M")
    path = Path(f'./output/result/{config.task}/{dt_string}-{os.getpid()}')
    path.mkdir(parents=True, exist_ok=True)
    metrics = ['adj_mi', 'accuracy', 'recall',
               'adj_rand', 'nmi', 'precision', 'f1_score']
    avg_result = {key: [] for key in metrics}
    avg_result['tissue'] = []
    for tissue in result.keys():
        for m in metrics:
            l = np.array([result[tissue][rep][m] for rep in result[tissue].keys()])
            avg_result[m].append(f'{l.mean():.3f} +- {l.std():.3f}')
        avg_result['tissue'].append(tissue)
    df_result = pd.DataFrame(avg_result)
    df_result = df_result.set_index('tissue')
    with open(path / f'mean_result.txt', 'w') as f:
        f.write(tabulate(df_result,
                headers='keys', tablefmt='psql'))
    with open(path / 'configs.txt', 'w') as f:
        f.write(show_config(config))
    df_result.to_csv(path / f'mean_result_csv.txt')
    return df_result


def get_experiments(config):
    experiments = {}
    for src_path in config.source:
        adata = ann.read_h5ad(config.data_path+src_path)
        exp = Experiment(adata.X, adata.obs_names,
                         adata.var_names.values, src_path, adata.obs.y)
        experiments[src_path[:src_path.index('.')]] = exp

    return experiments


def main():
    config = init_config("configs/endoderm.yml", sys.argv)
    if config.fix_seed:
        init_seed(config.seed)

    experiments = get_experiments(config)
    all_result = {}
    for i, tissue in enumerate(experiments.keys()):
        result = {}
        tgt_experiment = experiments[tissue]
        src_experiments = [experiments[key]
                           for key in experiments.keys() if key != tissue]

        src_labels_uniq = set()
        for exp in src_experiments:
            src_labels_uniq = src_labels_uniq.union(set(exp.y))

        for rep in range(config.reps):
            if config.verbose:
                print(f'repetition {rep}')

            if torch.cuda.is_available() and not config.cuda:
                print(
                    "WARNING: You have a CUDA device, so you should probably run with --cuda")
            device = 'cuda:0' if torch.cuda.is_available() and config.cuda else 'cpu'

            model = Trainer(src_experiments, tgt_experiment,
                            n_clusters=len(src_labels_uniq),
                            config=config,
                            device=device)

            adata_result, eval_result, src_result = model.train()
            print(f'repetition {rep} results')
            print(f"\tTarget ARI: {eval_result['adj_rand']}")
            print(f"\tSource ARI: {src_result['adj_rand']}")
            eval_result['adata'] = adata_result
            result[rep] = eval_result
        all_result[tissue] = result
    store_result(config, all_result)


if __name__ == '__main__':
    main()
