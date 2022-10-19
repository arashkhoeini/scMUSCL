from tabulate import tabulate
import anndata as ann
from configs.init_configs import init_config, show_config
import pickle
from pathlib import Path
from data.experiment import Experiment
import numpy as np
import torch
from model.trainer import Trainer
import sys
import pandas as pd
from datetime import datetime
import warnings
import os
warnings.filterwarnings("ignore")


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
    avg_result['target'] = []
    for tissue in result.keys():
        for m in metrics:
            l = np.array([result[tissue][rep][m] for rep in result[tissue].keys()])
            avg_result[m].append(f'{l.mean():.3f} +- {l.std():.3f}')
        avg_result['target'].append(tissue)
    df_result = pd.DataFrame(avg_result)
    df_result = df_result.set_index('target')
    with open(path / f'mean_result.txt', 'w') as f:
        f.write(tabulate(df_result,
                headers='keys', tablefmt='psql'))
    with open(path / 'configs.txt', 'w') as f:
        f.write(show_config(config))
    df_result.to_csv(path / f'mean_result_csv.txt')
    return df_result



def get_experiments(config, idx):
    src_experiments = []
    src_labels_uniq = set()
    src_list = config.source[idx]
    for src_path in src_list:
        adata = ann.read_h5ad(config.data_path+src_path)
        exp = Experiment(adata.X, adata.obs_names,
                         adata.var_names.values, src_path, adata.obs.y)
        src_labels_uniq = src_labels_uniq.union(set(exp.y))
        src_experiments.append(exp)
    tgt_adata = ann.read_h5ad(config.data_path+config.target[idx])
    tgt_experiment = Experiment(tgt_adata.X, tgt_adata.obs_names,
                                tgt_adata.var_names.values, config.target[idx], tgt_adata.obs.y)
    return src_experiments, tgt_experiment, src_labels_uniq


def main():
    config = init_config("configs/cross_tissue.yml", sys.argv)
    if config.fix_seed:
        init_seed(config.seed)

    
    all_result = {}
    for i in range(len(config.source)):
        src_experiments, tgt_experiment, src_labels_uniq = get_experiments(
            config, i)

        result = {}
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
            print(f'Repetition {rep}')
            print(f"\tTarget ARI: {eval_result['adj_rand']}")
            print(f"\tSource ARI: {src_result['adj_rand']}")
            eval_result['adata'] = adata_result
            result[rep] = eval_result
        all_result[config.target[i][config.target[i].index('/')+1:config.target[i].index('.')]] = result
    store_result(config, all_result)


if __name__ == '__main__':
    main()
