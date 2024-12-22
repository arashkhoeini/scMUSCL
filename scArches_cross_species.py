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
import warnings
import os
warnings.filterwarnings("ignore")
from model.metrics import compute_scores
import scanpy as sc
import scarches as sca
import torch


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
            l = np.array([result[tissue][rep][m]
                         for rep in result[tissue].keys()])
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
        # exp = Experiment(adata.X, adata.obs_names,
        #                  adata.var_names.values, src_path, adata.obs.y)
        # src_labels_uniq = src_labels_uniq.union(set(exp.y))
        src_experiments.append(adata)
    tgt_adata = ann.read_h5ad(config.data_path+config.target[idx])
    # tgt_experiment = Experiment(tgt_adata.X, tgt_adata.obs_names,
    #                             tgt_adata.var_names.values, config.target[idx], tgt_adata.obs.y)
    return src_experiments, tgt_adata


def main():
    config = init_config("configs/cross_species.yml", sys.argv)
    if config.fix_seed:
        init_seed(config.seed)

    all_result = {}
    for i in range(len(config.source)):
        src_experiments, tgt_experiment = get_experiments(
            config, i)
        
        result = {}
        for rep in range(config.reps):
            
            if config.verbose:
                print(f'repetition {rep}')

            if torch.cuda.is_available() and not config.cuda:
                print(
                    "WARNING: You have a CUDA device, so you should probably run with --cuda")
            # device = 'cuda:0' if torch.cuda.is_available() and config.cuda else 'cpu'
            celltypes, preds = train(src_experiments, tgt_experiment)
            eval_result = compute_scores(celltypes, preds)
            result[rep] = eval_result
        all_result[config.target[i][config.target[i].index(
            '/')+1:config.target[i].index('.')]] = result
    df_result = store_result(config, all_result)
    return df_result

def train(src_data, tgt_data):
    
    for i, src in enumerate(src_data):
        src.obs['batch'] = f'src_{i}'
    src = ann.concat(src_data, axis=0)
    tgt_data.obs['batch'] = 'tgt_0'
    sca.models.SCVI.setup_anndata(src, batch_key=f'batch')
    vae = sca.models.SCVI(
                            src,
                            n_layers=3,
                            encode_covariates=True,
                            deeply_inject_covariates=False,
                            use_layer_norm="both",
                            use_batch_norm="none",
                        )
    vae.train(max_epochs=400)
    ref_path = 'ref_model/'
    vae.save(ref_path, overwrite=True)
    model = sca.models.SCVI.load_query_data(
                                tgt_data,
                                ref_path,
                                freeze_dropout = True,
                            )
    model.train(max_epochs=200, plan_kwargs=dict(weight_decay=0.0))
    query_latent = sc.AnnData(model.get_latent_representation())
    stoi = {t:i for i,t in enumerate(tgt_data.obs['y'].unique())}
    query_latent.obs['y'] = [stoi[t] for t in tgt_data.obs['y'].tolist()]
    sc.pp.neighbors(query_latent)
    sc.tl.leiden(query_latent)
    celltypes = query_latent.obs['y'].values
    preds = query_latent.obs['leiden'].values

    return celltypes, preds

if __name__ == '__main__':
    main()
