
import scanpy as sc


# def preprocess_data(adata: ann.AnnData, scale :bool=True):
#     """Preprocessing dataset: filtering genes/cells, normalization and scaling."""
#     sc.pp.filter_cells(adata, min_counts=5000)
#     sc.pp.filter_cells(adata, min_genes=500)

#     sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
#     adata.raw = adata

#     sc.pp.log1p(adata)
#     if scale:
#         sc.pp.scale(adata, max_value=10, zero_center=True)
#         adata.X[np.isnan(adata.X)] = 0

#     return adata

def preprocess_data(adata):
    sc.pp.normalize_total(adata, target_sum = 10000)
    sc.pp.log1p(adata)
    return adata
