"""This file will investigate the ARCHS4 data by comparing it to the MRSA data we have
on-hand. The goal is to see if bimodal expression patterns are present in both"""

from mrsa_ca_rna.utils import map_genes, concat_datasets, calculate_cpm
from mrsa_ca_rna.import_data import import_mrsa_tfac

def get_data():

    # Import MRSA archs4 data
    X = concat_datasets(filter_threshold=-1)
    X = X[X.obs["disease"] == "MRSA", :]
    
    X.layers["cpm"] = calculate_cpm(X.layers["raw"].copy())

    # Import MRSA TFAC data
    tfac_mrsa = import_mrsa_tfac()
    tfac_mrsa.layers["cpm"] = calculate_cpm(tfac_mrsa.X)

    # Map tfac genes from ensembl to symbol
    mapping = map_genes(
        tfac_mrsa.var_names.to_list(),
        gtf_path="mrsa_ca_rna/data/gencode.v48.comprehensive.annotation.gtf.gz",
        from_type="ensembl",
        to_type="symbol",
    )

    tfac_mrsa.var_names.map(mapping, inplace=True)

    return X, tfac_mrsa
