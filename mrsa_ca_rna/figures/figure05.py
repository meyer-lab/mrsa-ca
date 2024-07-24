"""plotting time dependent patients"""

import pandas as pd

from mrsa_ca_rna.import_data import extract_time_data, concat_datasets
from mrsa_ca_rna.pca import perform_PCA

def figure05_setup():
    rna_notime = concat_datasets()
    rna_time = extract_time_data()

    rna_all = pd.concat([rna_notime.loc[:, ~rna_notime.columns.str.contains("ststus|disease")], rna_time.loc[:, ~rna_time.columns.str.contains("subject_id|days_rel_first_time|status|disease")]],
        axis=0,
        join="inner"
        )

    rna_decomp = perform_PCA(rna_all)

    return "foo"
figure05_setup()