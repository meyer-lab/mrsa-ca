from mrsa_ca_rna.import_data import extract_time_data
from mrsa_ca_rna.figures.base import setupBase

import seaborn as sns


def figure06_setup():
    ca_time_rna = extract_time_data()

    pat_time_meta = ca_time_rna.loc[:, [("meta", "subject_id"), ("meta", "time")]]

    return pat_time_meta


def genFig():
    fig_size = (6, 4)
    layout = {"ncols": 1, "nrows": 2}
    ax, f, _ = setupBase(fig_size, layout)

    data = figure06_setup()

    # drop level 0 of the multiindex columns because seaborn can only see the exterior level
    data.columns = data.columns.droplevel(0)

    # cast time as int so stipplot infers properly, and order by subject_id to line up plots
    data["time"] = data["time"].astype(int)
    sorted_data = data.sort_values(by=["subject_id"])

    a = sns.stripplot(
        data=sorted_data, x="subject_id", y="time", ax=ax[0], jitter=False
    )
    a.set_title("Time points collected for each patient")
    a.set_xlabel("Patient")
    a.set_ylabel("Time points (days)")
    a.tick_params(axis="x", rotation=45)

    a = sns.barplot(
        data=sorted_data.groupby("subject_id").count(),
        x="subject_id",
        y="time",
        ax=ax[1],
    )
    a.set_title("Total time points associated with each patient")
    a.set_xlabel("Patient")
    a.set_ylabel("# of time points")
    a.tick_params(axis="x", rotation=45)

    return f


genFig()
