from typing import Any

import pandas as pd

class data:
    @staticmethod
    def series(file_path: str, geo_accession: str) -> pd.DataFrame: ...

class utils:
    @staticmethod
    def aggregate_duplicate_genes(counts: pd.DataFrame) -> pd.DataFrame: ...
    @staticmethod
    def normalize(
        counts: pd.DataFrame, method: str = "tmm", tmm_outlier: float = 0.05
    ) -> pd.DataFrame: ...
    @staticmethod
    def get_config() -> dict[str, Any]: ...
    @staticmethod
    def versions() -> list[str]: ...
    @staticmethod
    def ls() -> list[str]: ...

class meta:
    @staticmethod
    def series(file_path: str, geo_accession: str) -> pd.DataFrame: ...

class align:
    @staticmethod
    def load(sras: list[str], outfolder: str) -> None: ...
    @staticmethod
    def fastq(
        species: str,
        fastq: str,
        release: str = "latest",
        t: int = 8,
        overwrite: bool = False,
        return_type: str = "transcript",
        identifier: str = "symbol",
    ) -> pd.DataFrame | pd.Series: ...
    @staticmethod
    def folder(
        species: str,
        folder: str,
        return_type: str = "transcript",
        release: str = "latest",
        overwrite: bool = False,
        t: int = 8,
        identifier: str = "symbol",
    ) -> pd.DataFrame: ...
    @staticmethod
    def aggregate(
        transcript_count: pd.DataFrame, species: str, release: str, identifier: str
    ) -> pd.DataFrame: ...

class download:
    @staticmethod
    def counts(
        species: str, path: str = "", type: str = "GENE_COUNTS", version: str = "latest"
    ) -> str: ...

def versions() -> list[str]: ...
def normalize(
    counts: pd.DataFrame, method: str = "tmm", tmm_outlier: float = 0.05
) -> pd.DataFrame: ...
def ls() -> list[str]: ...
