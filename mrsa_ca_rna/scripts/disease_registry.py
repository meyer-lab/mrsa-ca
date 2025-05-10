#!/home/jamesp/mrsa-ca/mrsa_ca_rna/.venv/bin/activate

from mrsa_ca_rna.import_data import build_disease_registry


def main():
    build_disease_registry("disease_registry.json")


if __name__ == "__main__":
    main()
