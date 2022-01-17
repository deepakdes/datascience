import pandas as pd

def build_dataframe():
    print("Dataframe built")

if __name__ == '__main__':
    if os.path.isfile('/data/something.parquet'):
        print("Skipping as the data frame file already exsists")
    else:
        build_dataframe()

