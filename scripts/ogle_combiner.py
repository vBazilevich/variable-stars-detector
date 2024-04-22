import csv
import os
import argparse
import pandas as pd
import tqdm

if __name__ == '__main__':
    # Define parser
    parser = argparse.ArgumentParser(prog='OGLE-III preprocessed data combiner',
                                     description='Combine preprocessed OGLE-III dataset')
    parser.add_argument('data_root', help='Root directory of preprocessed OGLE-III dataset')
    parser.add_argument('destination', help='Location where processed file must be stored')

    args = parser.parse_args()
    data_root = args.data_root
    destination = args.destination

    if os.path.exists(data_root):
        print('Succesfully found dataset root directory')
    else:
        print('Invalid path to root directory. Quitting...')
        sys.exit(1)

    preprocessed_data = []
    for rt, _, files in os.walk(data_root):
        for file in tqdm.tqdm(files, leave=False):
            data = pd.read_csv(os.path.join(rt, file))
            preprocessed_data.append(data)

    combined_data = pd.concat(preprocessed_data)

    if not os.path.exists(destination):
        print(f'Destination path {destination} does not exist. Creating necessary folders.')
        os.makedirs(destination)

    combined_data.to_csv(os.path.join(destination, 'combined.csv'))
