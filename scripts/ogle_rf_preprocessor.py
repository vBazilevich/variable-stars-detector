import warnings
import argparse
import os
import sys
import queue
import multiprocessing
import tqdm
import numpy as np
import FFATS

warnings.filterwarnings('error') # Treat warning as errors

MIN_ROWS = 10 # Skip curves for which we have less that MIN_ROWS measurements
MAX_ROWS = 500 # Limit the length of timeseries

def preprocessor(filepath):
    feature_space = FFATS.FeatureSpace(Data=['magnitude', 'time', 'error'], featureList=None)
    # Open file
    time, mag, error = [], [], []
    with open(filepath, 'r') as f:
        n_read = 0
        for line in f.readlines():
            line = line.strip()
            values = list(map(float, line.split()))
            time.append(values[0])
            mag.append(values[1])
            error.append(values[2])
            n_read += 1
            if n_read >= MAX_ROWS:
                break

    if n_read < MIN_ROWS:
        print(f'File {filepath} doesn\'t have enough data points. Skipping', flush=True)
        return '', [], None

    time = np.array(time)
    mag = np.array(mag)
    error = np.array(error)

    try:
        preprocessed_data = FFATS.Preprocess_LC(mag, time, error)
        lc = np.array(preprocessed_data.Preprocess())
        feature_space.calculateFeature(lc)
    except Exception as e:
        print(f'Exception encountered while processing {filepath}')
        print(e, flush=True)
        return '', [],  None

    label = os.path.split(os.path.dirname(filepath))[-1]

    return filepath, feature_space.featureList, feature_space.result(method=None) + [label]

def cache_writer(write_cache):
    for path, features, result in write_cache:
        with open(path, 'w') as f:
            f.write(','.join(features + ['label']) + '\n')
            f.write(','.join(map(str, result)) + '\n')

if __name__ == '__main__':
    # Define parser
    parser = argparse.ArgumentParser(prog='OGLE-III preprocessor',
                                     description='Preprocesses OGLE-III dataset using FFATS features extractor')
    parser.add_argument('data_root', help='Root directory of OGLE-III dataset')
    parser.add_argument('destination', help='Location where processed file must be stored')

    args = parser.parse_args()
    data_root = args.data_root
    destination = args.destination

    if os.path.exists(data_root):
        print('Succesfully found dataset root directory')
    else:
        print('Invalid path to root directory. Quitting...')
        sys.exit(1)


    ogle_files = []
    for rt, _, files in os.walk(data_root):
        for file in files:
            if not os.path.exists(os.path.join(destination, file)):
                ogle_files.append(os.path.join(rt, file))
    TOTAL_FILES = len(ogle_files)

    if not os.path.exists(destination):
        os.makedirs(destination)

    pbar = tqdm.tqdm(total=TOTAL_FILES, desc='Files written so far: ', smoothing=0.05)
    WC_MAX_SIZE = 128 # Save every WC_MAX_SIZE files on a disk
    write_cache = []
    batch_writers = []
    with multiprocessing.Pool(20) as p:
        for fname, features, result in p.imap_unordered(preprocessor, ogle_files):
            if result is None:
                continue

            pbar.update()
            if len(write_cache) < WC_MAX_SIZE:
                fname = os.path.basename(fname)
                path = os.path.join(destination, fname)
                write_cache.append((path, features, result))
            else:
                writer = multiprocessing.Process(target=cache_writer, args=(write_cache,), daemon=False)
                writer.start()
                batch_writers.append(writer)
                write_cache = []

                # Try to join writers
                running_writers = []
                for writer in batch_writers:
                    writer.join(timeout=0)
                    if writer.is_alive():
                        running_writers.append(writer)
                batch_writers = running_writers

    # Write last batch and join all writers
    cache_writer(write_cache)
    for w in batch_writers:
        w.join()
