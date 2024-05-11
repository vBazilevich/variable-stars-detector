import joblib
from concurrent.futures.process import ProcessPoolExecutor
import asyncio
import FFATS
from fastapi import FastAPI, UploadFile
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
import shap
import logging
from sklearn.model_selection import train_test_split

ml_models = {}
FEATURE_NAMES = []

MAX_ROWS = 500 # Limit the length of timeseries

# List of features excluded from training as they have the constant value
ALL_SAME = ['Freq1_harmonics_rel_phase_0',
           'Freq2_harmonics_rel_phase_0',
           'Freq3_harmonics_rel_phase_0']

# List of classes which were excluded from training due to small number of observations
EXCLUDED_CLASSES = ['rcb', 'acv', 'yso', 'wd']


def load_train_data():
    logging.info('Loading train data for LIME')
    # Read processed data
    data = pd.read_csv('../data/combined.csv').drop(columns='Unnamed: 0')

    # Drop columns containing same values
    data = data.drop(columns=ALL_SAME)
    data = data[~data.label.isin(EXCLUDED_CLASSES)]

    np.random.seed(42)
    MAX_LCS = 4000
    samples = []
    for cl in data.label.unique():
        cl_data = data[data.label == cl]
        if cl_data.shape[0] <= MAX_LCS:
            samples.append(cl_data)
        else:
            sample = cl_data.sample(n=MAX_LCS)
            samples.append(sample)
    sampled_data = pd.concat(samples)

    y = sampled_data.label.reset_index(drop=True)
    X = sampled_data.drop(columns=['label']).reset_index(drop=True)
    FEATURE_NAMES = X.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y)

    logging.info('Successfully loaded the training data')
    return X_train

def preprocessor(contents):
    feature_space = FFATS.FeatureSpace(Data=['magnitude', 'time', 'error'], featureList=None)
    # Open file
    time, mag, error = [], [], []
    n_read = 0
    for line in contents.splitlines():
        line = line.strip()
        values = list(map(float, line.split()))
        time.append(values[0])
        mag.append(values[1])
        error.append(values[2])
        n_read += 1
        if n_read >= MAX_ROWS:
            break

    time = np.array(time)
    mag = np.array(mag)
    error = np.array(error)

    try:
        preprocessed_data = FFATS.Preprocess_LC(mag, time, error)
        lc = np.array(preprocessed_data.Preprocess())
        feature_space.calculateFeature(lc)
    except Exception as e:
        return None

    data = pd.DataFrame(np.array(feature_space.result(method=None)).reshape(1, -1), columns=feature_space.featureList)
    data = data.drop(columns=ALL_SAME)
    return data


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info('Loading RF classifier...')
    ml_models['rf'] = joblib.load('../models/tuned_rf.joblib')
    logging.info('Loaded RF classifier successfully')

    logging.info('Preparing LIME...')
    X_train = load_train_data()
    ml_models['interpret'] = shap.TreeExplainer(ml_models['rf'], data=X_train)
    logging.info('LIME is ready')

    app.state.executor = ProcessPoolExecutor(2)
    yield
    ml_models.clear()
    app.state.executor.shutdown()

app = FastAPI(lifespan=lifespan)

@app.post('/predict')
async def predict(file: UploadFile):
    data = await file.read()
    loop = asyncio.get_event_loop()
    features = await loop.run_in_executor(app.state.executor, preprocessor, data)
    pred = ml_models['rf'].predict_proba(features)[0]

    logging.info(f'pred: {pred}, best: {np.argmax(pred)}')
    explanation = ml_models['interpret'].shap_values(features)[0, :, np.argmax(pred)]

    response = {
            'probas': dict(zip(ml_models['rf'].classes_, pred)),
            'explanations': dict(zip(ml_models['rf'].feature_names_in_, explanation)),
            }

    return response
