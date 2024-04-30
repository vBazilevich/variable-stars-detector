import joblib
from concurrent.futures.process import ProcessPoolExecutor
import asyncio
import FFATS
from fastapi import FastAPI, UploadFile
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd

ml_models = {}

MAX_ROWS = 500 # Limit the length of timeseries

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
    data = data.drop(columns=['Freq1_harmonics_rel_phase_0', 'Freq2_harmonics_rel_phase_0', 'Freq3_harmonics_rel_phase_0'])
    return data


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models['rf'] = joblib.load('../models/tuned_rf.joblib')
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
    response = {tag : proba for tag, proba in zip(ml_models['rf'].classes_, pred)}
    print(response)
    return response
