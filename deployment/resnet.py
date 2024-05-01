from concurrent.futures.process import ProcessPoolExecutor
import asyncio
from fastapi import FastAPI, UploadFile
from contextlib import asynccontextmanager
import torch
from torch import nn
from torchvision import models, transforms
import torchaudio
import sklearn
import sklearn.preprocessing
import numpy as np
import pandas as pd
import logging
import io

ml_models = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_ROWS = 500 # Limit the length of timeseries

class BasicBlockWithDropout(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockWithDropout, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def replace_layers(model, block):
    for name, module in model.named_children():
        if isinstance(module, models.resnet.BasicBlock):
            setattr(model, name, block(module.conv1.in_channels,
                                       module.conv2.out_channels,
                                       module.stride,
                                       module.downsample))
        elif "layer" in name:
            # Replace all basic blocks in each layer
            for name2, sub_module in module.named_children():
                if isinstance(sub_module, models.resnet.BasicBlock):
                    module[int(name2)] = block(sub_module.conv1.in_channels,
                                               sub_module.conv2.out_channels,
                                               sub_module.stride,
                                               sub_module.downsample)
        else:
            replace_layers(module, block)

def restore_label_encoder():
    le = sklearn.preprocessing.LabelEncoder()
    le.classes_ = np.array(['acep', 'cep', 'dn', 'dsct', 'dpv', 'ecl', 'lpv', 'rrlyr', 't2cep'])
    return le

def restore_resnet():
    resnet = models.resnet18(pretrained=True)
    replace_layers(resnet, BasicBlockWithDropout)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 9)
    resnet.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return resnet

def preprocessor(contents):
    data = pd.read_csv(io.StringIO(contents.decode('utf-8')), sep = "\s+", header=None).reset_index(drop=True)
    data.columns = ['Time', 'Magnitude', 'Err']
    data = data.astype('float64')
    data = data[data.Time > 0]
    data.sort_values(by=['Time'])
    data.drop_duplicates(subset='Time', keep='first')

    # Compute differences between consecutive observations for time and magnitude
    # Note: diff() leaves the first element as NaN, which we replace with zeros
    data['Time_diff'] = data['Time'].diff().fillna(0)
    data['Magnitude_diff'] = data['Magnitude'].diff().fillna(0)

    # Prepare for padding if necessary, targeting a minimum of 500 observations
    min_observations = 500
    padding_needed = max(0, min_observations - len(data))

    # If padding is needed, extend the dataframe with rows of zeros
    if padding_needed > 0:
        padding_df = pd.DataFrame({
            'Time': [0] * padding_needed,
            'Magnitude': [0] * padding_needed,
            'Err': [0] * padding_needed,
            'Time_diff': [0] * padding_needed,
            'Magnitude_diff': [0] * padding_needed
        })
        data = pd.concat([data, padding_df], ignore_index=True)

    # After padding, if the dataset is larger than the target size, truncate it to the target size
    data = data.head(min_observations)

    # Since we're only interested in Time_diff and Magnitude_diff for the network input,
    # we reshape our data accordingly to a shape of 1 × 2 × N (1 light curve, 2 features, N observations)
    data = torch.tensor(data[['Magnitude_diff', 'Time_diff']].values, dtype=torch.float32).transpose(0, 1)

    sample_rate = 8000
    n_fft = 512  # Reducing the FFT size can help capture shorter time variations
    win_length = 512  # Window size for the FFT, can be the same as n_fft
    hop_length = 9 # Reducing hop length to increase the number of time steps
    n_mels = 64
    spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels
    )
    data = spectrogram_transform(data)

    return data.reshape(1, 2, 64, 56)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models['le'] = restore_label_encoder()
    resnet = restore_resnet().to(device)
    ml_models['resnet'] = resnet
    logging.info(f'{device} will be used for inference')
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
    pred = nn.functional.softmax(ml_models['resnet'](features.to(device)).cpu())[0]
    response = {tag : proba.item() for tag, proba in zip(ml_models['le'].classes_, pred)}
    print(response)
    return response
