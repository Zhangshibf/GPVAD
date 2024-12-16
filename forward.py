import torch
import sys
from loguru import logger
from pathlib import Path
from tqdm import tqdm
import utils
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import uuid
import argparse
from models import crnn
import os


import sklearn.preprocessing
sys.modules['sklearn.preprocessing.label'] = sys.modules['sklearn.preprocessing._label'] 


SAMPLE_RATE = 22050
EPS = np.spacing(1)
LMS_ARGS = {
    'n_fft': 2048,
    'n_mels': 64,
    'hop_length': int(SAMPLE_RATE * 0.02),
    'win_length': int(SAMPLE_RATE * 0.04)
}
DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
DEVICE = torch.device(DEVICE)


def extract_feature(wavefilepath, **kwargs):
    wavefilepath = wavefilepath.replace("\n","")
    _, file_extension = os.path.splitext(wavefilepath)
    if file_extension == '.wav':
        
        wav, sr = sf.read(wavefilepath, dtype='float32')
    if file_extension == '.mp3':
        wav, sr = librosa.load(wavefilepath)
    elif file_extension not in ['.mp3', '.wav']:
        print("________________")
        print(wavefilepath)
        print("________________")
        print(file_extension)
        print("________________")
        raise NotImplementedError('Audio extension not supported... yet ;)')
    if wav.ndim > 1:
        wav = wav.mean(-1)
    wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
    return np.log(
        librosa.feature.melspectrogram(y=wav.astype(np.float32), sr=SAMPLE_RATE, **
                                       kwargs) + EPS).T


class OnlineLogMelDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, **kwargs):
        super().__init__()
        self.dlist = data_list
        self.kwargs = kwargs

    def __getitem__(self, idx):
        return extract_feature(wavefilepath=self.dlist[idx],
                               **self.kwargs), self.dlist[idx]

    def __len__(self):
        return len(self.dlist)


MODELS = {
    'sre': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'labelencoders/students.pth',
        'pretrained': 'sre/model.pth',
        'resolution': 0.02
    }
}


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-w',
        '--wav',
        help=
        'A single wave/mp3/flac or any other compatible audio file with soundfile.read'
    )

    parser.add_argument('-o',
                        '--output_path',
                        default=None,
                        help='Output file to save soft predictions')

    args = parser.parse_args()
    pretrained_dir = Path('pretrained_models')

    logger.info("Passed args")
    for k, v in vars(args).items():
        logger.info(f"{k} : {str(v):<10}")

    wavlist = [args.wav]

    dset = OnlineLogMelDataset(wavlist, **LMS_ARGS)
    dloader = torch.utils.data.DataLoader(dset,
                                          batch_size=1,
                                          num_workers=3,
                                          shuffle=False)

    model_kwargs_pack = MODELS['sre']
    model_resolution = float(model_kwargs_pack['resolution'])
    # Load model from relative path
    model = model_kwargs_pack['model'](
        outputdim=model_kwargs_pack['outputdim'],
        pretrained_from=pretrained_dir /
        model_kwargs_pack['pretrained']).to(DEVICE).eval()
    encoder = torch.load(pretrained_dir / model_kwargs_pack['encoder'])
    logger.trace(model)


    speech_label_idx = np.where('Speech' == encoder.classes_)[0].squeeze()

    with torch.no_grad(), tqdm(total=len(dloader), leave=False,
                               unit='clip') as pbar:
        for feature, filename in dloader:
            feature = torch.as_tensor(feature).to(DEVICE)
            _, prediction_time = model(feature)
            prediction_time = prediction_time.to('cpu')
            speech_soft_pred = prediction_time[
                ..., speech_label_idx].numpy()


    output_path = Path(args.output_path)
    folder_path= os.path.dirname(output_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(output_path,'w') as wp:
        for n,prob in enumerate(speech_soft_pred[0]):
            start = n*model_resolution
            end = (n+1)*model_resolution
            print(prob)
            prob = float(prob)
            line = start+"\t"+end+"\t"+prob+"\n"
            wp.write(line)


if __name__ == "__main__":
    main()
