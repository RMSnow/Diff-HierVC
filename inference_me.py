import os
import torch
import argparse 
from tqdm import tqdm
import numpy as np 
from glob import glob
from scipy.io.wavfile import write
from torch.nn import functional as F   
import torchaudio
import copy  
import utils.utils as utils
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic 
from vocoder.hifigan import HiFi 
from vocoder.bigvgan import BigvGAN 
from model.diffhiervc import DiffHierVC, Wav2vec2 
from utils.utils import MelSpectrogramFixed

h = None
device = None
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)      
    
def load_audio(path):
    audio, sr = torchaudio.load(path) 
    audio = audio[:1]
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000, resampling_method="kaiser_window")
    
    p = (audio.shape[-1] // 1280 + 1) * 1280 - audio.shape[-1] 
    audio = torch.nn.functional.pad(audio, (0, p)) 
     
    return audio 

def save_audio(wav, out_file, syn_sr=16000):
    wav = (wav.squeeze() / wav.abs().max() * 0.999 * 32767.0).cpu().numpy().astype('int16')
    write(out_file, syn_sr, wav) 

def get_yaapt_f0(audio, sr=16000, interp=False):
    to_pad = int(20.0 / 1000 * sr) // 2
    f0s = []
    for y in audio.astype(np.float64):
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0) 
        pitch = pYAAPT.yaapt(basic.SignalObj(y_pad, sr), 
                             **{'frame_length': 20.0, 'frame_space': 5.0, 'nccf_thresh1': 0.25, 'tda_frame_length': 25.0})
        f0s.append(pitch.samp_interp[None, None, :] if interp else pitch.samp_values[None, None, :])

    return np.vstack(f0s)  
 
    

def conversion(content_wav_path, reference_wav_path, output_path):
    # Convert audio 
    print('>> Converting each utterance...') 
    src_name = os.path.splitext(os.path.basename(content_wav_path))[0]
    audio = load_audio(content_wav_path)   

    src_mel = mel_fn(audio.cuda())
    src_length = torch.LongTensor([src_mel.size(-1)]).cuda()
    w2v_x = w2v(F.pad(audio, (40, 40), "reflect").cuda())

    try:
        f0 = get_yaapt_f0(audio.numpy())
    except:
        f0 = np.zeros((1, audio.shape[-1] // 80), dtype=np.float32)
 
    f0_x = f0.copy()
    f0_x = torch.log(torch.FloatTensor(f0_x+1)).cuda()
    ii = f0 != 0
    f0[ii] = (f0[ii] - f0[ii].mean()) / f0[ii].std()
    f0_norm_x = torch.FloatTensor(f0).cuda()
   
    trg_name = os.path.splitext(os.path.basename(reference_wav_path))[0] 
    trg_audio = load_audio(reference_wav_path)    

    trg_mel = mel_fn(trg_audio.cuda())
    trg_length = torch.LongTensor([trg_mel.size(-1)]).to(device)     
    
    with torch.no_grad(): 
        c = model.infer_vc(src_mel, w2v_x, f0_norm_x, f0_x, src_length, trg_mel, trg_length, 
                                    diffpitch_ts=a.diffpitch_ts, diffvoice_ts=a.diffvoice_ts) 
        converted_audio = net_v(c)  
        
    # f_name = f'{src_name}_to_{trg_name}.wav' 
    # out = os.path.join(a.output_dir, f_name)
    save_audio(converted_audio, output_path)   



if __name__ == '__main__':
    print('>> Initializing Inference Process...')
     
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='/data/home/xueyao/workspace/Diff-HierVC/sample/src_p241_004.wav')  
    parser.add_argument('--trg_path', type=str, default='/data/home/xueyao/workspace/Diff-HierVC/sample/tar_p239_022.wav')  
    parser.add_argument('--ckpt_model', type=str, default='./ckpt/model_diffhier.pth')
    parser.add_argument('--voc', type=str, default='bigvgan')  
    parser.add_argument('--ckpt_voc', type=str, default='./vocoder/voc_bigvgan.pth')  
    parser.add_argument('--output_dir', '-o', type=str, default='./converted') 
    parser.add_argument('--diffpitch_ts', '-dpts', type=int, default=30) 
    parser.add_argument('--diffvoice_ts', '-dvts', type=int, default=6)  
    
    # global hps, hps_voc, device, a 
    global hps_voc
    a = parser.parse_args()
    config = os.path.join(os.path.split(a.ckpt_model)[0], 'config_bigvgan.json')  
    hps = utils.get_hparams_from_file(config) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(a.output_dir, exist_ok=True) 
    mel_fn = MelSpectrogramFixed(
        sample_rate=hps.data.sampling_rate,
        n_fft=hps.data.filter_length,
        win_length=hps.data.win_length,
        hop_length=hps.data.hop_length,
        f_min=hps.data.mel_fmin,
        f_max=hps.data.mel_fmax,
        n_mels=hps.data.n_mel_channels,
        window_fn=torch.hann_window
    ).cuda()

    # Load pre-trained w2v (XLS-R)
    w2v = Wav2vec2().cuda()  
    
    # Load model
    model = DiffHierVC(hps.data.n_mel_channels, hps.diffusion.spk_dim,
                        hps.diffusion.dec_dim, hps.diffusion.beta_min, hps.diffusion.beta_max, hps).cuda() 
    model.load_state_dict(torch.load(a.ckpt_model))
    model.eval()
    
    # Load vocoder
    if a.voc == "hifigan":
        net_v = HiFi(hps.data.n_mel_channels, hps.train.segment_size // hps.data.hop_length, **hps.model).cuda()
        utils.load_checkpoint(a.ckpt_voc, net_v, None)
    elif a.voc == "bigvgan":
        net_v = BigvGAN(hps.data.n_mel_channels, hps.train.segment_size // hps.data.hop_length, **hps.model).cuda()
        utils.load_checkpoint(a.ckpt_voc, net_v, None) 
    net_v.eval().dec.remove_weight_norm()

    # conversion(content_wav_path=a.src_path, reference_wav_path=a.trg_path, output_path="test.wav")
    output_root = "/data/home/xueyao/workspace/dataset/EvaluationSamples"

    # ========= g0, g1 =========
    for data_dir in ["g0", "g1"]:
        data_dir = os.path.join(output_root, data_dir)
        print("\nFor {}...".format(data_dir))
        for i in tqdm(range(200)):
            filename = "{:04}.wav".format(i+1)
            content_wav_path = os.path.join(data_dir, "content", filename)
            reference_wav_path = os.path.join(data_dir, "reference", filename)
            output_path = os.path.join(data_dir, "conversion", filename)

            assert os.path.exists(content_wav_path)
            assert os.path.exists(reference_wav_path)
            # print(output_path)

            # Conversion
            conversion(content_wav_path, reference_wav_path, output_path)

    # ========= g2, g3 =========
    for data_dir in ["g2", "g3"]:
        data_dir = os.path.join(output_root, data_dir)
        print("\nFor {}...".format(data_dir))

        content_files = glob(os.path.join(data_dir, "content", "*.wav"))
        reference_files = glob(os.path.join(data_dir, "reference", "*.wav"))
        content_files.sort()
        reference_files.sort()
        
        assert len(content_files) == 30
        assert len(reference_files) == 6

        conversion_num = 0
        for content_wav_filename in tqdm(content_files):
            for reference_wav_filename in (reference_files):
                content_wav_filename = os.path.basename(content_wav_filename)
                reference_wav_filename = os.path.basename(reference_wav_filename)

                content_wav_path = os.path.join(data_dir, "content", content_wav_filename)
                reference_wav_path = os.path.join(data_dir, "reference", reference_wav_filename)
                output_path = os.path.join(data_dir, "conversion", "{}-{}".format(content_wav_filename.split('.')[0], reference_wav_filename))

                assert os.path.exists(content_wav_path)
                assert os.path.exists(reference_wav_path)
                # print(output_path)

                # Conversion
                conversion(content_wav_path, reference_wav_path, output_path)
                conversion_num += 1
        
        assert conversion_num == 180
        print("#Conversion = {}".format(conversion_num))
