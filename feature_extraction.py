from audio_utils import *
from utils import *

def _get_mfcc_and_spec(wav, preemphasis_coeff, n_fft, win_length, hop_length):

    # Pre-emphasis
    if hp.default.do_preemphasis:
        y_preem = preemphasis(wav, coeff=preemphasis_coeff)
    else:
        y_preem = wav

    # Get spectrogram
    D = librosa.stft(y=y_preem, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(D)

    # Get mel-spectrogram
    mel_basis = librosa.filters.mel(hp.default.sr, hp.default.n_fft, hp.default.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t) # mel spectrogram

    # Get mfccs, amp to db
    mag_db = amp2db(mag)
    mel_db = amp2db(mel)
    mfccs = np.dot(librosa.filters.dct(hp.default.n_mfcc, mel_db.shape[0]), mel_db)

    # Normalization (0 ~ 1)
    mag_db = normalize_0_1(mag_db, hp.default.max_db, hp.default.min_db)
    mel_db = normalize_0_1(mel_db, hp.default.max_db, hp.default.min_db)

    return mfccs.T, mag_db.T, mel_db.T  # (t, n_mfccs), (t, 1+n_fft/2), (t, n_mels)


def get_mfccs_and_phones(wav_file, phone_vocab, trim=False, random_crop=True):

    '''This is applied in `train1` or `test1` phase.
    '''

    # Load
    wav = read_wav(wav_file, sr=hp.default.sr)

    mfccs, _, _ = _get_mfcc_and_spec(wav, hp.default.preemphasis, hp.default.n_fft,
                                     hp.default.win_length,
                                     hp.default.hop_length)

    # timesteps
    num_timesteps = mfccs.shape[0]

    # phones (targets)
    phn_file = wav_file.replace("WAV.wav", "PHN").replace("wav", "PHN")
    #phn_file = wav_file.replace("av0","avb0").replace("wav", "lbl")
    phn2idx, idx2phn = load_vocab(phone_vocab)
    phns = np.zeros(shape=(num_timesteps,))
    bnd_list = []
    for line in open(phn_file, 'r').read().splitlines():
        start_point, _, phn = line.split()
        bnd = int(start_point) // hp.default.hop_length
#         start_point,phn = line.split("\t")
#         start_point = float(start_point) * hp.default.sr
#         bnd = int(round(start_point)) // hp.default.hop_length
        phns[bnd:] = phn2idx[phn]
        bnd_list.append(bnd)

    # Trim
    if trim:
        start, end = bnd_list[1], bnd_list[-1]
        mfccs = mfccs[start:end]
        phns = phns[start:end]
        assert (len(mfccs) == len(phns))

    # Random crop
    n_timesteps = (hp.default.duration * hp.default.sr) // hp.default.hop_length + 1
    if random_crop:
        start = np.random.choice(range(np.maximum(1, len(mfccs) - n_timesteps)), 1)[0]
        end = start + n_timesteps
        mfccs = mfccs[start:end]
        phns = phns[start:end]
        assert (len(mfccs) == len(phns))

    # Padding or crop
    mfccs = librosa.util.fix_length(mfccs, n_timesteps, axis=0)
    phns = librosa.util.fix_length(phns, n_timesteps, axis=0)

    return mfccs, phns

def load_vocab(phns):
    phn2idx = {phn: idx for idx, phn in enumerate(phns)}
    idx2phn = {idx: phn for idx, phn in enumerate(phns)}

    return phn2idx, idx2phn

def get_mfccs_and_spectrogram(wav_file, trim=True, random_crop=False):
    '''This is applied in `train2`, `test2` or `convert` phase.
    '''


    # Load
    wav, _ = librosa.load(wav_file, sr=hp.default.sr)

    # Trim
    if trim:
        wav, _ = librosa.effects.trim(wav, frame_length=hp.default.win_length, hop_length=hp.default.hop_length)

    if random_crop:
        wav = wav_random_crop(wav, hp.default.sr, hp.default.duration)

    # Padding or crop
    length = hp.default.sr * hp.default.duration
    wav = librosa.util.fix_length(wav, length)

    return _get_mfcc_and_spec(wav, hp.default.preemphasis, hp.default.n_fft, hp.default.win_length, hp.default.hop_length)