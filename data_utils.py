import os
import random
import hashlib
import numpy as np
import torch
import torch.utils.data

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, text_to_sequence_auto


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.detect_text_encoding = getattr(
            hparams, 'detect_text_encoding', True)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)

        # Transparent mel cache. Filelists keep their .wav paths; each mel is
        # computed once and reused, which matters when the same wav appears on
        # several lines (grapheme / ARPAbet / IPA transcriptions).
        self.mel_cache_dir = getattr(hparams, 'mel_cache_dir', '')
        self._audio_key = '|'.join(str(v) for v in (
            hparams.sampling_rate, hparams.filter_length, hparams.hop_length,
            hparams.win_length, hparams.n_mel_channels, hparams.mel_fmin,
            hparams.mel_fmax, hparams.max_wav_value))
        if self.mel_cache_dir:
            os.makedirs(self.mel_cache_dir, exist_ok=True)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel)

    def get_mel(self, filename):
        cache_path = self._mel_cache_path(filename)
        if cache_path is not None and os.path.isfile(cache_path):
            try:
                return torch.from_numpy(np.load(cache_path))
            except Exception:
                pass  # corrupt or half-written; fall through and recompute

        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        if cache_path is not None:
            self._save_mel_atomic(cache_path, melspec)

        return melspec

    def _mel_cache_path(self, filename):
        """Cache filename, keyed on the wav path AND the audio hparams.

        Keying on the audio settings means changing hop_length or mel_fmax
        produces different cache files rather than silently reusing stale
        mels computed under the old settings.
        """
        if not self.mel_cache_dir:
            return None
        digest = hashlib.md5(
            (os.path.abspath(filename) + '|' + self._audio_key).encode()
        ).hexdigest()[:16]
        stem = os.path.splitext(os.path.basename(filename))[0]
        return os.path.join(self.mel_cache_dir, f"{stem}_{digest}.npy")

    @staticmethod
    def _save_mel_atomic(cache_path, melspec):
        """Write via a temp file so concurrent workers can't read a partial .npy."""
        try:
            tmp = f"{cache_path}.{os.getpid()}.tmp"
            np.save(tmp, melspec.numpy())
            os.replace(tmp, cache_path)
        except Exception:
            pass  # caching is an optimisation; never fail training over it

    def get_text(self, text):
        # Route per line: grapheme, ARPAbet and IPA transcriptions of the same
        # wav can coexist in one filelist. Set hparams.detect_text_encoding to
        # False to restore the old single-cleaner behaviour.
        if getattr(self, 'detect_text_encoding', True):
            sequence = text_to_sequence_auto(text, self.text_cleaners)
        else:
            sequence = text_to_sequence(text, self.text_cleaners)
        return torch.IntTensor(sequence)

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths
