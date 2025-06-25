from collections import OrderedDict
import os
import torch
import torch.nn as nn
import numpy as np 
from torch.nn.utils.rnn import pad_sequence

from ..interfaces import UpstreamBase
from .audio import create_transform
from .decoar2 import Decoar2

EXAMPLE_FEAT_SEQLEN = 1000


class UpstreamExpert(UpstreamBase):
    """
    The APC wrapper
    """

    def __init__(self, ckpt, **kwargs):
        super(UpstreamExpert, self).__init__()
        models = torch.load(ckpt)["model"]
        self.model = Decoar2()
        component_state_dict = OrderedDict()
        for key in models.keys():
            component_state_dict[key] = models[key]
        self.model.load_state_dict(component_state_dict, strict=False)

        self.model.encoder.layerdrop = 0.0

        self.preprocessor = create_transform()
        self.output_dim = 768

        if len(self.hooks) == 0:
            module_name = "self.model.encoder.layers"
            for module_id in range(len(eval(module_name))):
                self.add_hook(
                    f"{module_name}[{module_id}]",
                    lambda input, output: input[0].transpose(0, 1),
                )
            self.add_hook("self.model.encoder", lambda input, output: output[0])

            def postprocess(xs):
                names, hiddens = zip(*xs)
                unpad_len = min([hidden.size(1) for hidden in hiddens])
                hiddens = [hidden[:, :unpad_len, :] for hidden in hiddens]
                return list(zip(names, hiddens))

            self.hook_postprocess = postprocess

        self._init_layerdrop = self.model.encoder.layerdrop
        self.cache = '/home/nervjack2/s3prl-merge/decoar2_cache/'

    def set_require_grad(self, finetune_step, global_step, freeze_conv_layer=True, lpft_or_not=True):
        if global_step < finetune_step:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            for name, param in self.model.named_parameters():
                if (not name.startswith('feature_extractor')) or (not freeze_conv_layer):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    @property
    def layer_drop(self):
        return self.model.encoder.layerdrop

    def set_layer_drop(self, layerdrop: float = None):
        if isinstance(layerdrop, float):
            self.model.encoder.layerdrop = layerdrop
        elif layerdrop is None:
            self.model.encoder.layerdrop = self._init_layerdrop
        else:
            raise ValueError("layerdrop can only be float or None")

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs, file_name=None):
        """
        Args:
            wavs:
                list of unpadded wavs [wav1, wav2, ...]
                each wav is in torch.FloatTensor with sample rate 16000
                and already put in the device assigned by command-line args

        Return:
            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args
        """
        if file_name:
            features = []
            for wav, f in zip(wavs, file_name):
                path = f'{self.cache}/{f}.pth'
                if not os.path.exists(path):
                    feat = self.preprocessor(wav.unsqueeze(0))
                    torch.save(feat, path)
                else:
                    feat = torch.load(path)
                features.append(feat)
        else:
            features = [self.preprocessor(wav.unsqueeze(0)) for wav in wavs]
        feat_lengths = [len(feat) for feat in features]
        size = max(feat_lengths)
        features = pad_sequence(features, batch_first=True)

        padding_mask = torch.BoolTensor(features.shape).fill_(False).to(features.device)

        for i in range(len(feat_lengths)):
            diff = feat_lengths[i] - size
            if diff == 0:
                continue
            padding_mask[i, diff:] = True

        features, layer_results = self.model(features, padding_mask)

        # This forward function only does the model forward
        # The return dict is then handled by UpstreamBase's hooks
