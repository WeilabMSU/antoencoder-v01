# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re

from fairseq.data.encoders import register_tokenizer
from fairseq.dataclass import FairseqDataclass
from fairseq.tokenizer import tokenize_smiles



@register_tokenizer("smi", dataclass=FairseqDataclass)
class SmiTokenizer(object):
    def __init__(self, *unused):
        pass
        
    def encode(self, x: str) -> str:
        return ' '.join(tokenize_smiles(x))

    def decode(self, x: str) -> str:
        return x
