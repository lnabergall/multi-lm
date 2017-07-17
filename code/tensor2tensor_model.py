"""
Classes and functions for multi-domain language modeling 
using Tensor2Tensor.

Four primary models: LSTM, LSTM + Mixture-of-Experts, Attention Network,
                     Attention Network + Mixture-of-Experts
"""

from tensor2tensor.models import lstm, attention_lm, attention_lm_moe
from tensor2tensor.utils import registry


