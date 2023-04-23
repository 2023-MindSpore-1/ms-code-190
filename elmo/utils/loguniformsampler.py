import numpy as np
from mindspore import Tensor
import mindspore.ops as P
import mindspore.nn as nn
from mindspore import dtype as mstype

def choice_func(num_words: int, num_samples: int, targets):
        """
        Chooses `num_samples` samples without replacement from [0, ..., num_words).
        Returns a tuple (samples, num_tries).
        """
        num_tries = 0
        num_chosen = 0

        def get_buffer() -> np.ndarray:
            log_samples = np.random.rand(num_samples) * np.log(num_words + 1)
            samples = np.exp(log_samples).astype("int64") - 1
            return np.clip(samples, a_min=0, a_max=num_words - 1)

        sample_buffer = get_buffer()
        buffer_index = 0
        samples = set()
       
        while num_chosen < num_samples:
            num_tries += 1
            sample_id = sample_buffer[buffer_index]

            if sample_id not in targets:
                if sample_id not in samples:
                    samples.add(sample_id)
                    num_chosen += 1

            buffer_index += 1
            if buffer_index == num_samples:
                # Reset the buffer
                sample_buffer = get_buffer()
                buffer_index = 0

        return np.array(list(samples)), num_tries

class LogUniformCandidateSampler(nn.Cell):
    def __init__(self, vocab_size, num_sampled, num_true=1):
        super(LogUniformCandidateSampler,self).__init__()
        self._num_words = vocab_size
        self._num_samples = num_sampled
        self.exp = P.Exp()
        self.log = P.Log()
        self.initialize_num_words()
       
    def initialize_num_words(self):

        self._log_num_words_p1 = np.log(self._num_words+1.0)
        # compute the probability of each sampled id
        self._probs = (
            np.log(np.arange(self._num_words) + 2) - np.log(np.arange(self._num_words) + 1)
        ) / self._log_num_words_p1
  
    

    def construct(self, targets):
        # returns sampled, true_expected_count, sampled_expected_count
        # targets = (batch_size, )
        #
        #  samples = (n_samples, )
        #  true_expected_count = (batch_size, )
        #  sampled_expected_count = (n_samples, )

        # see: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/range_sampler.h
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/range_sampler.cc

        # algorithm: keep track of number of tries when doing sampling,
        #   then expected count is
        #   -expm1(num_tries * log1p(-p))
        # = (1 - (1-p)^num_tries) where p is self._probs[id]
        targets = targets.view((-1,1))
        targets = targets.asnumpy()
        sampled_ids, num_tries = choice_func(self._num_words, self._num_samples, targets)

        
        # Compute expected count = (1 - (1-p)^num_tries) = -expm1(num_tries * log1p(-p))
        # P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)
        target_probs = (
            np.log((targets + 2.0) / (targets + 1.0)) / self._log_num_words_p1
        )
        target_expected_count = -1.0 * (np.exp(num_tries * np.log1p(-target_probs)) - 1.0)
        sampled_probs = (
            np.log((sampled_ids + 2.0) / (sampled_ids + 1.0))
            / self._log_num_words_p1
        )
        sampled_expected_count = -1.0 * (np.exp(num_tries * np.log1p(-sampled_probs)) - 1.0)

        return (sampled_ids, target_expected_count, sampled_expected_count)


