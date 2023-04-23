import mindspore
import numpy as np
from mindspore.nn.cell import Cell
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.loss.loss import _Loss, _check_label_dtype
from mindspore.ops.primitive import constexpr

class LossBase(Cell):
    """
    Base class for other losses.

    Other losses derived from this should implement their own `construct` and use method `self.get_loss`
    to apply reduction to loss values.

    Args:
        reduction (str): Type of reduction to be applied to loss. The optional values are "mean", "sum", and "none".
            Default: "mean".

    Raises:
        ValueError: If `reduction` is not one of 'none', 'mean', 'sum'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    def __init__(self, reduction='mean'):
        """Initialize Loss."""
        super(LossBase, self).__init__()

        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"The reduction method for {reduction} is not supported")

        self.average = True
        self.reduce = True
        if reduction == 'sum':
            self.average = False
        if reduction == 'none':
            self.reduce = False

        self.reduce_mean = P.ReduceMean()
        self.reduce_sum = P.ReduceSum()
        self.mul = P.Mul()
        self.cast = P.Cast()

    def get_axis(self, x):
        """
        Get a range of axis for input.

        Args:
            x (Tensor): Tensor of any shape.
        """
        shape = F.shape(x)
        length = F.tuple_len(shape)
        perm = F.make_range(0, length)
        return perm

    def get_loss(self, x, weights=1.0):
        """
        Computes the weighted loss.

        Args:
            x (Tensor): Tensor of shape :math:`(N, *)` where :math:`*` means, any number of
                additional dimensions.
            weights (Union[float, Tensor]): Optional `Tensor` whose rank is either 0, or the same rank as inputs,
                and must be broadcastable to inputs (i.e., all dimensions must be either `1`,
                or the same as the corresponding inputs dimension).
        """
        input_dtype = x.dtype
        x = self.cast(x, mstype.float32)
        weights = self.cast(weights, mstype.float32)
        x = self.mul(weights, x)
        if self.reduce and self.average:
            x = self.reduce_mean(x, self.get_axis(x))
        if self.reduce and not self.average:
            x = self.reduce_sum(x, self.get_axis(x))
        x = self.cast(x, input_dtype)
        return x

    def construct(self, base, target):
        raise NotImplementedError

@constexpr
def _check_is_tensor(param_name, input_data, cls_name):
    if input_data is not None and not isinstance(F.typeof(input_data), mstype.tensor_type):
        raise TypeError(f"For '{cls_name}', the '{param_name}' should be '{mstype.tensor_type}', "
                        f"but got '{F.typeof(input_data)}'")

class SampledSoftmaxLoss(LossBase):
    r"""
    Computes the sampled softmax training loss. This operator can accelerate the training of the softmax classifier
    over a large number of classes. It is generally an underestimate of the full softmax loss.

    Args:
        num_sampled (int): The number of classes to randomly sample per batch.
        num_classes (int): The number of possible classes.
        num_true (int): The number of target classes per training example. Default: 1.
        sampled_values (Union[list, tuple]):  List or tuple of (`sampled_candidates`, `true_expected_count`,
            `sampled_expected_count`) returned by a `*CandidateSampler` function.
            Default to None, `UniformCandidateSampler` is applied.
        remove_accidental_hits (bool): Whether to remove "accidental hits"
            where a sampled class equals to one of the target classes. Default: True.
        seed (int): Random seed for candidate sampling. Default: 0
        reduction (str): Type of reduction to be applied to loss. The optional values are "mean", "sum", and "none".
            If "none", do not perform reduction. Default: "none".

    Inputs:
        - **weights** (Tensor) - Tensor of shape :math:`(C, dim)`.
        - **bias** (Tensor) - Tensor of shape :math:`(C,)`. The class biases.
        - **labels** (Tensor) - Tensor of shape :math:`(N, num\_true)`, type `int64, int32`. The target classes.
        - **logits** (Tensor) - Tensor of shape :math:`(N, dim)`. The forward activations of the input network.

    Outputs:
        Tensor or Scalar, if `reduction` is 'none', then output is a tensor with shape :math:`(N,)`.
        Otherwise, the output is a scalar.

    Raises:
        TypeError: If `sampled_values` is not a list or tuple.
        TypeError: If dtype of `labels` is neither int32 not int64.
        ValueError: If `reduction` is not one of 'none', 'mean', 'sum'.
        ValueError: If `num_sampled` or `num_true` is greater than `num_classes`.
        ValueError: If length of `sampled_values` is not equal to 3.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> mindspore.set_seed(1)
        >>> loss = nn.SampledSoftmaxLoss(num_sampled=4, num_classes=7, num_true=1)
        >>> weights = Tensor(np.random.randint(0, 9, [7, 10]), mindspore.float32)
        >>> biases = Tensor(np.random.randint(0, 9, [7]), mindspore.float32)
        >>> labels = Tensor([0, 1, 2])
        >>> logits = Tensor(np.random.randint(0, 9, [3, 10]), mindspore.float32)
        >>> output = loss(weights, biases, labels, logits)
        >>> print(output)
        [4.6051701e+01 1.4000047e+01 6.1989022e-06]
    """

    def __init__(self, num_sampled, num_classes, num_true=1,
                 sampled_values=None, remove_accidental_hits=True, seed=0,
                 reduction='none'):
        """Initialize SampledSoftmaxLoss."""
        super(SampledSoftmaxLoss, self).__init__(reduction)

        if num_true < 1:
            raise ValueError(f"The num_true {num_true} is less than 1.")
        if seed < 0:
            raise ValueError(f"The seed {seed} is less than 0.")
        if num_sampled > num_classes:
            raise ValueError(f"The num_sampled {num_sampled} is greater than num_classes {num_classes}.")
        if num_true > num_classes:
            raise ValueError(f"The num_true {num_true} is greater than num_classes {num_classes}.")
        if sampled_values is not None:
            if not isinstance(sampled_values, (list, tuple)):
                raise TypeError(f"The sampled_values {sampled_values} is not a list or tuple.")
            if len(sampled_values) != 3:
                raise ValueError(f"The sampled_values size {len(sampled_values)} is not 3.")

        self.num_sampled = num_sampled
        self.num_classes = num_classes
        self.num_true = num_true
        self.sampled_values = sampled_values
        self.remove_accidental_hits = remove_accidental_hits
        self.seed = seed
        self.sampler = P.UniformCandidateSampler(
            num_true,
            num_sampled,
            True,
            num_classes,
            seed,
            remove_accidental_hits)
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.exp = P.Exp()
        self.log = P.Log()
        self.slice_op = P.Slice()
        self.matmul = P.MatMul(False, True)
        self.gather_v2 = P.Gather()
        self.reduce_max_true = P.ReduceMax(True)
        self.reduce_sum = P.ReduceSum()
        self.reduce_sum_true = P.ReduceSum(True)
        self.concat_dim0 = P.Concat(0)
        self.concat_dim1 = P.Concat(1)
        self.ones_like = P.OnesLike()
        self.zeros_like = P.ZerosLike()
        self.mul = P.Mul()
        self.expand_dims = P.ExpandDims()
        self.dtype = P.DType()

    def construct(self, weights, biases, labels, inputs):
        _check_is_tensor('weights', weights, self.cls_name)
        _check_is_tensor('biases', biases, self.cls_name)
        _check_is_tensor('labels', labels, self.cls_name)
        _check_is_tensor('inputs', inputs, self.cls_name)
        _check_label_dtype(self.dtype(labels), self.cls_name)

        logits, labels = self._compute_sampled_logits(
            weights=weights,
            biases=biases,
            labels=labels,
            inputs=inputs,
            num_true=self.num_true,
            sampled_values=self.sampled_values,
            subtract_log_q=True)

        x = self._softmax_cross_entropy(logits, labels)
        return x

    def _softmax_cross_entropy(self, logits, targets):
        stable_exp_logits = self.exp(logits - self.reduce_max_true(logits, 1))
        pred = stable_exp_logits / self.reduce_sum_true(stable_exp_logits, 1)
        return -self.reduce_sum(targets * self.log(pred + 1.0e-20), 1)

    def _compute_sampled_logits(self, weights,
                                biases,
                                labels,
                                inputs,
                                num_true=1,
                                sampled_values=None,
                                subtract_log_q=True):
        """Helper function for SampledSoftmaxLoss functions.

        Computes sampled output training logits and labels suitable

        Note: In the case where num_true > 1, we assign to each target class
        with the target probability (1/num_true) so that the target probabilities
        sum to 1 per-example.

        Args:
            weights (Tensor): Tensor of shape `[num_classes, dim]`.
            biases (Tensor): Tensor of shape `[num_classes]`.
            labels (Tensor): Tensor of shape `[batch_size, num_true]`. The target classes.
            inputs (Tensor): Tensor of shape `[batch_size, dim]`. The forward
                activations of the input network.
            num_true (int): The number of target classes per training example.
            sampled_values: A tuple of (`sampled_candidates`, `true_expected_count`,
                `sampled_expected_count`) returned by a `UniformCandidateSampler` function.
            subtract_log_q: A `bool`. whether to subtract the log expected count of
                the labels in the sample to get the logits of the true labels. Default: True.
        Returns:
            out_logits: `Tensor` object with shape
                `[batch_size, num_true + num_sampled]`
            out_labels: A tensor object with the same shape as `out_logits`.
        """

        if not labels.dtype == mstype.int32:
            labels = self.cast(labels, mstype.int32)
        labels = self.reshape(labels, (-1, num_true))
        labels_flat = self.reshape(labels, (-1,))

        # Sample the negative labels.
        #   sampled shape: [num_sampled] tensor
        #   true_expected_count shape is [batch_size, 1] tensor
        #   sampled_expected_count shape is [num_sampled] tensor
        if sampled_values is None:
            sampled_values = self.sampler(self.cast(labels, mstype.int64))

        (sampled, true_expected_count, sampled_expected_count) = sampled_values
        sampled = ops.stop_gradient(sampled)
        true_expected_count = ops.stop_gradient(true_expected_count)
        sampled_expected_count = ops.stop_gradient(sampled_expected_count)

        if not sampled.dtype == mstype.int32:
            sampled = self.cast(sampled, mstype.int32)
        all_ids = self.concat_dim0((labels_flat, sampled))
        all_w = self.gather_v2(weights, all_ids, 0)

        n_true = self.shape(labels_flat)[0]
        n_sampled = self.shape(sampled)[0]
        n_dim = self.shape(all_w)[1]

        true_w = self.slice_op(all_w, [0, 0], [n_true, n_dim])
        sampled_w = self.slice_op(all_w, [n_true, 0], [n_sampled, n_dim])
        sampled_w = self.cast(sampled_w, mindspore.float16)
        sampled_logits = self.matmul(inputs, sampled_w)

        all_b = self.gather_v2(biases, all_ids, 0)
        true_b = self.slice_op(all_b, [0], [n_true])
        sampled_b = self.slice_op(all_b, [n_true], [n_sampled])

        new_true_w_shape = (-1, num_true, n_dim)
        row_wise_dots = self.mul(self.expand_dims(inputs, 1),
                                 self.reshape(true_w, new_true_w_shape))

        # We want the row-wise dot plus biases which yields a
        # [batch_size, num_true] tensor of true_logits.
        dots_as_matrix = self.reshape(row_wise_dots, (-1, n_dim))
        true_logits = self.reshape(self.reduce_sum(dots_as_matrix, 1), (-1, num_true))
        true_b = self.reshape(true_b, (-1, num_true))
        true_logits += true_b
        sampled_logits += sampled_b

        if subtract_log_q:
            # Subtract log of Q(l), prior probability that l appears in sampled.
            true_logits -= self.log(true_expected_count)
            sampled_logits -= self.log(sampled_expected_count)

        # Construct output logits and labels. The true labels/logits start at col 0.
        out_logits = self.concat_dim1((true_logits, sampled_logits))

        # true_logits is a float tensor, ones_like(true_logits) is a float
        # tensor of ones. We then divide by num_true to ensure the per-example
        # labels sum to 1.0, i.e. form a proper probability distribution.
        out_labels = self.concat_dim1((
            self.ones_like(true_logits) / num_true,
            self.zeros_like(sampled_logits)
        ))
        return out_logits, out_labels