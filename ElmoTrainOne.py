import mindspore
import mindspore.nn as nn
import mindspore.ops as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore import context, Tensor
from mindspore.context import ParallelMode
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.communication.management import get_group_size


grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()
@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)

_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()

@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)

class ElmoTrainOnestepWithLoss(nn.Cell):
    def __init__(self, network, optimizer, scale_update_cell=None, enable_global_norm=True):
        super().__init__()
        self.network = network
        self.optimizer = optimizer
        self.weights = optimizer.parameters
        self.grad = P.GradOperation(get_by_list=True)
        self.reducer_flag = False
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.allreduce = P.AllReduce()
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = F.identity
        self.degree = 1
        self.less_equal = P.LessEqual()
        self.base = Tensor(1, mstype.float32)
        if self.reducer_flag:
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)

        self.cast = P.Cast()
        if context.get_context("device_target") == "GPU":
            self.gpu_target = True
            self.float_status = P.FloatStatus()
            self.addn = P.AddN()
            self.reshape = P.Reshape()
        else:
            self.gpu_target = False
            self.alloc_status = P.NPUAllocFloatStatus()
            self.get_status = P.NPUGetFloatStatus()
            self.clear_before_grad = P.NPUClearFloatStatus()

        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))    
        self.hyper_map = C.HyperMap()

    def construct(self, inputs, inputs_back, targets, targets_back, sens=None):
        weights = self.weights
        loss = self.network(inputs, inputs_back, targets, targets_back)

        init = False
        if not self.gpu_target:
            # alloc status and clear should be right before gradoperation
            init = self.alloc_status()
            self.clear_before_grad(init)

        grads = self.grad(self.network, weights)(inputs, 
                                                inputs_back,
                                                targets,
                                                targets_back)

        # grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(F.partial(grad_scale, self.loss_scale*self.degree), grads)
        grads = P.clip_by_global_norm(grads, 10.0)

        train_perplexity = P.Exp()(loss/20)

        if not self.gpu_target:
            self.get_status(init)
            flag_sum = self.reduce_sum(init, (0,))
        else:
            flag_sum = self.hyper_map(F.partial(_grad_overflow), grads)
            flag_sum = self.addn(flag_sum)
            flag_sum = self.reshape(flag_sum, (()))
        if self.reducer_flag:
            # sum overflow flag over devices
            flag_reduce = self.allreduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)
        ret = (train_perplexity, cond, self.loss_scale)
        return F.depend(train_perplexity, succ)




class ElmoTrainOnestepWithLossAscend(nn.TrainOneStepWithLossScaleCell):
    def __init__(self, network, optimizer, scale_update_cell=None, enable_global_norm=True):
        super(ElmoTrainOnestepWithLossAscend, self).__init__(network, optimizer, scale_update_cell)
        self.network = network
        self.optimizer = optimizer
        self.weights = optimizer.parameters
        self.grad = P.GradOperation(get_by_list=True)
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)

        self.cast = P.Cast()     

        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell is not None:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))    
        self.hyper_map = C.HyperMap()

    def construct(self, inputs, inputs_back, targets, targets_back, sens=None):
        weights = self.weights
        loss = self.network(inputs, inputs_back, targets, targets_back)

        grads = self.grad(self.network, weights)(inputs, 
                                                inputs_back,
                                                targets,
                                                targets_back)
        
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        # grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens*self.degree), grads)
        grads = P.clip_by_global_norm(grads, 10.0)

        cond = self.get_overflow_status(status, grads)
        
        train_perplexity = P.Exp()(loss/20)

        overflow = cond
        if self.loss_scaling_manager is not None:
            overflow = self.loss_scaling_manager(scaling_sens, cond)
       
        succ = self.optimizer(grads)
        ret = (train_perplexity, cond, scaling_sens)
        return F.depend(train_perplexity, succ)
