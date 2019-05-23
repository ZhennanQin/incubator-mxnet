import logging
import warnings
import time
import mxnet as mx
import mxnet.gluon as gluon
from mxnet import autograd
import gluoncv as gcv
from gluoncv.model_zoo import get_model

from mxnet.contrib import amp

amp.init(target_dtype='bfloat16')

data_shape = 512
batch_size = 8
lr = 0.001
wd = 0.0005
momentum = 0.9

# training contexts
ctx = [mx.cpu()]

# set up logger
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

ce_metric = mx.metric.Loss('CrossEntropy')
smoothl1_metric = mx.metric.Loss('SmoothL1')

class SyntheticDataLoader(object):
    def __init__(self, data_shape, batch_size):
        super(SyntheticDataLoader, self).__init__()
        self.counter = 0
        self.epoch_size = 200
        shape = (batch_size, 3, data_shape, data_shape)
        cls_targets_shape = (batch_size, 6132)
        box_targets_shape = (batch_size, 6132, 4)
        self.data = mx.nd.random.uniform(-1, 1, shape=shape, ctx=mx.cpu_pinned())
        self.cls_targets = mx.nd.random.uniform(0, 1, shape=cls_targets_shape, ctx=mx.cpu_pinned())
        self.box_targets = mx.nd.random.uniform(0, 1, shape=box_targets_shape, ctx=mx.cpu_pinned())

    def next(self):
        if self.counter >= self.epoch_size:
            self.counter = self.counter % self.epoch_size
            raise StopIteration
        self.counter += 1
        return [self.data, self.cls_targets, self.box_targets]

    __next__ = next

    def __iter__(self):
        return self

train_data = SyntheticDataLoader(data_shape, batch_size)

def get_network():
    # SSD with RN50 backbone
    net_name = 'ssd_512_resnet50_v1_coco'
    net = get_model(net_name, pretrained_base=True, norm_layer=gluon.nn.BatchNorm)
    async_net = net
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        net.initialize()
        net.collect_params().reset_ctx(ctx)

    return net

net = get_network()
net.hybridize(static_alloc=True, static_shape=True)

trainer = gluon.Trainer(
    net.collect_params(), 'sgd',
    {'learning_rate': lr, 'wd': wd, 'momentum': momentum})

amp.init_trainer(trainer)

mbox_loss = gcv.loss.SSDMultiBoxLoss()

for epoch in range(1):
    ce_metric.reset()
    smoothl1_metric.reset()
    tic = time.time()
    btic = time.time()

    for i, batch in enumerate(train_data):
        batch_size = batch[0].shape[0]
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
        with autograd.record():
            cls_preds = []
            box_preds = []
            for x in data:
                cls_pred, box_pred, _ = net(x)
                cls_preds.append(cls_pred)
                box_preds.append(box_pred)
            sum_loss, cls_loss, box_loss = mbox_loss(
                cls_preds, box_preds, cls_targets, box_targets)
            autograd.backward(sum_loss)
        trainer.step(1)
        ce_metric.update(0, [l * batch_size for l in cls_loss])
        smoothl1_metric.update(0, [l * batch_size for l in box_loss])
        if not (i + 1) % 50:
            name1, loss1 = ce_metric.get()
            name2, loss2 = smoothl1_metric.get()
            logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2))
        btic = time.time()
