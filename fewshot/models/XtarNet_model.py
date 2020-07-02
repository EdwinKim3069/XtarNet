from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import six
import tensorflow as tf

from tqdm import tqdm

from fewshot.models.model_factory import get_model
from fewshot.models.nnlib import weight_variable
from fewshot.models.nnlib import nullspace_signalling
from fewshot.models.nnlib import compute_euc, compute_logits_cosine
from fewshot.models.resnet_backbone import ResnetBackbone  # NOQA
import fewshot.models.metaCNN_resnet  # NOQA
from fewshot.models import MetaModules  # NOQA
from fewshot.models.basic_backbone import BasicBackbone  # NOQA
from fewshot.models.fc_backbone import FCBackbone  # NOQA
from fewshot.models.XtarNet_model_base import XtarNetModelBase
from fewshot.utils.checkpoint import build_checkpoint
from fewshot.utils.checkpoint import write_checkpoint
from fewshot.utils.logger import get as get_logger
import time

log = get_logger()


class XtarNetModel(XtarNetModelBase):
  """XtarNet model."""

  def __init__(self,
               config,
               x,
               y,
               x_b,
               y_b,
               x_b_v,
               y_b_v,
               num_classes_a,
               num_classes_b,
               is_training=True,
               y_sel=None,
               ext_wts=None,
               nshot=5):
    print('__init__')
    """

    Args:
      config: Model config object.
      x: Inputs on task A.
      y: Labels on task A.
      x_b: Support inputs on task B.
      y_b: Support labels on task B.
      x_b_v: Query inputs on task B.
      y_b_v: Query labels on task B.
      num_classes_a: Number of classes on task A.
      num_classes_b: Number of classes on task B.
      is_training: Whether in training mode.
      y_sel: Mask on base classes.
      ext_wts: External weights for initialization.
    """
    self._config = config
    self._is_training = is_training
    self._num_classes_a = num_classes_a
    self._num_classes_b = num_classes_b
    self._global_step = None
    self.nshot = nshot

    if config.backbone_class == 'resnet_backbone' or 'metaCNN' in config.backbone_class:
      bb_config = config.resnet_config
      self.bb_config = bb_config
    else:
      assert False, 'Not supported'
    opt_config = config.optimizer_config
    proto_config = config.protonet_config
    transfer_config = config.transfer_config
    ft_opt_config = transfer_config.ft_optimizer_config

    self._backbone = get_model(config.backbone_class, bb_config)
    self._inputs = x
    self._labels = y
    self._labels_all = self._labels

    self.encoder_g_gamma = get_model('Encoder_g_gamma', bb_config)
    self.encoder_g_beta = get_model('Encoder_g_beta', bb_config)
    self.encoder_r = get_model('Encoder_r', bb_config)
    self.encoder_h = get_model('Encoder_h', bb_config)
    self.encoder_h_pre = get_model('Encoder_h_pre', bb_config)

    self._y_sel = y_sel
    self._rnd = np.random.RandomState(0)  # Common random seed.

    # A step counter for the meta training stage.
    global_step = self.global_step

    log.info('LR decay steps {}'.format(opt_config.lr_decay_steps))
    log.info('LR list {}'.format(opt_config.lr_list))

    # Learning rate decay.
    learn_rate = tf.train.piecewise_constant(
        global_step, list(
            np.array(opt_config.lr_decay_steps).astype(np.int64)),
        list(opt_config.lr_list))
    self._learn_rate = learn_rate

    # Class matrix mask.
    self._mask = tf.placeholder(tf.bool, [], name='mask')

    # Optimizer definition.
    opt = self.get_optimizer(opt_config.optimizer, learn_rate)

    # Task A branch.
    # with tf.name_scope('TaskA'):
    #   self.build_task_a(x, y, is_training, ext_wts=ext_wts)
    #   if is_training:
    #     grads_and_vars_a = self.build_task_a_grad()
    #     with tf.variable_scope('Optimizer'):
    #       bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #       with tf.control_dependencies(bn_ops):
    #         self._train_op_a = opt.apply_gradients(
    #             grads_and_vars_a, global_step=global_step)
    # h_size = self._h_size  # Calculated in the function above.
    # w_class_a = self.w_class_a
    # b_class_a = self.b_class_a

    # The finetuning task.
    self._inputs_b = x_b
    self._labels_b = y_b
    self._inputs_b_v = x_b_v
    self._labels_b_v = y_b_v
    self._labels_b_v_all = y_b_v

    with tf.name_scope('TaskB'):
      self.build_task_b(x_b, y_b, x_b_v, y_sel)
      # if is_training:
      grads_and_vars_b = self.build_task_b_grad(x_b_v, y_b_v, y_sel)

    # Task A and Task B cost weights.
    assert transfer_config.cost_a_ratio == 0.0
    assert transfer_config.cost_b_ratio == 1.0
    cost_a_ratio_var = tf.constant(
        transfer_config.cost_a_ratio, name='cost_a_ratio', dtype=self.dtype)
    cost_b_ratio_var = tf.constant(
        transfer_config.cost_b_ratio, name='cost_b_ratio', dtype=self.dtype)

    # Update gradients for meta-leraning.
    if is_training:
      # total_grads_and_vars_ab = self._aggregate_grads_and_vars(
      #     [grads_and_vars_a, grads_and_vars_b],
      #     weights=[cost_a_ratio_var, cost_b_ratio_var])
      # with tf.variable_scope('Optimizer'):
      #   with tf.control_dependencies(bn_ops):
      #     self._train_op = opt.apply_gradients(
      #         total_grads_and_vars_ab, global_step=global_step)

      if len(grads_and_vars_b) > 0:
        with tf.variable_scope('Optimizer'):
          bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
          # print(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
          # print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
          with tf.control_dependencies(bn_ops):
            self._train_op_b = opt.apply_gradients(
               grads_and_vars_b, global_step=global_step)

      # else:
      #   self._train_op_b = tf.no_op()

    self._initializer = tf.global_variables_initializer()

    # var_list = tf.trainable_variables()
    # for v in var_list:
    #     print(v)
    print('init done!')
    print('')


  def build_task_b(self, x_b, y_b, x_b_v, y_sel):
    print('build_task_b')
    """Build task B.

    Args:
      x_b: Tensor. [S, H, W, C]. Support tensor.
      y_b: Tensor. [S]. Support labels.
      x_b_v: Tensor. [Q, H, W, C]. Query tensor.
      y_sel: Tensor. [K]. Mask class tensor.
    """
    transfer_config = self.config.transfer_config
    proto_config = self.config.protonet_config
    ft_opt_config = transfer_config.ft_optimizer_config
    opt_config = self.config.optimizer_config
    is_training = self._is_training
    # h_size = self._h_size
    num_classes_a = self._num_classes_a
    num_classes_b = self._num_classes_b
    # w_class_a = self._w_class_a
    # b_class_a = self._b_class_a
    # y_sel = self._y_sel
    old_and_new = transfer_config.old_and_new
    assert not proto_config.cosine_softmax_tau

    h_b, h_b_meta = self._run_backbone(x_b, reuse=True, is_training=is_training)
    h_shape = h_b.get_shape()
    h_size = 1
    for ss in h_shape[1:]:
        h_size *= int(ss)

    h_b_plh = tf.placeholder(self.dtype, [None, h_size], name='h_b_plh')
    h_b_plh_meta = tf.placeholder(self.dtype, [None, h_size], name='h_b_plh_meta')
    self._hidden_b = h_b
    self._hidden_b_plh = h_b_plh
    self._hidden_b_meta = h_b_meta
    self._hidden_b_plh_meta = h_b_plh_meta

    w_class_a = weight_variable([h_size, num_classes_a],
                                init_method='truncated_normal',
                                dtype=self.dtype,
                                init_param={'stddev': 0.01},
                                wd=self.bb_config.wd,
                                name='w_class_a')
    b_class_a = weight_variable([num_classes_a],
                                dtype=self.dtype,
                                init_method='constant',
                                init_param={'val': 0.0},
                                name='b_class_a')


    self._w_class_a = w_class_a
    self._b_class_a = b_class_a

    transfer_config = self.config.transfer_config
    num_classes_b = self.num_classes_b
    suffix = '' if self.num_classes_b == 5 else '_{}'.format(num_classes_b)
    if transfer_config.fast_model == 'lr':
        self.w_class_b = weight_variable([h_size, num_classes_b],
                                         dtype=self.dtype,
                                         init_method='truncated_normal',
                                         init_param={'stddev': 0.01},
                                         wd=transfer_config.finetune_wd,
                                         name='w_class_b' + suffix)


    self._h_b = h_b
    self._h_b_meta = h_b_meta

    support_key_all = tf.concat([h_b, h_b_meta], 1)
    average_key_all = tf.reduce_mean(tf.reshape(support_key_all, (num_classes_b, self.nshot, -1)), axis=1)
    average_key_task_all = tf.reduce_mean(average_key_all, axis=0, keepdims=True)

    w_meta = self.encoder_h(average_key_task_all, reuse=tf.AUTO_REUSE, h_size=h_size)
    w_pre = self.encoder_h_pre(average_key_task_all, reuse=tf.AUTO_REUSE, h_size=h_size)
    self.w_meta = w_meta
    self.w_pre = w_pre

    support_key = self.w_pre * h_b + self.w_meta * h_b_meta
    average_key = tf.reduce_mean(tf.reshape(support_key, (num_classes_b, self.nshot, -1)), axis=1)

    average_key_task = tf.reduce_mean(average_key, axis=0, keepdims=True)
    condi_vec = self.encoder_g_gamma(average_key_task, reuse=tf.AUTO_REUSE, h_size=h_size)
    condi_vec_b = self.encoder_g_beta(average_key_task, reuse=tf.AUTO_REUSE, h_size=h_size)

    mu = 0.1 * tf.transpose(condi_vec)
    lamb = 0.1 * tf.transpose(condi_vec_b)
    new_base_w = (1 + mu) * w_class_a + lamb

    corr = tf.matmul(average_key, new_base_w)
    corr_t = self.encoder_r(corr, reuse=tf.AUTO_REUSE, num_classes_a=self.num_classes_a)
    sm_corr = tf.nn.softmax(corr_t, axis=1)
    novel_w = self.w_class_b
    comp_w = tf.matmul(new_base_w, tf.transpose(sm_corr))
    novel_w = novel_w - comp_w

    M = nullspace_signalling(average_key, tf.transpose(novel_w), h_size=h_size)
    self.M = tf.stop_gradient(M)

    self.new_base_w_M = tf.matmul(tf.transpose(self.M), new_base_w)
    self.novel_w_M = tf.matmul(tf.transpose(self.M), novel_w)


  def resmlp(self, x, w3, w2, b2, w, b):
    """MLP with a residual connection."""
    return tf.matmul(tf.nn.tanh(tf.matmul(x, w2) + b2), w) + tf.matmul(x,
                                                                       w3) + b

  def fc(self, x, w, b):
    """Fully connected layer."""
    return tf.matmul(x, w) + b

  # Mask out the old classes.
  def get_mask_fn_wa(self, w_class_a, y_sel):
    """Mask the weights in task A."""
    num_classes_a = self.num_classes_a

    def _mask_fn():
      y_dense = tf.one_hot(y_sel, num_classes_a, dtype=self.dtype)
      bin_mask = tf.reduce_sum(y_dense, 0, keep_dims=True)
      wa = w_class_a * (1.0 - bin_mask) + 1e-7 * bin_mask
      return wa

    return _mask_fn

  def compute_logits_b(self, h, weights):
    print('compute_logits_b')
    """Compute logits for task B branch."""
    transfer_config = self.config.transfer_config
    # if transfer_config.fast_model == 'lr':
      # logits_b_v = self.fc(h, *weights)
    logits_b_v = compute_euc(tf.transpose(weights), h)
    # logits_b_v = compute_logits_cosine(tf.transpose(weights), h)
    # elif transfer_config.fast_model == 'resmlp':
    #   logits_b_v = self.resmlp(h, *weights)
    print('compute_logits_b done')
    return logits_b_v

  def compute_logits_b_all(self, h, w_b_list, w_a, b_a):
    print('compute_logits_b_all')
    """Compute logits for task B branch, possibly combined A logits."""
    # logits_b = self.compute_logits_b(h, w_b_list)
    # y_sel = self._y_sel
    if self.config.transfer_config.old_and_new:
      # logits_a = self.fc(h, w_a, b_a)
      new_weight = tf.concat([w_a, w_b_list], 1)
      # logits_a = compute_euc(tf.transpose(w_a), h)
      # logits_b = tf.concat([logits_a, logits_b], 1)
      logits_b = self.compute_logits_b(h, new_weight)
      # logits_b = tf.cond(self._mask, self.get_mask_fn(logits_b, y_sel),
      #                    lambda: logits_b)
    print('compute_logits_b_all done')
    return logits_b

  def build_task_b_grad(self, x_b_v, y_b_v, y_sel):
    print('build_task_b_grad')
    """Build gradients for task B.

    Args:
      x_b_v: Tensor. [Q, H, W, C]. Query tensor.
      y_b_v: Tensor. [Q]. Query label.
    """
    config = self.config
    transfer_config = config.transfer_config

    # This is the meta-learning gradients.
    assert transfer_config.meta_only

    meta_weights = self.get_meta_weights()

    # print('-----line-----')
    # for mw in meta_weights:
    #     print(mw)
    num_classes_a = self.num_classes_a
    num_classes_b = self.num_classes_b
    w_class_a = self.w_class_a
    b_class_a = self.b_class_a
    old_and_new = transfer_config.old_and_new
    is_training = self._is_training
    debug_rbp = False

    # Run again on the validation.
    h_b_v, h_b_v_meta = self._run_backbone(x_b_v, reuse=True, is_training=is_training)
    query_key = self.w_pre * h_b_v + self.w_meta * h_b_v_meta
    self.query_key_M = tf.matmul(query_key, self.M)

    # Loss on the task B validation set, for meta-learning.
    logits_b_v = self.compute_logits_b_all(self.query_key_M, self.novel_w_M,
                                           self.new_base_w_M, b_class_a)

    self.logits_b_v = logits_b_v
    self._prediction_b = logits_b_v
    self._prediction_b_all = self._prediction_b
    if is_training:
      correct_b_v = tf.equal(
          tf.argmax(self._prediction_b_all, axis=-1), self._labels_b_v_all)
      self._acc_b_v = tf.reduce_mean(tf.cast(correct_b_v, logits_b_v.dtype))

      xent_b_v = tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(
              logits=logits_b_v, labels=y_b_v))
      self._cost_b_v = xent_b_v
      meta_cost = xent_b_v + self._decay()

      # Stores gradients.
      grads_b_ckpt = build_checkpoint(meta_weights, 'grads_b')
      transfer_config = self.config.transfer_config
      has_tloss_cache = transfer_config.cache_transfer_loss_var
      assert not has_tloss_cache

      # Target of gradients.
      x = meta_weights  # [Trans + Meta]

      grads_b = tf.gradients(meta_cost, x)

      self.grads_b_rbp = grads_b
      # Write to gradient variables.
      log.info('Meta grads')
      grads_meta = grads_b

      # for x, grad in zip(meta_weights, grads_meta):
      #     print(x.name + ': ')
          # print(grad)

      self._meta_weights = meta_weights
      self._grads_meta = grads_meta
      update_grads_b = write_checkpoint(grads_b_ckpt, grads_meta)
      self._update_grads_b = update_grads_b
      print('build_task_b_grad done')
      return list(zip(grads_b_ckpt, meta_weights))


  def minibatch(self, x, y, batch_size, rnd=None, num=0):
    print('minibatch')
    """Samples a mini-batch from the episode."""
    if batch_size == -1:
      """Full batch mode"""
      return x, y
    idx = np.arange(x.shape[0])
    if rnd is not None:
      rnd.shuffle(idx)
      idx = idx[:batch_size]
      return x[idx], y[idx]
    else:
      length = x.shape[0]
      start = num * batch_size
      end = min((num + 1) * batch_size, length)
      return x[start:end], y[start:end]

  def monitor_b(self, sess, x_b_np, y_b_np, x_b_v_np, y_b_v_np, fdict=None):
    """Solve the few-shot classifier and monitor the progress on training and
    validation sets.

    Args:
      sess: TensorFlow session.
      x_b_np: Numpy array. Support images.
      y_b_np: Numpy array. Support labels.
      x_b_v_np: Numpy array. Query images.
      y_b_v_np: Numpy array. Query labels.
    """
    tconfig = self.config.transfer_config
    steps = tconfig.ft_optimizer_config.max_train_steps
    batch_size = tconfig.ft_optimizer_config.batch_size
    rnd = np.random.RandomState(0)
    # Re-initialize the fast weights.
    self.reset_b(sess)
    if fdict is None:
      fdict = {}
    if batch_size == -1:
      fdict[self.inputs_b] = x_b_np
      fdict[self.labels_b] = y_b_np
    fdict[self.inputs_b_v] = x_b_v_np
    fdict[self.labels_b_v] = y_b_v_np

    cost_b_list = np.zeros([steps])
    acc_b_list = np.zeros([steps])
    acc_b_v_list = np.zeros([steps])

    # Run 1st order.
    if tconfig.ft_optimizer_config.optimizer in ['adam', 'sgd', 'mom']:
      it = six.moves.xrange(steps)
      it = tqdm(it, ncols=0, desc='solve b')
      cost_b = 0.0
      for num in it:
        if batch_size == -1:
          # Use full batch size.
          x_, y_ = x_b_np, y_b_np
        else:
          # Use mini-batch.
          assert False
          x_, y_ = self.minibatch(x_b_np, y_b_np, batch_size, rnd=rnd)
          fdict[self.inputs_b] = x_
          fdict[self.labels_b] = y_
        cost_b, acc_b_tr, acc_b_v, _ = sess.run(
            [self.cost_b, self.acc_b_tr, self.acc_b_v, self._train_op_ft],
            feed_dict=fdict)
        cost_b_list[num] = cost_b
        acc_b_list[num] = acc_b_tr
        acc_b_v_list[num] = acc_b_v
        it.set_postfix(
            cost_b='{:.3e}'.format(cost_b),
            acc_b_tr='{:.3f}'.format(acc_b_tr * 100.0),
            acc_b_v='{:.3f}'.format(acc_b_v * 100.0))
    # Run 2nd order after initial burn in.
    elif tconfig.ft_optimizer_config.optimizer in ['lbfgs']:
      # Let's use first order optimizers for now.
      assert False, 'Not supported.'
    return cost_b_list, acc_b_list, acc_b_v_list

  def solve_b(self, sess, x_b_np, y_b_np, fdict=None):
    """Solve the few-shot classifier.

    Args:
      sess: TensorFlow session.
      x_b_np: Numpy array. Support images.
      y_b_np: Numpy array. Support labels.
      fdict: Feed dict used for forward pass.
    """
    tconfig = self.config.transfer_config
    steps = tconfig.ft_optimizer_config.max_train_steps
    batch_size = tconfig.ft_optimizer_config.batch_size
    rnd = np.random.RandomState(0)
    # Re-initialize the fast weights.
    # self.reset_b(sess)
    if fdict is None:
      fdict = {}
    if batch_size == -1:
      fdict[self.inputs_b] = x_b_np
      fdict[self.labels_b] = y_b_np

      # print('solve_b : fdict')
      # for k, v in fdict.items():
      #   print(k)
        # print(v)

    cost_b = sess.run(self.cost_b, feed_dict=fdict)
    return cost_b

  def reset_b(self, sess):
    """Restores the weights to its initial state."""
    sess.run(self._init_ops)

  def eval_step_b_custom_fetch(self, sess, fetches, task_b_data):
    """Evaluate one step on task B, with custom fetch."""
    fdict = self._prerun(sess, None, task_b_data)
    _ = self.solve_b(
        sess, task_b_data.x_train, task_b_data.y_train, fdict=fdict)
    return sess.run(fetches, feed_dict=fdict)

  def eval_step_b(self, sess, task_b_data):
    """Evaluate one step on task B."""
    fdict = self._prerun(sess, None, task_b_data)
    prediction_b, y_b = sess.run([self.prediction_b_all, self.labels_b_v_all],
                                 feed_dict=fdict)

    return prediction_b, y_b

  def eval_curve_b(self, sess, task_b_data):
    """Evaluate one episode with curves."""
    fdict = self._prerun(sess, None, task_b_data)
    cost_b, acc_b, acc_b_v = self.monitor_b(
        sess,
        task_b_data.x_train,
        task_b_data.y_train,
        task_b_data.x_test,
        task_b_data.y_test,
        fdict=fdict)
    return cost_b, acc_b, acc_b_v

  def _prerun(self, sess, task_a_data, task_b_data):
    """Some steps before running."""
    fdict = self.get_fdict(task_a_data=task_a_data, task_b_data=task_b_data)
    return fdict

  def train_step(self, sess, task_a_data, task_b_data):
    """Train a single step."""
    fdict = self._prerun(sess, None, task_b_data)
    sess.run(self._update_grads_b, feed_dict=fdict)
    train_op = self.train_op_b

    cost_b_v, _ = sess.run([self.cost_b_v, train_op],
                                   feed_dict=fdict)
    return None, None, cost_b_v

  @property
  def backbone(self):
    """Backbone."""
    return self._backbone

  @property
  def hidden_b(self):
    """Hidden B."""
    return self._hidden_b

  @property
  def hidden_b_plh(self):
    """Hidden B placeholder."""
    return self._hidden_b_plh

  @property
  def save_hidden_b(self):
    """Whether to save hidden state in task B to avoid recomputing."""
    tconfig = self.config.transfer_config
    return (tconfig.finetune_layers == 'none' and
            tconfig.ft_optimizer_config.batch_size == -1)

  @property
  def prediction_b(self):
    """Prediction on task B."""
    return self._prediction_b

  @property
  def prediction_b_all(self):
    """All prediction on task B."""
    return self._prediction_b_all

  @property
  def acc_b_tr(self):
    """Accuracy on task B support."""
    return self._acc_b_tr

  @property
  def acc_b_v(self):
    """Accuracy on task B query."""
    return self._acc_b_v

  @property
  def train_op(self):
    """Overall training op."""
    return self._train_op

  @property
  def train_op_ft(self):
    """Training op for learning task B support set."""
    return self._train_op_ft

  @property
  def train_op_a(self):
    """Training op on task A."""
    return self._train_op_a

  @property
  def train_op_b(self):
    """Training op on task B."""
    return self._train_op_b

  @property
  def config(self):
    """Model config."""
    return self._config

  @property
  def learn_rate(self):
    """Learning rate."""
    return self._learn_rate

  @property
  def num_classes_a(self):
    """Number of classes on task A."""
    return self._num_classes_a

  @property
  def num_classes_b(self):
    """Number of classes on task B."""
    return self._num_classes_b
