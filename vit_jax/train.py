# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import glob
import os
import time

from clu import metric_writers

import numpy as np

import jax
import jax.numpy as jnp

import flax
import flax.optim as optim
import flax.jax_utils as flax_utils

from flax.training import checkpoints as flax_checkpoints 

import tensorflow as tf

from vit_jax import checkpoint
from vit_jax import flags
from vit_jax import hyper
from vit_jax import logging
from vit_jax import input_pipeline
from vit_jax import models
from vit_jax import momentum_clip

from jax import random  

class CosineInc:
    def __init__(self, std: float, num_epochs:int, steps_per_epoch: int, inc: int):
        self.base = std
        self.halfwavelength_steps = num_epochs * steps_per_epoch
        self.inc = inc

    def __call__(self, step):
        scale_factor = -jnp.cos(step * jnp.pi / self.halfwavelength_steps) * 0.5 + 0.5
        self.current = self.base * (scale_factor * self.inc + 1)
        return self.current

def make_update_fn(vit_fn, accum_steps):

  # Update step, replicated over all TPUs/GPUs
  @functools.partial(jax.pmap, axis_name='batch', donate_argnums=(0,))
  def update_fn(opt, t, lr, batch, update_rng, rho_max=2.5, rho_min=1.5, lr_max=3e-3, lr_min=3e-5, alpha=0.3): 
      

    # Bind the rng key to the device id (which is unique across hosts)
    # Note: This is only used for multi-host training (i.e. multiple computers
    # each with multiple accelerators).
    update_rng = jax.random.fold_in(update_rng, jax.lax.axis_index('batch'))
    update_rng, new_update_rng = jax.random.split(update_rng)

    def sigmoid_xent(*, logits, labels):
      log_p = jax.nn.log_sigmoid(logits)
      log_not_p = jax.nn.log_sigmoid(-logits)
      nll = -jnp.sum(labels * log_p + (1. - labels) * log_not_p, axis=-1)
      return jnp.mean(nll)
    """
    def cross_entropy_loss(*, logits, labels):
      logp = jax.nn.log_softmax(logits)
      return -jnp.mean(jnp.sum(logp * labels, axis=1))
    """
    def loss_fn(params, images, labels):
      with flax.nn.stochastic(update_rng):
        logits = vit_fn(params, images, train=True)
        label_smoothing = 1e-4
        num_classes = 1000
        labels = labels * (1 - label_smoothing) + label_smoothing 
      return sigmoid_xent(logits=logits, labels=labels)


    def random_vector(params, key):
      param_flat, treedef = jax.tree_flatten(params)
      param_flat = treedef.flatten_up_to(params)
      # key = random.PRNGKey(0)
      key, subkey = random.split(key)
      pr = [np.prod(np.array(p.shape[:-1])) if p.ndim>1 else np.prod(np.array(p.shape)) for p in param_flat]
      print([np.array(p.shape) for p in param_flat])
      gradient_norm = []
      std_scheduler = CosineInc(0.0005, 1, 93834, 5 - 1)
      std = std_scheduler(t)

      for p, p1 in zip(param_flat, pr):
        if p.ndim>1:
            # print(1)
            if p.ndim == 4:
                p = p.transpose((3,0,1,2))
            elif p.ndim == 3:
                p = p.transpose((2,0,1))
            else:
                p = p.transpose((1,0))
            para = std*jax.random.normal(subkey, shape=p.shape)*jnp.sqrt(jnp.sum(jnp.square(p).reshape([p.shape[0], -1]), axis=1)).reshape((p.shape[0],1)).repeat(int(p1), axis=1).reshape(p.shape)
            if p.ndim == 4:
                gradient_norm.append(para.transpose((1,2,3,0)))
            elif p.ndim == 3:
                gradient_norm.append(para.transpose((1,2,0)))
            else:
                gradient_norm.append(para.transpose((1,0)))
        else:
            # print(11)
            gradient_norm.append(std*jax.random.normal(subkey, shape=p.shape)*(jnp.sqrt(jnp.sum(jnp.square(p))).reshape((1,)).repeat(int(p1), axis=0).reshape(p.shape)+1e-16))

      randomed_param = treedef.unflatten(gradient_norm)

      return randomed_param, key

    def dual_vector(y, ):

      gradient_norm = jnp.sqrt(sum(
        [jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(y)]))
      normalized_gradient = jax.tree_map(lambda x: x / gradient_norm, y)
      return normalized_gradient, gradient_norm

    def sam_gradient(loss_fn, base_opt, inputs, targets, grad_accum_steps,
                  rho_max, rho_min, alpha, lr, lr_max, lr_min=0.0, eps=1e-12):
 
      key = update_rng

      model = base_opt.target

      perturb, key = random_vector(model, key)

      noised_model = jax.tree_multimap(lambda a, b: a + b,
                                     model, perturb)

      l_clean, g_clean = hyper.accumulate_gradient(jax.value_and_grad(loss_fn), noised_model,
                                inputs, targets, grad_accum_steps)
      g_clean_normalized, g_clean_length = dual_vector(g_clean)

      def w1(lr):
          return lr/lr_max * rho_max
      def w2(lr):
          return rho_min + (rho_max - rho_min) * (lr - lr_min) / (lr_max - lr_min)

      if lr_max == lr_min:
        sam_rho = rho_max
      else:
        sam_rho = jax.lax.cond(t<10000, w1, w2, lr)   

      g_clean_v2 = jax.tree_multimap(lambda a, b: 2.0 * a + b, g_clean, perturb) 
      
      g_clean_length_v2 = jnp.sqrt(sum(
        [jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(g_clean_v2)]))

      param_sam = jax.tree_multimap(lambda a, b: a + sam_rho * b / (g_clean_length_v2 + eps),
                                base_opt.target, g_clean_v2) 

      l_robust, g_robust = hyper.accumulate_gradient(jax.value_and_grad(loss_fn), param_sam,
                                inputs, targets, grad_accum_steps)

      return l_clean, g_robust, sam_rho, g_clean_length     

    
    l, g, sam_rho, g_clean_residual_length = sam_gradient(loss_fn=loss_fn, base_opt=opt, inputs=batch['image'], targets=batch['label'],
        grad_accum_steps=accum_steps, rho_max=rho_max, rho_min=rho_min, alpha=alpha, lr=lr, lr_max=lr_max, lr_min=lr_min)

    
    g = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch'), g)
      
    grad_clip_norm = 1.0
    gradients, _ = jax.tree_flatten(g)
    g_l2 = jnp.sqrt(sum([jnp.vdot(p, p) for p in gradients]))
    g_factor = jnp.minimum(1.0, grad_clip_norm / g_l2)
    g = jax.tree_map(lambda p: g_factor * p, g)



    opt = opt.apply_gradient(g, learning_rate=lr)
    return opt, l, new_update_rng, sam_rho, g_clean_residual_length

  return update_fn


def main(args):
  logdir = os.path.join(args.logdir, args.name)
  logger = logging.setup_logger(logdir)
  logger.info(args)

  logger.info(f'Available devices: {jax.devices()}')

  # Setup input pipeline
  dataset_info = input_pipeline.get_dataset_info(args.dataset, 'train')

  ds_train = input_pipeline.get_data(
      dataset=args.dataset,
      mode='train',
      repeats=None,
      mixup_alpha=args.mixup_alpha,
      batch_size=args.batch,
      shuffle_buffer=args.shuffle_buffer,
      tfds_data_dir=args.tfds_data_dir,
      tfds_manual_dir=args.tfds_manual_dir)
  batch = next(iter(ds_train))
  logger.info(ds_train)
  ds_test = input_pipeline.get_data(
      dataset=args.dataset,
      mode='test',
      repeats=1,
      batch_size=args.batch_eval,
      tfds_data_dir=args.tfds_data_dir,
      tfds_manual_dir=args.tfds_manual_dir)
  logger.info(ds_test)

  # Build VisionTransformer architecture
  model = models.KNOWN_MODELS[args.model]
  VisionTransformer = model.partial(num_classes=dataset_info['num_classes'])
  _, params = VisionTransformer.init_by_shape(
      jax.random.PRNGKey(0),
      # Discard the "num_local_devices" dimension for initialization.
      [(batch['image'].shape[1:], batch['image'].dtype.name)])

  params["head"]["bias"] = jnp.full_like(params["head"]["bias"], -10.)
  # pmap replicates the models over all TPUs/GPUs
  vit_fn_repl = jax.pmap(VisionTransformer.call)
  update_fn_repl = make_update_fn(VisionTransformer.call, args.accum_steps)


  # Create optimizer and replicate it over all TPUs/GPUs
  opt = flax.optim.Adam(beta1=0.9, beta2=0.999, weight_decay=0.3).create(params) 

  initial_step = 1  

  if initial_step != 1:
    initial_step = initial_step + 1 

  opt_repl = flax_utils.replicate(opt)

  # Delete references to the objects that are not needed anymore
  del opt
  del params

  def copyfiles(paths):
    """Small helper to copy files to args.copy_to using tf.io.gfile."""
    if not args.copy_to:
      return
    for path in paths:
      to_path = os.path.join(args.copy_to, args.name, os.path.basename(path))
      tf.io.gfile.makedirs(os.path.dirname(to_path))
      tf.io.gfile.copy(path, to_path, overwrite=True)
      logger.info(f'Copied {path} to {to_path}.')

  total_steps = args.total_steps or (
      input_pipeline.DATASET_PRESETS[args.dataset]['total_steps'])

  # Prepare the learning-rate and pre-fetch it to device to avoid delays.
  lr_fn = hyper.create_learning_rate_schedule(total_steps, args.base_lr,
                                              args.decay_type,
                                              args.warmup_steps)
  lr_iter = hyper.lr_prefetch_iter(lr_fn, initial_step - 1, total_steps)
  update_rngs = jax.random.split(
      jax.random.PRNGKey(0), jax.local_device_count())

  # Run training loop
  writer = metric_writers.create_default_writer(logdir, asynchronous=False)
  writer.write_hparams({k: v for k, v in vars(args).items() if v is not None})
  logger.info('Starting training loop; initial compile can take a while...')
  t0 = time.time()

  lstep = initial_step

  for step, batch, lr_repl in zip(
      range(initial_step, total_steps + 1),
      input_pipeline.prefetch(ds_train, args.prefetch), lr_iter):

    opt_repl, loss_repl, update_rngs, sam_rho, g_clean_residual_length = update_fn_repl(
        opt_repl, flax.jax_utils.replicate(step), lr_repl, batch, update_rngs)

    if step == initial_step:
      logger.info(f'First step took {time.time() - t0:.1f} seconds.')
      t0 = time.time()
    if args.progress_every and step % args.progress_every == 0:
      writer.write_scalars(step, dict(train_loss=float(loss_repl[0])))
      done = step / total_steps
      logger.info(f'Step: {step}/{total_steps} {100*done:.1f}%, '
                  f'ETA: {(time.time()-t0)/done*(1-done)/3600:.2f}h')
      copyfiles(glob.glob(f'{logdir}/*'))

    # Run eval step
    if ((args.eval_every and step % args.eval_every == 0) or
        (step == total_steps)):

      accuracy_test = np.mean([
          c for batch in input_pipeline.prefetch(ds_test, args.prefetch)
          for c in (
              np.argmax(vit_fn_repl(opt_repl.target, batch['image']),
                        axis=2) == np.argmax(batch['label'], axis=2)).ravel()
      ])

      lr = float(lr_repl[0])
      logger.info(f'Step: {step} '
                  f'Learning rate: {lr:.7f}, '
                  f'Test accuracy: {accuracy_test:0.5f}')
      writer.write_scalars(step, dict(accuracy_test=accuracy_test, lr=lr, sam_rho=sam_rho[0], g_clean_residual_length=g_clean_residual_length[0])) 
      copyfiles(glob.glob(f'{logdir}/*'))

    if (step == 93834 or step%5000 == 0): 

      # checkpoint.save(flax_utils.unreplicate(opt_repl.target), args.output)
      checkpoint_path = flax_checkpoints.save_checkpoint(
          args.output, (flax.jax_utils.unreplicate(opt_repl), step), step)
      logger.info('Stored checkpoint at step %d to "%s"', step,
                   checkpoint_path)
      # logger.info(f'Stored fine tuned checkpoint to {args.output}')
      # copyfiles([args.output])

if __name__ == '__main__':
  # Make sure tf does not allocate gpu memory.
  tf.config.experimental.set_visible_devices([], 'GPU')

  parser = flags.argparser(models.KNOWN_MODELS.keys(),
                           input_pipeline.DATASET_PRESETS.keys())

  main(parser.parse_args())
