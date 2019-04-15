from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import os

import tensorflow as tf
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', os.path.join(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0], 'logs/cifar_train'),
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000 + 1,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      images, labels = cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)
    
    # KJ: add variable
    i = 0

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        save_checkpoint_secs=10, # Save checkpoint by interval
        save_summaries_steps=10, # Save summary by interval
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement
            # , intra_op_parallelism_threads=1
            # , inter_op_parallelism_threads=1
            , allow_soft_placement=True
            # , device_count = {'GPU': 0}
            )) as mon_sess:
      # KJ: Create profiler
      cifar_profiler = model_analyzer.Profiler(graph=mon_sess.graph)
      run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      
      while not mon_sess.should_stop():
          if i % FLAGS.log_frequency == 0:
              mon_sess.run(train_op, options=run_options, run_metadata=run_metadata)
              cifar_profiler.add_step(i, run_metadata)
            
          else:
              mon_sess.run(train_op)            # cifar_profiler.profile_name_scope(
          i+=1
      cifar_profiler.advise(options=model_analyzer.ALL_ADVICE)            #     options=(option_builder.ProfileOptionBuilder.trainable_variables_parameter()))
      profile_graph_opts_builder = option_builder.ProfileOptionBuilder(
      option_builder.ProfileOptionBuilder.time_and_memory())
    
      profile_graph_opts_builder.with_timeline_output(timeline_file='/content/drive/My Drive/hpc-project/cifar10_profiler.json')
    
      # Last step
      profile_graph_opts_builder.with_step(20)

      #显示视图为graph view
      cifar_profiler.profile_graph(profile_graph_opts_builder.build())
    
      #统计内容为所有trainable Variable Op
      profile_scope_opt_builder = option_builder.ProfileOptionBuilder(option_builder.ProfileOptionBuilder.trainable_variables_parameter())

      #显示的嵌套深度为4
      profile_scope_opt_builder.with_max_depth(4)
      #显示字段是params，即参数
      profile_scope_opt_builder.select(['params'])
      #根据params数量进行显示结果排序
      profile_scope_opt_builder.order_by('params')
    
      #显示视图为scope view
      cifar_profiler.profile_name_scope(profile_scope_opt_builder.build())



def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
