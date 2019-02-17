#!/usr/bin/env python3
# Copyright 2018 Google LLC
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

import kfp
import kfp.dsl.arena as arena
import kfp.dsl as dsl

class PrintOp(dsl.ContainerOp):
  """Print a message."""

  def __init__(self, msg):
    super(PrintOp, self).__init__(
      name='Print',
      image='alpine:3.6',
      command=['echo', msg],
  )
    


@dsl.pipeline(
  name='pipeline mpirun',
  description='shows how to use mpirun.'
)
def mpirun(name=dsl.PipelineParam(name='name',
                                  value='mpirun'),
    image=dsl.PipelineParam(name='image',
                            value='registry.cn-hangzhou.aliyuncs.com/tensorflow-samples/horovod:0.13.11-tf1.10.0-torch0.4.0-py3.5'),
    workers=dsl.PipelineParam(name='workers',
                                  value='2'),
    gpus=dsl.PipelineParam(name='gpus',
                           value='1'),
    cpu=dsl.PipelineParam(name='cpu',
                           value='0'),
    memory=dsl.PipelineParam(name='memory',
                           value='0'),
    rdma=dsl.PipelineParam(name='rdma',
                           value='False'),
    data=dsl.PipelineParam(name='data',
                           value=''),
    outputData=dsl.PipelineParam(name='outputData',
                           value=''),
    tensorboard=dsl.PipelineParam(name='tensorboard',
                                  value='False'),
    arenaImage=dsl.PipelineParam(name='arenaImage',
                                  value='cheyang/arena_launcher'),
    command=dsl.PipelineParam(name='command',
            value='mpirun python /benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model resnet101 --batch_size 64 --variable_update horovod --train_dir=/training_logs --summary_verbosity=3 --save_summaries_steps=10')):


  mpi = arena.MPIOp(
		image=image,
		tensorboard=tensorboard,
		workers=workers,
		gpus=gpus,
    cpu=cpu,
    rdma=rdma,
    memory=memory,
    data=data,
		command=command)
  PrintOp('mpioutput %s !' % mpi.output)


if __name__ == '__main__':
  EXPERIMENT_NAME="tf_cnn_benchmarks"
  import kfp.compiler as compiler
  compiler.Compiler().compile(mpirun, __file__ + '.tar.gz')
  # client = kfp.Client()
  # exp = client.create_experiment(name=EXPERIMENT_NAME)
  # run = client.run_pipeline(exp.id, 'mpirun', 'mpirun.py.tar.gz')
