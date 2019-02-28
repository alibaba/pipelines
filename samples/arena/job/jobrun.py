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


@dsl.pipeline(
  name='pipeline to run jobs',
  description='shows how to run pipeline jobs.'
)
def jobpipeline():

  data="training-data:/training"
  gpus="1"

  prepare_data = arena.JobOp(
    name="prepare-data",
		image="byrnedo/alpine-curl",
    data=data,
		command="mkdir -p /training/dataset/mnist && \
  cd /training/dataset/mnist && \
  curl -O https://code.aliyun.com/xiaozhou/tensorflow-sample-code/raw/master/data/t10k-images-idx3-ubyte.gz && \
  curl -O https://code.aliyun.com/xiaozhou/tensorflow-sample-code/raw/master/data/t10k-labels-idx1-ubyte.gz && \
  curl -O https://code.aliyun.com/xiaozhou/tensorflow-sample-code/raw/master/data/train-images-idx3-ubyte.gz && \
  curl -O https://code.aliyun.com/xiaozhou/tensorflow-sample-code/raw/master/data/train-labels-idx1-ubyte.gz")
  train = arena.JobOp(
    name="train",
    image="tensorflow/tensorflow:1.11.0-gpu-py3",
    gpus=gpus,
    data=data,
    command="cat %s;python /training/models/tensorflow-sample-code/tfjob/docker/mnist/main.py --max_steps 10000 --data_dir /training/dataset/mnist --log_dir /training/output/mnist" % prepare_data.output)
  export = arena.JobOp(
    name="export",
    image="tensorflow/tensorflow:1.11.0-py3",
    data=data,
    command="cat %s;" % train.output)

if __name__ == '__main__':
  # EXPERIMENT_NAME="tf_cnn_benchmarks"
  import kfp.compiler as compiler
  compiler.Compiler().compile(jobpipeline, __file__ + '.tar.gz')
  client = kfp.Client()
  # exp = client.create_experiment(name=EXPERIMENT_NAME)
  id = '55f3d3b2-f230-41f4-936e-a2ec8c6842d6'
  run = client.run_pipeline(id, 'jobpipeline', 'jobpipeline.py.tar.gz')
