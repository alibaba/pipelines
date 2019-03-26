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
import arena
import kfp.dsl as dsl


@dsl.pipeline(
  name='pipeline to run jobs',
  description='shows how to run pipeline jobs.'
)
def jobpipeline():
  """A pipeline for end to end machine learning workflow."""
  data="user-susan:/training"
  gpus="1"

  # 1. prepare data
  prepare_data = arena.StandaloneOp(
    name="prepare-data",
		image="byrnedo/alpine-curl",
    data=data,
		command="mkdir -p /training/dataset/mnist && \
  cd /training/dataset/mnist && \
  curl -O https://code.aliyun.com/xiaozhou/tensorflow-sample-code/raw/master/data/t10k-images-idx3-ubyte.gz && \
  curl -O https://code.aliyun.com/xiaozhou/tensorflow-sample-code/raw/master/data/t10k-labels-idx1-ubyte.gz && \
  curl -O https://code.aliyun.com/xiaozhou/tensorflow-sample-code/raw/master/data/train-images-idx3-ubyte.gz && \
  curl -O https://code.aliyun.com/xiaozhou/tensorflow-sample-code/raw/master/data/train-labels-idx1-ubyte.gz")
  # 2. prepare source code
  prepare_code = arena.StandaloneOp(
    name="source-code",
    image="alpine/git",
    data=data,
    command="mkdir -p /training/models/ && \
  cd /training/models/ && \
  if [ ! -d /training/models/tensorflow-sample-code ]; then git clone https://code.aliyun.com/xiaozhou/tensorflow-sample-code.git; else echo no need download;fi")

  # 3. train the models
  train = arena.StandaloneOp(
    name="train",
    image="tensorflow/tensorflow:1.11.0-gpu-py3",
    gpus=gpus,
    data=data,
    command="echo %s;echo %s;python /training/models/tensorflow-sample-code/tfjob/docker/mnist/main.py --max_steps 500 --data_dir /training/dataset/mnist --log_dir /training/output/mnist" % (prepare_data.output, prepare_code.output))
  # 4. export the model
  export_model = arena.StandaloneOp(
    name="export-model",
    image="tensorflow/tensorflow:1.11.0-py3",
    data=data,
    command="echo %s;python /training/models/tensorflow-sample-code/tfjob/docker/mnist/export_model.py --model_version=1 --checkpoint_path=/training/output/mnist /training/output/models" % train.output)

if __name__ == '__main__':
  EXPERIMENT_NAME="standalonejob"
  import kfp.compiler as compiler
  compiler.Compiler().compile(jobpipeline, __file__ + '.tar.gz')
  client = kfp.Client()
  exp = client.create_experiment(name=EXPERIMENT_NAME)
  run = client.run_pipeline(exp.id, 'jobpipeline-1', 'jobpipeline.py.tar.gz')
