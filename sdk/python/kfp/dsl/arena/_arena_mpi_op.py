#!/usr/bin/env python3
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import kfp.dsl as dsl
import datetime
import logging

# def arena_submit_standalone_job_op(name, image, gpus: int, ):

class MPIOp(dsl.ContainerOp):
  """Submit MPI Job."""

  def __init__(self, name="mpi", image="", workers=1, gpus=0, cpu=0,memory=0,
          tensorboard=False,tensorboardImage="tensorflow/tensorflow:1.12.0",data=[],arenaImage="cheyang/arena_launcher",outputData="",command=""):

    if workers < 1:
          raise ValueError("Invalid workers %d." % workers)

    if not isinstance(data, (list)):
        raise ValueError("Invalid data {0} with type {1}. should be like [data:/test]".format(data, type(data)))

    commandArray = [
    'arena', 'submit', 'mpijob',
    '--name={0}'.format(name),
    '--workers={0}'.format(workers),
    '--image={0}'.format(image),
  ]

    if gpus > 0:
        commandArray.append("--gpus={0}".format(gpus))

    if cpu > 0:
        commandArray.append("--cpu={0}".format(cpu))

    if memory >0:
        commandArray.append("--memory={0}".format(memory))

    if tensorboardImage != "tensorflow/tensorflow:1.12.0":
        commandArray.append("--tensorboardImage={0}".format(tensorboardImage))    

    if tensorboard:
        commandArray.append("--tensorboard")

    if len(data) > 0:
        if len(outputData) > 0:
            data= data + list(set([outputData]) - set(data))

        for i in range(len(data)):
            commandArray.append("--data={0}".format(data[i]))

    if len(command) == 0:
        raise ValueError("You must specify command")

    commandArray.append('"{0}"'.format(command))

    arenaCommand = ' '.join(commandArray)

    arguments = [
    '--name={0}'.format(name),
    '--job-type=mpijob',
    '--tensorboard={0}'.format(str(tensorboard).lower()),
  ]

    if len(outputData) > 0:
          arguments.append("--output={0}".format(outputData))

    arguments.append("--")
    arguments.append(arenaCommand)

    super(MPIOp, self).__init__(
          name=name,
          image=arenaImage,
          command=['python','arena_launcher.py'],
          arguments=arguments,
          file_outputs = {'train': '/output.txt'})


