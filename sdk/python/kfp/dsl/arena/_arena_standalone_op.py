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

class StandaloneOp(dsl.ContainerOp):
  """Submit standalone Job."""

  # arena Image is "cheyang/arena_launcher"
  def __init__(self, name, image, command, gpus='0', cpu='0', memory='0',
          tensorboard='False', tensorboardImage='', 
          data='None', outputData='None',
          arenaImage='cheyang/arena_launcher'):

    super(StandaloneOp, self).__init__(
          name=name,
          image=arenaImage,
          command=['python','arena_launcher.py'],
          arguments=[ "--name", '%s-{{workflow.name}}' % name,
                      "--tensorboard", tensorboard,
                      "--data", data,
                      "--output-data", outputData,
                      "--image", image,
                      "--gpus", gpus,
                      "--cpu", cpu,
                      "--memory", memory,
                      "job",
                      "--", command],
          file_outputs={'train': '/output.txt'})


