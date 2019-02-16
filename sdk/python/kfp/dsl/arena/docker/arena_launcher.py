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

"""
Usage:
python arena_launcher.py
    --name=tf-test
    --tensorboard=true
    mpijob
    --gpus=1
    --workers=2
    --image=registry.cn-hangzhou.aliyuncs.com/tensorflow-samples/horovod:0.13.11-tf1.10.0-torch0.4.0-py3.5
    --
    mpirun python /benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model resnet101 --batch_size 64     --variable_update horovod --train_dir=/training_logs --summary_verbosity=3 --save_summaries_steps=10
"""
# TODO: Add unit/integration tests

import argparse
import datetime
import json
import os
import sys
import logging
import requests
import subprocess
import six
import time
import yaml

def setup_custom_logging():
  logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', 
                      level=logging.INFO,
                      datefmt='%Y-%m-%d %H:%M:%S')

def _submit_job(command):
  logging.info("command: {0}".format(command))
  try:
    output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
    result = output.decode()
  except subprocess.CalledProcessError as exc:
    print("Status : FAIL", exc.returncode, exc.output)
    sys.exit(-1)
  logging.info('Submit Job: %s.' % result)

def _is_active_status(status):
    logging.info("status: {0}".format(status))
    return status == 'PENDING' or status == 'RUNNING'

def _wait_job_done(name, job_type, timeout):
  end_time = datetime.datetime.now() + timeout
  status = _get_job_status(name, job_type)
  while _is_active_status(status):
    if datetime.datetime.now() > end_time:
      timeoutMsg = "Timeout waiting for job {0} with job type {1} completing.".format(name ,job_type)
      logging.error(timeoutMsg)
      raise Exception(timeoutMsg)
    time.sleep(3)
    status = _get_job_status(name, job_type)
  logging.info("job {0} with type {1} status is {2}".format(name, job_type, status))

def _get_job_status(name, job_type):
  get_cmd = "arena get %s --type %s | grep -i STATUS:|awk -F: '{print $NF}'" % (name, job_type)
  status = ""
  try:
    output=subprocess.check_output(get_cmd, stderr=subprocess.STDOUT, shell=True)
    status = output.decode()
    status = status.strip()
  except subprocess.CalledProcessError as e:
    logging.warning("Failed to get job status due to" + e)

  return status

def _get_tensorboard_url(name, job_type):
  get_cmd = "arena get %s --type %s | tail -1" % (name, job_type)
  url = "N/A"
  try:
    output = subprocess.check_output(get_cmd, stderr=subprocess.STDOUT, shell=True)
    url = output.decode()
  except subprocess.CalledProcessError as e:
    logging.warning("Failed to get job status due to" + e)

  return url

def generate_mpjob_command(args):
    name = args.name
    workers = args.workers
    gpus = args.gpus
    cpu = args.cpu
    memory = args.memory
    tensorboard = args.tensorboard
    image = args.image
    output_data = args.output_data
    dataList = args.dataList

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
      dataList = data.split(",")
      if len(output_data) > 0:
        dataList = dataList + list(set([output_data]) - set(dataList))

        for i in range(len(dataList)):
            commandArray.append("--data={0}".format(dataList[i]))

    return commandArray, "mpijob"

def main(argv=None):
  setup_custom_logging()
  parser = argparse.ArgumentParser(description='Arena launcher')
  parser.add_argument('--name', type=str,
                      help='The job name to specify.',default=None)
  parser.add_argument('--job-type', type=str,
                      help='The job type to specify, tfjob or mpijob',default=None)
  parser.add_argument('--tensorboard', type=bool, default=False)
  parser.add_argument('--timeout-minutes', type=int,
                      default=30,
                      help='Time in minutes to wait for the Job submitted by arena to complete')
  # parser.add_argument('--command', type=str)
  parser.add_argument('--output-dir', type=str)
  parser.add_argument('--output-data', type=str)
  parser.add_argument('--data', type=str)
  subparsers = parser.add_subparsers(help='arena sub-command help')

  #create the parser for the 'mpijob' command
  parser_mpi = subparsers.add_parser('mpijob', help='mpijob help')
  parser_mpi.add_argument('--image', type=str)
  parser_mpi.add_argument('--workers', type=int, default=2)
  parser_mpi.add_argument('--gpus', type=int, default=0)
  parser_mpi.add_argument('--cpus', type=int, default=1)
  parser_mpi.add_argument('--memory', type=int, default=1)
  parser_mpi.set_defaults(func=generate_mpjob_command)


  import sys
  all_args = sys.argv[1:]
  separator_idx = all_args.index('--')
  launcher_args = all_args[:separator_idx]
  remaining_args = all_args[separator_idx + 1:]
  
  args = parser.parse_args(launcher_args)
  commandArray, jobtype = args.func(args)

  args_dict = vars(args)
  if args.name is None:
    logging.error("Please specify the name")
    sys.exit(-1)
  if len(remaining_args) == 0:
    logging.error("Please specify the command.")
    sys.exit(-1)

  name = args.name
  fullname = name + datetime.datetime.now().strftime("%Y%M%d%H%M%S")
  timeout_minutes = args_dict.pop('timeout_minutes')


  enableTensorboard = args_dict.pop('tensorboard')

  commandArray.append(remaining_args[0])
  command = ' '.join(commandArray)
  
  command=command.replace("--name={0}".format(name),"--name={0}".format(fullname))
  
  logging.info('Start training.')
  
  _submit_job(command)
  
  tensorboardUrl = "N/A"
  if enableTensorboard:
      tensorboardUrl = _get_tensorboard_url(fullname, job_type)
  output = "N/A"

  if args.output_dir:
    # Create metadata.json file for visualization.
    output = args.output_dir
    

  if args.output_data:
    output = args.output_data

  metadata = {
      'outputs' : [{
        'jobname': fullname, 
        'tensorboard': tensorboardUrl,
        'output': output,
      }]
    }

  with open('/mlpipeline-ui-metadata.json', 'w') as f:
      json.dump(metadata, f)

  
  succ = True

  # wait for job done
  _wait_job_done(fullname, job_type, datetime.timedelta(minutes=timeout_minutes))
  
  status = _get_job_status(fullname, job_type)

  if status == "SUCCEEDED":
    logging.info("Training Job {0} success.".format(fullname))
  elif status == "FAILED":
    logging.error("Training Job {0} fail.".format(fullname))
  else:
    logging.error("Training Job {0}'s status {1}".format(fullname, status))

  with open('/output.txt', 'w') as f:
    f.write(output)


if __name__== "__main__":
  main()
