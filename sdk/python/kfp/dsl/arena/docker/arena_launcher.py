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
from subprocess import Popen,PIPE
from shlex import split

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

def _is_pending_status(status):
    logging.info("status: {0}".format(status))
    return status == 'PENDING'

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

def _wait_job_running(name, job_type, timeout):
  end_time = datetime.datetime.now() + timeout
  status = _get_job_status(name, job_type)
  while _is_pending_status(status):
    if datetime.datetime.now() > end_time:
      timeoutMsg = "Timeout waiting for job {0} with job type {1} completing.".format(name ,job_type)
      logging.error(timeoutMsg)
      raise Exception(timeoutMsg)
    time.sleep(3)
    status = _get_job_status(name, job_type)
  logging.info("job {0} with type {1} status is {2}".format(name, job_type, status))

def _job_logging(name, job_type):
  logging_cmd = "arena logs -f %s" % (name)
  process = Popen(split(logging_cmd), stdout = PIPE, stderr = PIPE, encoding='utf8')
  while True:
    output = process.stdout.readline()
  if output == "" and process.poll() is not None:
    break
  if output:
    print("", output.strip())
  rc = process.poll()
  return rc



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

# 

# Generate standalone job
def generate_job_command(args):
    name = args.name
    gpus = args.gpus
    cpu = args.cpu
    memory = args.memory
    tensorboard = args.tensorboard
    image = args.image
    output_data = args.output_data
    data = args.data
    tensorboard_image = args.tensorboard_image
    tensorboard = str2bool(args.tensorboard)
    log_dir = args.log_dir

    commandArray = [
    'arena', 'submit', 'tfjob',
    '--name={0}'.format(name),
    '--image={0}'.format(image),
    ]

    if gpus > 0:
        commandArray.append("--gpus={0}".format(gpus))

    if cpu > 0:
        commandArray.append("--cpu={0}".format(cpu))

    if memory >0:
        commandArray.append("--memory={0}".format(memory))

    if tensorboard_image != "tensorflow/tensorflow:1.12.0":
        commandArray.append("--tensorboardImage={0}".format(tensorboard_image))    

    if tensorboard:
        commandArray.append("--tensorboard")

    if os.path.isdir(args.log_dir):  
        commandArray.append("--logdir={0}".format(args.log_dir))
    else:
        logging.info("skip log dir :{0}".format(args.log_dir))

    if len(data) > 0 and data != 'None':
      dataList = data.split(",")
      if len(output_data) > 0 and data != 'None':
        dataList = dataList + list(set([output_data]) - set(dataList))

        for i in range(len(dataList)):
          if dataList[i] != "None":
            commandArray.append("--data={0}".format(dataList[i]))

    return commandArray, "tfjob"

# Generate mpi job
def generate_mpjob_command(args):
    name = args.name
    workers = args.workers
    gpus = args.gpus
    cpu = args.cpu
    memory = args.memory
    tensorboard = args.tensorboard
    image = args.image
    output_data = args.output_data
    data = args.data
    tensorboard_image = args.tensorboard_image
    tensorboard = str2bool(args.tensorboard)
    rdma = str2bool(args.tensorboard)
    log_dir = args.log_dir

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

    if tensorboard_image != "tensorflow/tensorflow:1.12.0":
        commandArray.append("--tensorboardImage={0}".format(tensorboard_image))    

    if tensorboard:
        commandArray.append("--tensorboard")

    if rdma:
        commandArray.append("--rdma")

    if os.path.isdir(args.log_dir):  
        commandArray.append("--logdir={0}".format(args.log_dir))
    else:
        logging.info("skip log dir :{0}".format(args.log_dir))

    if len(data) > 0 and data != 'None':
      dataList = data.split(",")
      if len(output_data) > 0 and data != 'None':
        dataList = dataList + list(set([output_data]) - set(dataList))

        for i in range(len(dataList)):
          if len(output_data) > 0 and data != 'None':
            commandArray.append("--data={0}".format(dataList[i]))

    return commandArray, "mpijob"

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")


def main(argv=None):
  setup_custom_logging()
  import sys
  all_args = sys.argv[1:]
  logging.info("args: {0}".format(' '.join(sys.argv)))
  parser = argparse.ArgumentParser(description='Arena launcher')
  parser.add_argument('--name', type=str,
                      help='The job name to specify.',default=None)
  parser.add_argument('--tensorboard', type=str, default="False")
  parser.add_argument('--rdma', type=str, default="False")
  parser.add_argument('--tensorboard-image', type=str, default='tensorflow/tensorflow:1.12.0')
  parser.add_argument('--timeout-minutes', type=int,
                      default=30,
                      help='Time in minutes to wait for the Job submitted by arena to complete')
  # parser.add_argument('--command', type=str)
  parser.add_argument('--output-dir', type=str, default='')
  parser.add_argument('--output-data', type=str, default='None')
  parser.add_argument('--log-dir', type=str, default='')
  parser.add_argument('--data', type=str, default='None')
  parser.add_argument('--image', type=str)
  parser.add_argument('--gpus', type=int, default=0)
  parser.add_argument('--cpu', type=int, default=0)
  parser.add_argument('--memory', type=int, default=0)
  parser.add_argument('--workers', type=int, default=2)
  subparsers = parser.add_subparsers(help='arena sub-command help')

  #create the parser for the 'mpijob' command
  parser_mpi = subparsers.add_parser('mpijob', help='mpijob help')
  parser_mpi.set_defaults(func=generate_mpjob_command)

  #create the parser for the 'job' command
  parser_job = subparsers.add_parser('job', help='job help')
  parser_job.set_defaults(func=generate_job_command)


  separator_idx = all_args.index('--')
  launcher_args = all_args[:separator_idx]
  remaining_args = all_args[separator_idx + 1:]
  
  args = parser.parse_args(launcher_args)
  commandArray, job_type = args.func(args)

  args_dict = vars(args)
  if args.name is None:
    logging.error("Please specify the name")
    sys.exit(-1)
  if len(remaining_args) == 0:
    logging.error("Please specify the command.")
    sys.exit(-1)

  internalCommand = ' '.join(remaining_args)

  name = args.name
  fullname = name + datetime.datetime.now().strftime("%Y%M%d%H%M%S")
  timeout_minutes = args_dict.pop('timeout_minutes')


  enableTensorboard = str2bool(args.tensorboard)

  commandArray.append('"{0}"'.format(internalCommand))
  command = ' '.join(commandArray)
  
  command=command.replace("--name={0}".format(name),"--name={0}".format(fullname))
  
  logging.info('Start training {0}.'.format(command))
  
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
  # _wait_job_done(fullname, job_type, datetime.timedelta(minutes=timeout_minutes))
  _wait_job_running(fullname, job_type, datetime.timedelta(minutes=timeout_minutes))

  rc = _job_logging(fullname, job_type)
  logging.info("rc: {0}", rc)
  
  status = _get_job_status(fullname, job_type)

  if status == "SUCCEEDED":
    logging.info("Training Job {0} success.".format(fullname))
  elif status == "FAILED":
    logging.error("Training Job {0} fail.".format(fullname))
    sys.exit(-1)
  else:
    logging.error("Training Job {0}'s status {1}".format(fullname, status))
    sys.exit(-1)

  with open('/output.txt', 'w') as f:
    f.write(output)


if __name__== "__main__":
  main()
