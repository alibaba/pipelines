# Simple Standalone Arena Job

The `standalone_pipeline.py` sample creates a pipeline runs preparing dataset, ML code, training and exporting the model.

## Requirements

This sample requires to create distributed storage. In this sample, we use NFS as example.

1. You need to create `/data` in the NFS Server, and prepare `mnist data`

```
# mkdir -p /nfs
# mount -t nfs -o vers=4.0 NFS_SERVER_IP:/ /nfs
# mkdir -p /data
# cd /data
# wget https://raw.githubusercontent.com/cheyang/tensorflow-sample-code/master/data/t10k-images-idx3-ubyte.gz
# wget https://raw.githubusercontent.com/cheyang/tensorflow-sample-code/master/data/t10k-labels-idx1-ubyte.gz
# wget https://raw.githubusercontent.com/cheyang/tensorflow-sample-code/master/data/train-images-idx3-ubyte.gz
# wget https://raw.githubusercontent.com/cheyang/tensorflow-sample-code/master/data/train-labels-idx1-ubyte.gz
# cd /
# umount /nfs
```

2\. Create Persistent Volume. Moidfy `NFS_SERVER_IP` to yours.

```
# cat nfs-pv.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: tfdata
  labels:
    tfdata: nas-mnist
spec:
  persistentVolumeReclaimPolicy: Retain
  capacity:
    storage: 10Gi
  accessModes:
  - ReadWriteMany
  nfs:
    server: NFS_SERVER_IP
    path: "/data"
    
 # kubectl create -f nfs-pv.yaml
```

3\. Create Persistent Volume Claim.

```
# cat nfs-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: tfdata
  annotations:
    description: "this is the mnist demo"
    owner: Tom
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
       storage: 5Gi
  selector:
    matchLabels:
      tfdata: nas-mnist
# kubectl create -f nfs-pvc.yaml
```

> Notice: suggest to add `description` and `owner`

## Instructions

### 1. With command line

First, install the necessary Python Packages
```shell
pip3 install http://kubeflow.oss-cn-beijing.aliyuncs.com/kfp/v0.5.0/kfp.tar.gz --upgrade
pip3 install http://kubeflow.oss-cn-beijing.aliyuncs.com/kip-arena/kfp-arena-0.1.tar.gz --upgrade
```
