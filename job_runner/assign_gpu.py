import xml.etree.ElementTree as ET
import os
import subprocess


def assign_gpu():
    device_ids = os.environ['CUDA_VISIBLE_DEVICES']
    device_ids = device_ids.split(',')
    slurm_localid = int(os.environ['SLURM_LOCALID'])
    assert slurm_localid < len(device_ids)
    cuda_id = int(device_ids[slurm_localid])

    out = subprocess.check_output(['nvidia-smi', '-q', '--xml-format'])
    tree = ET.fromstring(out)
    gpus = tree.findall('gpu')
    gpu = gpus[cuda_id]
    dev_id = gpu.find('minor_number').text

    print(f"export EGL_VISIBLE_DEVICES={dev_id}")
    print(f"export CUDA_VISIBLE_DEVICES={cuda_id}")


if __name__ == '__main__':
    assign_gpu()
