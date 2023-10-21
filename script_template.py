import gzip
import base64
import os
from pathlib import Path
from typing import Dict


# this is base64 encoded source code
file_data: Dict = {file_data}


for path, encoded in file_data.items():
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.decompress(base64.b64decode(encoded)))


def run(command):
    os.system("export PYTHONPATH=${PYTHONPATH}:/kaggle/working && " + command)


print("hoge1")
run("pip install --quiet -r requirements.txt")
print("hoge2")
# run("python setup.py develop --install-dir /kaggle/working")
print("hoge3")
run("cd my_recommender_experiments")
# run("python easy_gold/main.py")
run("python main.py recommender_experiments.Main --local-scheduler")
print("hoge4")
