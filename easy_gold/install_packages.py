# build.pyによって依存パッケージのinstall処理も行いたい為、pythonスクリプト上で書いている。
import subprocess


subprocess.run(["pip", "install", "gokart==1.2.3"])  # https://pypi.org/project/gokart/#history
