#mkdir /home/neuralEntropy
cp -r /mnt /home/neuralEntropy

cd /home/neuralEntropy/
mkdir models
/usr/bin/python3 -m pip install --upgrade pip
pip install -r requirements.txt

sh rs_test.sh