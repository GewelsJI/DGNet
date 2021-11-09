echo '>>> Create Environment'
conda create -n DGNet python=3.6
source activate DGNet
pip install -r requirements.txt

echo '>>> Start Inference'
python inference.py
