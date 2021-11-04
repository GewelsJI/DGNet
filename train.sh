_EXPID='V4_Small_Polyp_32BS'
_GPUID='1'

echo '>>> mkdir'
mkdir -p ./snapshot/Exp${_EXPID}

echo '>>> save train script with configurations'
cp ./train.sh ./snapshot/Exp${_EXPID}/train.sh
cp ./MyTrain.py ./snapshot/Exp${_EXPID}/MyTrain_Exp${_EXPID}.py

echo '>>> strat training'
python MyTrain.py --gpu_id=${_GPUID} --save_path=./snapshot/Exp${_EXPID}/

# echo '>>> start evaluation'
# python MyEval.py --gpu_id=${_GPUID} --snap_path=./snapshot/Exp${_EXPID}/Net_epoch_best.pth
