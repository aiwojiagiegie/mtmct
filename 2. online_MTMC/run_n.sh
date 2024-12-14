nohup python MTMCT.py -train=True -batch=2 -epoch=600 -gpu=1 -pretrain_type=n > logn1.log &
nohup python MTMCT.py -train=True -batch=4 -epoch=600 -gpu=1 -pretrain_type=n > logn2.log &
nohup python MTMCT.py -train=True -batch=8 -epoch=600 -gpu=1 -pretrain_type=n > logn3.log &
nohup python MTMCT.py -train=True -batch=16 -epoch=600 -gpu=1 -pretrain_type=n > logn4.log &
nohup python MTMCT.py -train=True -batch=32 -epoch=600 -gpu=1 -pretrain_type=n > logn5.log &