nohup python MTMCT.py -train=True -batch=2 -epoch=600 -gpu=1 -pretrain_type=s > logs1.log &
nohup python MTMCT.py -train=True -batch=4 -epoch=600 -gpu=1 -pretrain_type=s > logs2.log &
nohup python MTMCT.py -train=True -batch=8 -epoch=600 -gpu=1 -pretrain_type=s > logs3.log &
nohup python MTMCT.py -train=True -batch=16 -epoch=600 -gpu=1 -pretrain_type=s > logs4.log &
nohup python MTMCT.py -train=True -batch=32 -epoch=600 -gpu=1 -pretrain_type=s > logs5.log &