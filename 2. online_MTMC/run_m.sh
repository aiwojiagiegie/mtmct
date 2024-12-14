nohup python MTMCT.py -train=True -batch=2 -epoch=600 -gpu=0 -pretrain_type=m > logm1.log &
nohup python MTMCT.py -train=True -batch=4 -epoch=600 -gpu=0 -pretrain_type=m > logm2.log &
nohup python MTMCT.py -train=True -batch=8 -epoch=600 -gpu=0 -pretrain_type=m > logm3.log &
nohup python MTMCT.py -train=True -batch=16 -epoch=600 -gpu=0 -pretrain_type=m > logm4.log &
nohup python MTMCT.py -train=True -batch=32 -epoch=600 -gpu=0 -pretrain_type=m > logm5.log &