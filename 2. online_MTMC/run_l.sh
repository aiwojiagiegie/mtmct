nohup python MTMCT.py -train=True -batch=2 -epoch=600 -gpu=2 -pretrain_type=l > logl1.log &
nohup python MTMCT.py -train=True -batch=4 -epoch=600 -gpu=2 -pretrain_type=l > logl2.log &
nohup python MTMCT.py -train=True -batch=8 -epoch=600 -gpu=2 -pretrain_type=l > logl3.log &
nohup python MTMCT.py -train=True -batch=16 -epoch=600 -gpu=2 -pretrain_type=l > logl4.log &
nohup python MTMCT.py -train=True -batch=32 -epoch=600 -gpu=2 -pretrain_type=l > logl5.log &