nohup python MTMCT.py -train=True -batch=2 -epoch=600 -gpu=3 -pretrain_type=b > logb1.log &
nohup python MTMCT.py -train=True -batch=4 -epoch=600 -gpu=3 -pretrain_type=b > logb2.log &
nohup python MTMCT.py -train=True -batch=8 -epoch=600 -gpu=3 -pretrain_type=b > logb3.log &
nohup python MTMCT.py -train=True -batch=16 -epoch=600 -gpu=3 -pretrain_type=b > logb4.log &
nohup python MTMCT.py -train=True -batch=32 -epoch=600 -gpu=3 -pretrain_type=b > logb5.log &