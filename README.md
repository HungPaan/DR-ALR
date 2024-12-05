# Doubly Robust Learning with Adaptive Learning Rates for Debiasing Post-Click Conversion Rate Estimation Under Sparse Data
## Introduction
This is the implementation of our work titled ''Doubly Robust Learning with Adaptive Learning Rates for Debiasing Post-Click Conversion Rate Estimation Under Sparse Data''.
## Environment
We provide the environment that our code depends on in ./requirements.txt. To install the environment, run
```bash
pip install -r requirements.txt
```
## Dataset
We propose our processed data in ./data/

## Run the Code
For Yahoo!R3, run:
```bash
python dr_alr_main.py --data_name=yahoo --thres=3 --debias_name=dr_alr --pred_model_name=mf --prop_model_name=logistic_regression --copy_model_pred=0 --embedding_k=64 --batch_size=4096 --batch_size_prop=32768 --hyper_str=4096_32768_0.05451113517934365_8_0.002430549177577126_2.379971870762413e-05_0.022213149039376996_0.00023193441568994852_0.0005500798277262966_2.345803458032337e-07 --is_tensorboard=0
```

For Coat, run:
```bash
python dr_alr_main.py --data_name=coat --thres=3 --val_rate=0.2 --debias_name=dr_alr --pred_model_name=mf --prop_model_name=logistic_regression --copy_model_pred=0 --embedding_k=16 --batch_size=128 --batch_size_prop=1024 --hyper_str=128_1024_0.08849835812057934_5_0.0023816388518405167_0.0004625855484434969_0.09988215027329404_3.394376569365401e-07_0.031771989079119455_0.006549999743438459 --is_tensorboard=0
```

For KuaiRec, run:
```bash
python dr_alr_main.py --data_name=kuai --thres=2 --debias_name=dr_alr --pred_model_name=mf --prop_model_name=logistic_regression --copy_model_pred=0 --embedding_k=64 --batch_size=4096 --batch_size_prop=32768 --hyper_str=4096_32768_0.07640626255241306_5_0.006152742961925225_0.00010705245764563371_0.07647306971338631_0.00014177672747735153_0.04798410261969416_0.9423863023273242 --is_tensorboard=0
```

For MovieLens-100K, run:
```bash
python dr_alr_main.py --data_name=ml100k --thres=3 --debias_name=dr_alr --pred_model_name=mf --prop_model_name=logistic_regression --copy_model_pred=0 --embedding_k=16 --grad_type=0 --sever=sui1 --tune_type=val --hyper_str=1024_16384_0.06412902800268186_10_0.055884573696062395_0.0001553561044280189_0.09996824747919476_0.000822310202324646_0.011620648238062525_1.9545901971467357e-05 --is_tensorboard=0
```
