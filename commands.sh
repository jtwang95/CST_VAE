python train_CSTVAE.py -ic 1 -data multimnist -ns 4 -tp "112" -gpu 0 -key "cstvae_multimnist_ns_4_digit_4_tp_112"

python train_CSTVAE.py --model "cst_vae_lstm" -ic 1 -data multimnist -ns 4 -tp "112" -gpu 0 112 -key "cstvae_lstm_ns_4_digit_4_tp_112" 
