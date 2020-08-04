export CUDA_VISIBLE_DEVICES=7
dataset=raw
data_tag=Twitter_all_nonshare
emb_size=200
rnn=300
learning_rate=0.0008
learning_decay=0.5
seed=23


model_name=${learning_rate}_${learning_decay}_${rnn}rnn_${emb_size}emb_seed${seed}_${data_tag}
nohup \
python -u ../train.py \
    -word_vec_size ${emb_size} \
    --train_steps 10000 \
    -save_checkpoint_steps 630 \
    -model_type text \
    -encoder_type biattention  \
    -decoder_type rnn \
    -valid_steps  3200 \
    -enc_layers 2  \
    -dec_layers 1 \
    -rnn_size ${rnn} \
    -rnn_type GRU \
    -global_attention general \
    -copy_attn \
    -save_model saved_models/${model_name} \
    -seed ${seed} \
    -data ../processed_data/${dataset}/${data_tag} \
    -batch_size 64 \
    -input_feed 1 \
    -optim adam \
    -max_grad_norm 1 \
    -dropout 0.1 \
    -learning_rate ${learning_rate} \
    -gpu_ranks 0 \
    -learning_rate_decay ${learning_decay} \
   > log/train_${model_name}.log&
