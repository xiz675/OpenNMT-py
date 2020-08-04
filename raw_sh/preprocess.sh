data_tag='Twitter'
base_path=raw
dataset=../data/Twitter

vs=40000
cvs=40000
sl=35
slt=35
cl=200
clt=100
tl=10


if [[ ! -e ../processed_data ]]
then
    mkdir ../processed_data
fi

#full_data_tag=${data_tag}_vs${vs}_cvs_${cvs}_nonshare
full_data_tag=${data_tag}_all_nonshare

python -u ../preprocess.py \
    -train_src $dataset/${base_path}/trainnew_post.txt \
    -train_conv $dataset/${base_path}/trainnew_conv.txt \
    -train_tgt $dataset/${base_path}/trainnew_tag.txt \
    -valid_src $dataset/${base_path}/validnew_post.txt \
    -valid_conv $dataset/${base_path}/validnew_conv.txt \
    -valid_tgt $dataset/${base_path}/validnew_tag.txt \
    -save_data ../processed_data/${base_path}/${full_data_tag}  \
    -tgt_seq_length ${tl} \
    -src_seq_length_trunc ${slt} \
    -conv_seq_length_trunc ${clt} \
    -dynamic_dict \
    -overwrite \
    #-src_vocab_size ${vs} \
    #-conv_vocab_size ${cvs} \
    # -share_vocab 
#    -tgt_vocab_size 10000
# -share_vocab
