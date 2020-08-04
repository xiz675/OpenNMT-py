python -u ../translate.py \
    -model saved_models/$1  \
    -output prediction/${1/%pt/txt} \
    -src ../data/Twitter/tweetnews/test_post.txt \
    -conv ../data/Twitter/tweetnews/test_conv_wobm.txt \
    -beam_size 30 \
    -max_length 10 \
    -n_best 30 \
    -batch_size 64 > log/translate_${1%.pt}.log  \
&& python -u ../evaluate.py \
    -tgt ../data/Twitter/tweetnews/test_tag.txt \
    -pred prediction/${1/%pt/txt} \
>> log/translate_${1%.pt}.log &
