python_path=$1
upstream_name=$2
downstream_name=$3
mode=$4 # train, evaluate, or resume
exp_name=$5
config_path=$6
S3PRL_ROOT=$7
upstream_ckpt=$8
upstream_ckpt_arg=""
if [ -n "$upstream_ckpt" ]; then
    upstream_ckpt_arg="-k $upstream_ckpt"
fi

cd $S3PRL_ROOT/s3prl
if [ "$mode" = "resume" ]; then
    if [ "$downstream_name" = "er" ]; then
        # Resume training on ER
        $python_path run_downstream.py -m train -e result/downstream/er_finetune_$exp_name/
    elif [ "$downstream_name" = "ast_ted" ]; then
        # Resume training on ASR on TED
        $python_path run_downstream.py -m train -e result/downstream/asr_ted_finetune_$exp_name/
    else
        echo "Unknown downstream_name for resume."
    fi
elif [ "$mode" = "evaluate" ]; then
    if [ "$downstream_name" = "er" ]; then
        # Evaluate on ER
        $python_path run_downstream.py -m evaluate -e result/downstream/er_finetune_$exp_name/dev-best.ckpt
    elif [ "$downstream_name" = "asr_ted" ]; then
        # Evaluate on ASR on TED
        $python_path run_downstream.py -m evaluate -e result/downstream/asr_ted_finetune_$exp_name/dev-best.ckpt
    else
        echo "Unknown downstream_name for evaluate."
    fi
elif [ "$mode" = "train" ]; then
    if [ "$downstream_name" = "er" ]; then
        # Fine-tune on ER
        $python_path run_downstream.py -m train -u $upstream_name \
            -d emotion \
            -n er_finetune_$exp_name -c $config_path \
            -o "config.downstream_expert.datarc.test_fold='fold1'" \
            -s last_hidden_state --upstream_trainable $upstream_ckpt_arg
    elif [ "$downstream_name" = "asr_ted" ]; then
        # Fine-tune on ASR on TED
        $python_path run_downstream.py -m train -u $upstream_name \
            -d asr_ted -c $config_path \
            -n asr_ted_finetune_$exp_name -s last_hidden_state --upstream_trainable $upstream_ckpt_arg
    else
        echo "Unknown downstream_name."
    fi
else 
    echo "Unknown mode."
fi
