id=$1

group="20240812$id"
log_dir="/apdcephfs/share_1563664/ztjiaweixu/bandit_sz/$group"

time_period=1000000
n_features=512
n_arms=1
n_expe=1
NpS=16
noise_dim=4
action_noise=sp
update_noise=pm
buffer_noise=sp
batch_size=16
update_start=20
update_num=1
update_freq=1
lr=1e-5
hyper_weight_decay=0.01
based_weight_decay=0.01
prior_scale=0.1
posterior_scale=0.1
fine_tune=1
llm_name=pythia14m
# llm_name=gpt2
# model_type=linear
model_type=hyper
# model_type=ensemble
cuda_id=0
for game in hatespeech
do
    seed=2020
    for i in $(seq 4)
    do
        export CUDA_VISIBLE_DEVICES=${cuda_id}
        tag=$(date "+%Y%m%d%H%M%S")
        python -m scripts.run_llm --game=${game} --seed=${seed} --n-features=${n_features} --n-arms=${n_arms} \
            --noise-dim=${noise_dim} --NpS=${NpS} --model-type=${model_type} --llm-name=${llm_name} \
            --hyper-weight-decay=${hyper_weight_decay} --based-weight-decay=${based_weight_decay} \
            --batch-size=${batch_size} --lr=${lr} \
            --update-start=${update_start} --update-num=${update_num} --update-freq=${update_freq} \
            --prior-scale=${prior_scale} --posterior-scale=${posterior_scale} --fine-tune=${fine_tune} \
            --action-noise=${action_noise} --update-noise=${update_noise} --buffer-noise=${buffer_noise} \
            --time-period=${time_period} --n-expe=${n_expe} --log-dir=${log_dir} \
            > ~/logs/${game}_${tag}.out 2> ~/logs/${game}_${tag}.err &
        echo "run $model_type $game $cuda_id $seed $tag"
        let seed=$seed+1
        # let cuda_id=$cuda_id+1
        sleep 1.0
    done
    let cuda_id=$cuda_id+1
done

python taiji/run_gpu.py
# ps -ef | grep llm | awk '{print $2}'| xargs kill -9
