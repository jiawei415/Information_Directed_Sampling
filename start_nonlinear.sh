id=$1

log_dir="/apdcephfs/share_1563664/ztjiaweixu/bandit_sz/2024$id"

freq_task=1
n_context=1
time_period=1000000
n_features=100
n_arms=1000
NpS=16
noise_dim=4
action_noise=sp
update_noise=pm
buffer_noise=sp
method=Hyper
# method=Ensemble
# method=EpiNet
# method=LMCTS
cuda_id=0
for game in Synthetic-v1 Synthetic-v4
# for game in Synthetic-v1 Synthetic-v2 Synthetic-v3 Synthetic-v4
# for game in RealData-v1 RealData-v2 RealData-v3 RealData-v4
do
    export CUDA_VISIBLE_DEVICES=${cuda_id}
    seed=0
    for i in $(seq 5)
    do
        tag=$(date "+%Y%m%d%H%M%S")
        python -m scripts.run_nonlinear --game=${game} --method=${method} --seed=${seed} \
            --freq-task=${freq_task} --n-context=${n_context} --time-period=${time_period} \
            --n-features=${n_features} --n-arms=${n_arms} \
            --noise-dim=${noise_dim} --NpS=${NpS} \
            --action-noise=${action_noise} --update-noise=${update_noise} --buffer-noise=${buffer_noise} \
            --log-dir=${log_dir} \
            > ~/logs/${game}_${tag}.out 2> ~/logs/${game}_${tag}.err &
        echo "run $method $game $cuda_id $seed $tag"
        let seed=$seed+1
        sleep 1.0
    done
    let cuda_id=$cuda_id+1
done

python taiji/run_gpu.py
# ps -ef | grep hyper | awk '{print $2}'| xargs kill -9
