export CUDA_VISIBLE_DEVICES=$1
id=$2

group="20231223$id"
log_dir="~/results/hyperaction/bandit/$group"

freq_task=1
n_context=1
time_period=100000
n_features=100
n_arms=1000
n_expe=10
update_num=1
NpS=16
action_noise=pn
update_noise=pn
buffer_noise=sp
# method=TS
method=Hyper
# method=Ensemble
# method=EpiNet
for game in Synthetic-v1 Synthetic-v2 Synthetic-v3 Synthetic-v4
# for game in RealData-v1 RealData-v2 RealData-v3 RealData-v4
do
    tag=$(date "+%Y%m%d%H%M%S")
    python -m scripts.run_hyper_v2 --game=${game} --method=${method} \
        --freq-task=${freq_task} --n-context=${n_context} \
        --n-features=${n_features} --n-arms=${n_arms} \
        --action-noise=${action_noise} --update-noise=${update_noise} --buffer-noise=${buffer_noise} \
        --NpS=${NpS} --update-num=${update_num} \
        --time-period=${time_period} --n-expe=${n_expe} --log-dir=${log_dir} \
        > ~/logs/${game}_${tag}.out 2> ~/logs/${game}_${tag}.err &
    echo "run $game $tag"
    sleep 1.0
done

# ps -ef | grep Synthetic | awk '{print $2}'| xargs kill -9
