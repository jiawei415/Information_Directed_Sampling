export CUDA_VISIBLE_DEVICES=$1
id=$2

group="20231102$id"
log_dir="~/results/hyperaction/srcon/$group"

env=blackbox
d_theta=10
time_period=1000
n_expe=10
update_num=2
NpS=16
action_noise=sps
update_noise=sps
method=Hyper
# method=Ensemble
# method=EpiNet
for dataset in Bukin Branin Hartmann Schwefel 
# for dataset in Ackley Michalewicz Levy Rosenbrock
# for dataset in korea chengdu hanjiang nandong
do
    tag=$(date "+%Y%m%d%H%M%S")
    python srcon/run_hyper.py --env=${env} --dataset=${dataset} --d-theta=${d_theta} \
        --method=${method} --update-num=${update_num} --NpS=${NpS} \
        --action-noise=${action_noise} --update-noise=${update_noise} \
        --time-period=${time_period} --n-expe=${n_expe} --log-dir=${log_dir} \
        > ~/logs/${dataset}_${tag}.out 2> ~/logs/${dataset}_${tag}.err &
    echo "run $dataset $tag"
    sleep 1.0
done

# ps -ef | grep Synthetic | awk '{print $2}'| xargs kill -9
