id=$1
log_dir="/apdcephfs/share_1563664/ztjiaweixu/bandit_sz/2024$id"

time_period=10000
NpS=16
noise_dim=8
action_noise=sp
update_noise=pm
buffer_noise=sp
method=Hyper
# method=EpiNet
# method=Ensemble
# method=LMCTS
# method=NeuralUCB

pip install scikit-learn==0.22.1
pip install pandas==1.3.5
pip install numpy==1.20.0

cuda_id=0
for game in RealData-v1
# for game in RealData-v1 RealData-v3 RealData-v4
do
    export CUDA_VISIBLE_DEVICES=${cuda_id}
    seed=0
    for i in $(seq 5)
    do
        tag=$(date "+%Y%m%d%H%M%S")
        python -m scripts.run_nonlinear --game=${game} --method=${method} --seed=${seed} \
            --time-period=${time_period} \
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
