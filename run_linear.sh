d_index=$1 # 2^1 to 2^10
d_theta=50 # 10 50
n_arms=100 # 100 1000 10000
n_expe=200 # {200, 1000}

python scripts/run_linear.py --n-expe ${n_expe} --game Russo \
       	--time-period 1000 --d-index ${d_index} \
       	--d-theta ${d_theta} --n-arms ${n_arms}

# Zhang, FreqRusso, Russo, movieLens, Synthetic-v1, Synthetic-v2
# ps -ef | grep DeepSea | awk '{print $2}'| xargs kill -9