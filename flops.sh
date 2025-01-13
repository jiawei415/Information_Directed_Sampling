

for method in EpiNet Ensemble Hyper
do
    # for M in 4 8 16 32 64 128 256
    for M in 10 20 30 40 50 60 70 80 90 100
    # for M in 2 6 12 14 18
    do
        python -m scripts.test_flops --method=${method} --noise-dim=${M}
    done
    echo "done $method"
done