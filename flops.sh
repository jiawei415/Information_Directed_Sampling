

for method in Ensemble EpiNet Hyper 
do
    for M in 4 8 16 32 64 128 256
    do
        python -m scripts.test_flops --method=${method} --noise-dim=${M}
    done
    echo "done $method"
done