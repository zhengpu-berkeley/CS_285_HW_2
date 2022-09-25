for batchsize in 500 1000 2000 5000 10000
do
    for learningrate in 0.01 0.005 0.001 0.0005 0.0001
    do
        python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
        --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b $batchsize -lr $learningrate -rtg \
        --exp_name q2_b"$batchsize"_r"$learningrate"
    done
done