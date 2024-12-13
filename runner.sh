lrs=(3e-5 1e-4 3e-4 1e-3)

for lr in ${lrs[@]}; do
    torchrun --nproc_per_node=8 train_ref.py --lr=$lr
    torchrun --nproc_per_node=8 train.py --lr=$lr
done
