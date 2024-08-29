learning_rates=(0.003 0.0034 0.0067) 
# Loop through all combinations of parameters
for lr in "${learning_rates[@]}"; do
    # Submit job with parameters
    sbatch /n/home04/amuppidi/workshop/train.sh --lr="$lr"
    echo "Submitted: learning rate: $lr"
    sleep 0.3
done