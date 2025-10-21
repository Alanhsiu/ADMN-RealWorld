# Prepare dataset
python data/prepare_dataset.py \
    --raw_dir data/simple_raw_data \
    --output_dir data/processed \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --seed 42

# Verify dataset
python data/verify_dataset.py --data_dir data/processed