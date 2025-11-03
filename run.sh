start_time=$(date +%s)
python scripts/train_stage1.py \
    --data_dir data \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --layerdrop 0.0 \
    --weight_decay 0.0 \
    --label_smoothing 0.0 \
    --patience 30 \
    --output_dir checkpoints/stage1 > logs/stage1.log
end_time=$(date +%s)
echo "Time taken: $((end_time - start_time)) seconds"

start_time=$(date +%s)
python scripts/train_stage2.py \
    --stage1_checkpoint checkpoints/stage1/best_model.pth \
    --data_dir data \
    --total_layers 12 \
    --batch_size 8 \
    --lr 1e-4 \
    --alpha 1.0 \
    --beta 5.0 \
    --seed 42 \
    --epochs 100 \
    --output_dir checkpoints/stage2 > logs/stage2.log
end_time=$(date +%s)
echo "Time taken: $((end_time - start_time)) seconds"

start_time=$(date +%s)
python scripts/detailed_evaluation.py > logs/detailed_evaluation.log
end_time=$(date +%s)
echo "Time taken: $((end_time - start_time)) seconds"
