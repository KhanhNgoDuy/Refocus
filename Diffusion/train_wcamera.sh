export INSTANCE_DIR="/mnt/HDD1/tuong/workspace/khanh/courses/image_processing/final-project/datasets"
export OUTPUT_DIR="./checkpoints/camera"
# export CUDA_VISIBLE_DEVICES=4,5,6
export CUDA_VISIBLE_DEVICES=0

python train.py \
  --root=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --resolution=512 \
  --train_batch_size=1 \
  --num_workers=4 \
  --gradient_accumulation_steps=64 \
  --num_train_epochs=1000 \
  --learning_rate=1e-04 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --gradient_checkpointing \
  --split_file=$SPLIT_FILE \
  --cam