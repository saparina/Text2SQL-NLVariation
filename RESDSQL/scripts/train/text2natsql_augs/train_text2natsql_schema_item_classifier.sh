set -e

# train schema item classifier
python -u schema_item_classifier.py \
    --batch_size 16 \
    --gradient_descent_step 2 \
    --device "0" \
    --learning_rate 1e-5 \
    --gamma 2.0 \
    --alpha 0.75 \
    --epochs 128 \
    --patience 16 \
    --seed 42 \
    --save_path "../models/RESDSQL/text2natsql_schema_item_classifier_augs" \
    --tensorboard_save_path "./tensorboard_log/text2natsql_schema_item_classifier_augs" \
    --train_filepath "../data/RESDSQL_files/preprocessed_data/preprocessed_train_spider_natsql.json" \
    --dev_filepath "../data/RESDSQL_files/preprocessed_data/preprocessed_dev_natsql.json" \
    --model_name_or_path "roberta-large" \
    --model_type  "roberta-large" \
    --use_contents \
    --mode "train"