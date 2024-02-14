set -e

# train text2natsql-t5-3b model
python -u text2sql.py \
    --batch_size 3 \
    --gradient_descent_step 32 \
    --device "0" \
    --learning_rate 5e-5 \
    --epochs 128 \
    --seed 42 \
    --save_path "../models/RESDSQL/text2natsql-t5-3b-augs" \
    --tensorboard_save_path "./tensorboard_log/text2natsql-t5-3b-augs" \
    --model_name_or_path "t5-3b" \
    --use_adafactor \
    --mode train \
    --train_filepath "../data/RESDSQL_files/preprocessed_data/resdsql_train_spider_natsql.json" \
    --dev_filepath "../data/RESDSQL_files/preprocessed_data/resdsql_dev_natsql.json" \
    --original_dev_filepath "../data/spider_augs/dev.json" \
    --db_path "./database" \
    --tables_for_natsql "../data/RESDSQL_files/preprocessed_data/tables_for_natsql.json" \
    --num_beams 1 \
    --num_return_sequences 1 \
    --target_type "natsql"
