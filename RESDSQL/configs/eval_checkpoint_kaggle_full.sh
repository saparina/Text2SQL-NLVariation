# prepare table file for natsql
python NatSQL/table_transform.py \
    --in_file "../data/kaggle-dbqa/KaggleDBQA_tables.json" \
    --out_file "../data/RESDSQL_files/preprocessed_data/test_tables_for_natsql.json" \
    --correct_col_type \
    --remove_start_table  \
    --analyse_same_column \
    --table_transform \
    --correct_primary_keys \
    --use_extra_col_types \
    --db_path "../data/kaggle-dbqa/databases/"


declare -a all_datasets_for_eval=("GeoNuclearData" "GreaterManchesterCrime" "Pesticide" \
      "StudentMathScore" "TheHistoryofBaseball" "USWildFires" "WhatCDHipHop" "WorldSoccerDataBase")

for dataset_for_eval in ${all_datasets_for_eval[@]}; do
    input_dataset_path="../data/kaggle-dbqa/examples/${dataset_for_eval}_full/dev.json"
    output="./predictions/kaggle/resdsql_3b_natsql_augs/pred_${dataset_for_eval}.sql"

    # preprocess test set
    python preprocessing.py \
        --mode "test" \
        --table_path "../data/kaggle-dbqa/KaggleDBQA_tables.json" \
        --input_dataset_path $input_dataset_path \
        --output_dataset_path "../data/RESDSQL_files/preprocessed_data/preprocessed_test_natsql.json" \
        --db_path "../data/kaggle-dbqa/databases/" \
        --target_type "natsql"

    # predict probability for each schema item in the test set
    python schema_item_classifier.py \
        --batch_size 32 \
        --device 0 \
        --seed 42 \
        --save_path "../models/RESDSQL/text2natsql_schema_item_classifier_augs" \
        --dev_filepath "../data/RESDSQL_files/preprocessed_data/preprocessed_test_natsql.json" \
        --output_filepath "../data/RESDSQL_files/preprocessed_data/test_with_probs_natsql.json" \
        --use_contents \
        --mode "test"

    # generate text2natsql test set
    python text2sql_data_generator.py \
        --input_dataset_path "../data/RESDSQL_files/preprocessed_data/test_with_probs_natsql.json" \
        --output_dataset_path "../data/RESDSQL_files/preprocessed_data/resdsql_test_natsql.json" \
        --topk_table_num 4 \
        --topk_column_num 5 \
        --mode "test" \
        --use_contents \
        --output_skeleton \
        --target_type "natsql"

    # inference using the best text2natsql ckpt
    python text2sql.py \
        --batch_size 6 \
        --device 0 \
        --seed 42 \
        --save_path "../models/RESDSQL/text2natsql-t5-3b-augs/checkpoint-329967" \
        --mode "eval" \
        --dev_filepath "../data/RESDSQL_files/preprocessed_data/resdsql_test_natsql.json" \
        --original_dev_filepath $input_dataset_path \
        --db_path "../data/kaggle-dbqa/databases/" \
        --tables_for_natsql "../data/RESDSQL_files/preprocessed_data/test_tables_for_natsql.json" \
        --num_beams 8 \
        --num_return_sequences 8 \
        --target_type "natsql" \
        --output $output
done