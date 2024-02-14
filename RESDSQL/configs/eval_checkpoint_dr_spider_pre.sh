declare -a dr_spider_dataset_for_eval=("NLQ_keyword_synonym" "NLQ_keyword_carrier" \
                                    "NLQ_column_synonym"  "NLQ_column_carrier" "NLQ_column_attribute"  "NLQ_column_attribute" \
                                    "NLQ_column_value" "NLQ_value_synonym" \
                                    "NLQ_multitype" "NLQ_others" \
                                    "SQL_DB_text" "SQL_comparison" "SQL_DB_number" "SQL_NonDB_number" "SQL_sort_order" \
                                    "DB_schema_synonym" "DB_schema_abbreviation" "DB_DBcontent_equivalence")

for dataset_for_eval in ${dr_spider_dataset_for_eval[@]}; do

    output="./predictions/${dataset_for_eval}/resdsql_3b_natsql_augs/pred_${dataset_for_eval}_pre.sql"
    db_path="../data/diagnostic-robustness-text-to-sql/data/${dataset_for_eval}/databases"
    table_path="../data/diagnostic-robustness-text-to-sql/data/${dataset_for_eval}/tables.json"
    input_dataset_path="../data/diagnostic-robustness-text-to-sql/data/${dataset_for_eval}/questions_pre_perturbation.json"

    case $dataset_for_eval in "DB_DBcontent_equivalence"|"DB_schema_abbreviation"|"DB_schema_synonym")
            table_path="../data/diagnostic-robustness-text-to-sql/data/${dataset_for_eval}/tables_pre_perturbation.json"
            db_path="../data/diagnostic-robustness-text-to-sql/data/${dataset_for_eval}/databases_pre_perturbation"
    esac

    # prepare table file for natsql
    python NatSQL/table_transform.py \
        --in_file $table_path \
        --out_file "../data/RESDSQL_files/preprocessed_data/test_tables_for_natsql.json" \
        --correct_col_type \
        --remove_start_table  \
        --analyse_same_column \
        --table_transform \
        --correct_primary_keys \
        --use_extra_col_types \
        --db_path $db_path

    # preprocess test set
    python preprocessing.py \
        --mode "test" \
        --table_path $table_path \
        --input_dataset_path $input_dataset_path \
        --output_dataset_path "../data/RESDSQL_files/preprocessed_data/preprocessed_test_natsql_${dataset_for_eval}.json" \
        --db_path $db_path \
        --target_type "natsql"

    # predict probability for each schema item in the test set
    python schema_item_classifier.py \
        --batch_size 32 \
        --device 0 \
        --seed 42 \
        --save_path "../models/RESDSQL/text2natsql_schema_item_classifier_augs" \
        --dev_filepath "../data/RESDSQL_files/preprocessed_data/preprocessed_test_natsql_${dataset_for_eval}.json" \
        --output_filepath "../data/RESDSQL_files/preprocessed_data/test_with_probs_natsql_${dataset_for_eval}.json" \
        --use_contents \
        --mode "test"

    # generate text2natsql test set
    python text2sql_data_generator.py \
        --input_dataset_path "../data/RESDSQL_files/preprocessed_data/test_with_probs_natsql_${dataset_for_eval}.json" \
        --output_dataset_path "../data/RESDSQL_files/preprocessed_data/resdsql_test_natsql_${dataset_for_eval}.json" \
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
        --dev_filepath "../data/RESDSQL_files/preprocessed_data/resdsql_test_natsql_${dataset_for_eval}.json" \
        --original_dev_filepath $input_dataset_path \
        --db_path $db_path \
        --tables_for_natsql "../data/RESDSQL_files/preprocessed_data/test_tables_for_natsql.json" \
        --num_beams 8 \
        --num_return_sequences 8 \
        --target_type "natsql" \
        --output $output
done