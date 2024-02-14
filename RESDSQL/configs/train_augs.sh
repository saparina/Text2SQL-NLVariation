# Step1: preprocess dataset
echo "sh scripts/train/text2natsql_augs/preprocess.sh"
sh scripts/train/text2natsql_augs/preprocess.sh
# Step2: train cross-encoder
echo "sh scripts/train/text2natsql_augs/train_text2natsql_schema_item_classifier.sh"
sh scripts/train/text2natsql_augs/train_text2natsql_schema_item_classifier.sh
# Step3: prepare text-to-natsql training and development set for T5
echo "sh scripts/train/text2natsql_augs/generate_text2natsql_dataset.sh"
sh scripts/train/text2natsql_augs/generate_text2natsql_dataset.sh
# Step4: fine-tune T5-3B (RESDSQL-3B+NatSQL)
echo "sh scripts/train/text2natsql_augs/train_text2natsql_t5_3b.sh"
sh scripts/train/text2natsql_augs/train_text2natsql_t5_3b.sh