echo ./configs/geoquery/t5-spider-augs/eval_geoquery.json
PYTHONPATH='.' python seq2seq/run_seq2seq.py ./configs/geoquery/t5-spider-augs/eval_geoquery.json
echo "Evaluation is done!"