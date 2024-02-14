echo ./configs/geoquery/picard-spider-augs/eval_geoquery.json
PYTHONPATH='.' python seq2seq/run_seq2seq.py ./configs/geoquery/picard-spider-augs/eval_geoquery.json
echo "Evaluation is done!"