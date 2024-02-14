echo ./configs/kaggle/picard-spider-augs/eval_GeoNuclearData_full.json
PYTHONPATH='.' python seq2seq/run_seq2seq.py ./configs/kaggle/picard-spider-augs/eval_GeoNuclearData_full.json
echo ./configs/kaggle/picard-spider-augs/eval_GreaterManchesterCrime_full.json
PYTHONPATH='.' python seq2seq/run_seq2seq.py ./configs/kaggle/picard-spider-augs/eval_GreaterManchesterCrime_full.json
echo ./configs/kaggle/picard-spider-augs/eval_Pesticide_full.json
PYTHONPATH='.' python seq2seq/run_seq2seq.py ./configs/kaggle/picard-spider-augs/eval_Pesticide_full.json
echo ./configs/kaggle/picard-spider-augs/eval_StudentMathScore_full.json
PYTHONPATH='.' python seq2seq/run_seq2seq.py ./configs/kaggle/picard-spider-augs/eval_StudentMathScore_full.json
echo ./configs/kaggle/picard-spider-augs/eval_TheHistoryofBaseball_full.json
PYTHONPATH='.' python seq2seq/run_seq2seq.py ./configs/kaggle/picard-spider-augs/eval_TheHistoryofBaseball_full.json
echo ./configs/kaggle/picard-spider-augs/eval_USWildFires_full.json
PYTHONPATH='.' python seq2seq/run_seq2seq.py ./configs/kaggle/picard-spider-augs/eval_USWildFires_full.json
echo ./configs/kaggle/picard-spider-augs/eval_WhatCDHipHop_full.json
PYTHONPATH='.' python seq2seq/run_seq2seq.py ./configs/kaggle/picard-spider-augs/eval_WhatCDHipHop_full.json
echo ./configs/kaggle/picard-spider-augs/eval_WorldSoccerDataBase_full.json
PYTHONPATH='.' python seq2seq/run_seq2seq.py ./configs/kaggle/picard-spider-augs/eval_WorldSoccerDataBase_full.json
echo "Evaluation is done!"