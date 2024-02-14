# Improving Generalization in Semantic Parsing by Increasing Natural Language Variation

This repo is the implementation of the following paper:

Improving Generalization in Semantic Parsing by Increasing Natural Language Variation<br>
Irina Saparina and Mirella Lapata<br>
EACL'24

## License
This dataset is released under the [CC BY-SA 4.0 license](./LICENSE), meaning you must credit the original source and share any derivative works under the same license, even for commercial use.

## Data and Checkpoints

You can download augmentated Spider and evaluation datasets from [Google Drive](https://drive.google.com/file/d/1tfT7Zf-HKOuRDQ_1I_XT5Gebug7JWgnV/view?usp=sharing).

Preprocess Dr.Spider:
```bash
cd data/diagnostic-robustness-text-to-sql
python data_preprocess.py
```

Preprocess KaggleDBQA:
```bash
cd data/kaggle-dbqa
python preprocess.py
```

T5 checkpoint is available on the [HuggingFace Hub](https://huggingface.co/irisaparina/t5-3b-spider-nlvariation). 
RESDSQL checkpoints are available on [Google Drive](https://drive.google.com/file/d/1m1KEsZfJG-iptfaufT-vzHFDniMWmD76/view?usp=sharing). Download it and unzip files into ```models/RESDSQL```.


## Dependencies
Create conda env:
```bash
conda env create -n nlvariation_env -f enviroment.yaml
conda activate nlvariation_env
```

Install RESDSQL dependencies:
```bash
cd RESDSQL
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
python nltk_downloader.py
```

Clone evaluation scripts:
```bash
mkdir picard/third_party
cd picard/third_party
git clone https://github.com/facebookincubator/hsthrift
git clone https://github.com/facebook/zstd
git clone https://github.com/facebook/wangle
git clone https://github.com/facebook/folly
git clone https://github.com/elementai/spider
git clone https://github.com/elementai/test-suite-sql-eval
git clone https://github.com/hasktorch/tokenizers
git clone https://github.com/facebook/fbthrift
git clone https://github.com/fmtlib/fmt
git clone https://github.com/rsocket/rsocket-cpp
git clone https://github.com/facebookincubator/fizz
cd ../../

mkdir RESDSQL/third_party
cd RESDSQL/third_party
git clone https://github.com/ElementAI/spider.git
git clone https://github.com/ElementAI/test-suite-sql-eval.git
mv ./test-suite-sql-eval ./test_suite
```

## T5 and PICARD
The code used for exeperiments with T5 and PICARD is a fork of [official PICARD implementation](https://github.com/ServiceNow/picard):
```bash
cd picard
```

You can run T5 evaluation with:
```bash
sh ./configs/dr_spider/eval_dr_spider_t5-spider-augs.sh # Dr.Spider
sh ./configs/kaggle/eval_kaggle_t5-spider-augs.sh # KaggeDBQA
sh ./configs/geoquery/eval_geoquery_t5-spider-augs.sh # Dr.Spider
```

You need to use Docker (see more [info](https://github.com/ServiceNow/picard)) to run PICARD. You can run evaluation with:
```bash
sh ./configs/dr_spider/eval_dr_spider_t5-spider-augs.sh # Dr.Spider
sh ./configs/kaggle/eval_kaggle_t5-spider-augs.sh # KaggeDBQA
sh ./configs/geoquery/eval_geoquery_t5-spider-augs.sh # GeoQuery
```

You can run training on augmented dataset with:
```bash
python seq2seq/run_seq2seq.py configs/train_augs.json
```

## RESDSQL
The code used for exeperiments with RESDSQL is a fork of [official RESDSQL implementation](https://github.com/RUCKBReasoning/RESDSQL):
```bash
cd RESDSQL
```
You can run RESDSQL evaluation with:
```bash
sh ./configs/dr_spider/eval_dr_spider_t5-spider-augs.sh # Dr.Spider
sh ./configs/kaggle/eval_kaggle_t5-spider-augs.sh # KaggeDBQA
sh ./configs/geoquery/eval_geoquery_t5-spider-augs.sh # GeoQuery
```

You can run training on augmented dataset with:
```bash
sh ./configs/train_augs.sh
```

## Acknowledgements
We used the following datasets: [Spider](https://arxiv.org/abs/1809.08887), [Dr.Spider](https://arxiv.org/pdf/2301.08881.pdf), [KaggleDBQA](https://arxiv.org/abs/2106.11455), [GeoQuery](https://dl.acm.org/doi/10.5555/1864519.1864543). The code is based on [official PICARD implementation](https://github.com/ServiceNow/picard) and [official RESDSQL implementation](https://github.com/RUCKBReasoning/RESDSQL) (includes [NatSQL](https://arxiv.org/abs/2109.05153)).
We thank all authors for their work.