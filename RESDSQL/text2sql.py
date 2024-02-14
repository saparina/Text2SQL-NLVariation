import os
import json
import torch
import argparse
import shutil
import torch.optim as optim
import transformers
import random
import numpy as np

from tqdm import tqdm
from tokenizers import AddedToken

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import T5TokenizerFast, T5ForConditionalGeneration, MT5ForConditionalGeneration
from transformers.optimization import Adafactor
from transformers.trainer_utils import set_seed
from utils.spider_metric.evaluator import EvaluateTool
from utils.load_dataset import Text2SQLDataset
from utils.text2sql_decoding_utils import decode_sqls, decode_natsqls

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for fine-tuning pre-trained language model.")
    
    parser.add_argument('--batch_size', type = int, default = 8,
                        help = 'input batch size.')
    parser.add_argument('--gradient_descent_step', type = int, default = 4,
                        help = 'perform gradient descent per "gradient_descent_step" steps.')
    parser.add_argument('--device', type = str, default = "2",
                        help = 'the id of used GPU device.')
    parser.add_argument('--learning_rate',type = float, default = 3e-5,
                        help = 'learning rate.')
    parser.add_argument('--epochs', type = int, default = 128,
                        help = 'training epochs.')
    parser.add_argument('--seed', type = int, default = 42,
                        help = 'random seed.')
    parser.add_argument('--save_path', type = str, default = "models/text2sql",
                        help = 'save path of best fine-tuned text2sql model.')
    parser.add_argument('--tensorboard_save_path', type = str, default = "tensorboard_log/text2sql",
                        help = 'save path of tensorboard log.')
    parser.add_argument('--model_name_or_path', type = str, default = "t5-3b",
                        help = 
                        '''
                        pre-trained model name. 
                        options: 
                            t5-base, https://huggingface.co/t5-base;
                            t5-large, https://huggingface.co/t5-large;
                            t5-3b, https://huggingface.co/t5-3b;
                        ''')
    parser.add_argument('--use_adafactor', action='store_true',
                        help = 'whether to use adafactor optimizer.')
    parser.add_argument('--mode', type = str, default = "train",
                        help='trian, eval or test.')
    parser.add_argument('--train_filepath', type = str, default = "data/preprocessed_data/resdsql_train_spider.json",
                        help = 'file path of test2sql training set.')
    parser.add_argument('--dev_filepath', type = str, default = "data/preprocessed_data/resdsql_dev.json",
                        help = 'file path of test2sql dev set.')
    parser.add_argument('--original_dev_filepath', type = str, default = "data/spider/dev.json",
                        help = 'file path of the original dev set (for registing evaluator).')
    parser.add_argument('--db_path', type = str, default = "database",
                        help = 'file path of database.')
    parser.add_argument('--tables_for_natsql', type = str, default = "NatSQL/NatSQLv1_6/tables_for_natsql.json",
                        help = 'file path of tables_for_natsql.json.')
    parser.add_argument('--num_beams', type = int, default = 8,
                        help = 'beam size in model.generate() function.')
    parser.add_argument('--num_return_sequences', type = int, default = 8,
                        help = 'the number of returned sequences in model.generate() function (num_return_sequences <= num_beams).')
    parser.add_argument("--target_type", type = str, default = "sql",
                help = "sql or natsql.")
    parser.add_argument("--output", type = str, default = "predicted_sql.txt",
                help = "save file of the predicted sqls.")
    
    opt = parser.parse_args()

    return opt

def _train(opt):
    set_seed(opt.seed)
    print(opt)

    if opt.tensorboard_save_path is not None:
        writer = SummaryWriter(opt.tensorboard_save_path)
    else:
        writer = None

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device

    model_class = T5ForConditionalGeneration # MT5ForConditionalGeneration if "mt5" in opt.model_name_or_path else 

    text2sql_tokenizer = T5TokenizerFast.from_pretrained(
        # opt.model_name_or_path,
        "t5-3b",
        add_prefix_space = True,
        cache_dir="/transformers_cache/"
    )

    if isinstance(text2sql_tokenizer, T5TokenizerFast):
        text2sql_tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])
    
    train_dataset = Text2SQLDataset(
        dir_ = opt.train_filepath,
        mode = "train"
    )

    train_dataloder = DataLoader(
        train_dataset, 
        batch_size = opt.batch_size, 
        shuffle = True,
        collate_fn = lambda x: x,
        drop_last = True
    )

    dev_dataset = Text2SQLDataset(
        dir_ = opt.dev_filepath,
        mode = 'eval'
    )

    dev_dataloder = DataLoader(
        dev_dataset, 
        batch_size = opt.batch_size, 
        shuffle = False,
        collate_fn = lambda x: x,
        drop_last = False
    )
    if opt.target_type == "natsql":
        tables = json.load(open(opt.tables_for_natsql,'r'))
        table_dict = dict()
        for t in tables:
            table_dict[t["db_id"]] = t

    print("initializing text2sql model.")
    # initialize model
    model = model_class.from_pretrained(opt.model_name_or_path, cache_dir="/transformers_cache/")
    model.resize_token_embeddings(len(text2sql_tokenizer))
    if torch.cuda.is_available():
        model = model.cuda()
    
    print("finished.")

    # warm up steps (10% training step)
    # num_warmup_steps = int(0.5*0.1*opt.epochs*len(train_dataset)/opt.batch_size)
    num_warmup_steps = int(0.1*opt.epochs*len(train_dataset)/opt.batch_size)
    # total training steps
    # num_training_steps = int(0.5*opt.epochs*len(train_dataset)/opt.batch_size)
    num_training_steps = int(opt.epochs*len(train_dataset)/opt.batch_size)
    # save checkpoint for each 1.42857 epochs (about 1.42857*7000=10000 examples for Spider's training set)
    num_checkpoint_steps = int(10000/opt.batch_size) #int(1.42857 * len(train_dataset)/opt.batch_size)

    if opt.use_adafactor:
        print("Let's use Adafactor!")
        optimizer = Adafactor(
            model.parameters(), 
            lr=opt.learning_rate, 
            scale_parameter=False, 
            relative_step=False, 
            clip_threshold = 1.0,
            warmup_init=False
        )
    else:
        print("Let's use AdamW!")
        optimizer = optim.AdamW(
            model.parameters(), 
            lr = opt.learning_rate
        )

    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = num_warmup_steps,
        num_training_steps = num_training_steps
    )

    model.train()
    train_step = 0
    saved_checkpoints = []
    resume_epoch, resume_train_step = None, None

    if opt.model_name_or_path.find('checkpoint') >= 0:
        print(f'Loading from {opt.model_name_or_path}')
        optimizer_scheduler = torch.load(os.path.join(opt.model_name_or_path, 'optimizer_scheduler.pth'))
        optimizer.load_state_dict(optimizer_scheduler['optimizer'])
        scheduler.load_state_dict(optimizer_scheduler['scheduler'])
        saved_checkpoints = optimizer_scheduler['saved_checkpoints']
        # new_saved_checkpoints = []
        # for ch in saved_checkpoints:
        #     if ch.find('checkpoint-156651')>=0  or \
        #         ch.find('checkpoint-163317') >=0 or ch.find('checkpoint-166650') >=0:
        #         new_saved_checkpoints.append(ch)
        # saved_checkpoints = new_saved_checkpoints
        # print(saved_checkpoints)
        resume_epoch = optimizer_scheduler['epoch']
        resume_train_step = optimizer_scheduler['train_step']

        checkpoint_rng_state = torch.load(os.path.join(opt.model_name_or_path, 'rng_states.pth'))
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        if torch.cuda.is_available():
            torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])
    
    for epoch in range(opt.epochs):
        if resume_epoch is not None and epoch < resume_epoch:
            for batch in train_dataloder:
                train_step += 1
            continue
        print(f"This is epoch {epoch+1}.")
        for batch in train_dataloder:
            if resume_train_step is not None and train_step <= resume_train_step:
                train_step += 1
                continue
            elif resume_train_step is not None and train_step - 1 == resume_train_step:
                print(f'Resume from epoch {resume_epoch} training step {resume_train_step}')

            train_step += 1
            
            batch_inputs = [data[0] for data in batch]
            batch_sqls = [data[1] for data in batch]
            batch_db_ids = [data[2] for data in batch] # unused
            batch_tc_original = [data[3] for data in batch] # unused
            
            # if epoch == 0:
            #     for batch_id in range(len(batch_inputs)):
            #         print(batch_inputs[batch_id])
            #         print(batch_sqls[batch_id])
            #         print("----------------------")

            tokenized_inputs = text2sql_tokenizer(
                batch_inputs, 
                padding = "max_length",
                return_tensors = "pt",
                max_length = 512,
                truncation = True
            )
            
            with text2sql_tokenizer.as_target_tokenizer():
                tokenized_outputs = text2sql_tokenizer(
                    batch_sqls, 
                    padding = "max_length", 
                    return_tensors = 'pt',
                    max_length = 256,
                    truncation = True
                )
            
            encoder_input_ids = tokenized_inputs["input_ids"]
            encoder_input_attention_mask = tokenized_inputs["attention_mask"]

            decoder_labels = tokenized_outputs["input_ids"]
            decoder_labels[decoder_labels == text2sql_tokenizer.pad_token_id] = -100
            decoder_attention_mask = tokenized_outputs["attention_mask"]

            if torch.cuda.is_available():
                encoder_input_ids = encoder_input_ids.cuda()
                encoder_input_attention_mask = encoder_input_attention_mask.cuda()
                decoder_labels = decoder_labels.cuda()
                decoder_attention_mask = decoder_attention_mask.cuda()
            
            model.train()
            model_outputs = model(
                input_ids = encoder_input_ids,
                attention_mask = encoder_input_attention_mask,
                labels = decoder_labels,
                decoder_attention_mask = decoder_attention_mask,
                return_dict = True
            )
            
            loss = model_outputs["loss"]
            loss.backward()

            if scheduler is not None: #and train_step % 2 == 1:
                scheduler.step()

            if writer is not None:
                # record training loss (tensorboard)
                writer.add_scalar('train loss', loss.item(), train_step)
                # record learning rate (tensorboard)
                writer.add_scalar('train lr', optimizer.state_dict()['param_groups'][0]['lr'], train_step)

            if train_step % opt.gradient_descent_step == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            if train_step > num_checkpoint_steps * 5 and train_step % num_checkpoint_steps == 0: # train_step > num_checkpoint_steps * 5 and 
                print(f"At {train_step} training step, start an evaluation.")
                #print(f"At {epoch + 1} training step, eval a checkpoint.")
                model.eval()
                predict_sqls = []
                for batch in tqdm(dev_dataloder):
                    batch_inputs = [data[0] for data in batch]
                    batch_db_ids = [data[1] for data in batch]
                    batch_tc_original = [data[2] for data in batch]
                
                    tokenized_inputs = text2sql_tokenizer(
                        batch_inputs, 
                        return_tensors="pt",
                        padding = "max_length",
                        max_length = 512,
                        truncation = True
                    )

                    encoder_input_ids = tokenized_inputs["input_ids"]
                    encoder_input_attention_mask = tokenized_inputs["attention_mask"]
                    if torch.cuda.is_available():
                        encoder_input_ids = encoder_input_ids.cuda()
                        encoder_input_attention_mask = encoder_input_attention_mask.cuda()

                    with torch.no_grad():
                        model_outputs = model.generate(
                            input_ids = encoder_input_ids,
                            attention_mask = encoder_input_attention_mask,
                            max_length = 256,
                            decoder_start_token_id = model.config.decoder_start_token_id,
                            num_beams = opt.num_beams,
                            num_return_sequences = opt.num_return_sequences
                        )
                    
                    model_outputs = model_outputs.view(len(batch_inputs), opt.num_return_sequences, model_outputs.shape[1])
                    if opt.target_type == "sql":
                        predict_sqls += decode_sqls(
                            opt.db_path, 
                            model_outputs, 
                            batch_db_ids, 
                            batch_inputs, 
                            text2sql_tokenizer, 
                            batch_tc_original
                        )
                    elif opt.target_type == "natsql":
                        predict_sqls += decode_natsqls(
                            opt.db_path, 
                            model_outputs, 
                            batch_db_ids, 
                            batch_inputs, 
                            text2sql_tokenizer, 
                            batch_tc_original, 
                            table_dict
                        )
                    else:
                        raise ValueError()
                    
                evaluator = EvaluateTool()
                evaluator.register_golds(opt.original_dev_filepath, opt.db_path)
                spider_metric_result = evaluator.evaluate(predict_sqls)
                print('exact_match score: {}'.format(spider_metric_result["exact_match"]))
                print('exec score: {}'.format(spider_metric_result["exec"]))
                
                os.makedirs(opt.save_path, exist_ok = True)
                model.save_pretrained(save_directory = opt.save_path + "/checkpoint-{}".format(train_step))
                text2sql_tokenizer.save_pretrained(save_directory = opt.save_path + "/checkpoint-{}".format(train_step))

                if len(saved_checkpoints) >= 2:
                    min_exec_score = 100
                    idx = None
                    for i, cur_checkpoint in enumerate(saved_checkpoints):
                        exec_score = json.load(open(cur_checkpoint+ '/eval_results.json'))["exec"]
                        if exec_score < min_exec_score:
                            min_exec_score = exec_score
                            idx = i

                    print(f'Deleting checkpoint {saved_checkpoints[idx]} with exec score {min_exec_score}')
                    shutil.rmtree(saved_checkpoints[idx], ignore_errors =True)
                    saved_checkpoints = [cur_ch for i, cur_ch in enumerate(saved_checkpoints) if i != idx]

                    # exec_scores1 = json.load(open(saved_checkpoints[0] + '/eval_results.json'))["exec"]
                    # exec_scores2 = json.load(open(saved_checkpoints[1] + '/eval_results.json'))["exec"]
                    # if exec_scores1 > exec_scores2:
                    #     shutil.rmtree(saved_checkpoints[1], ignore_errors =True)
                    #     saved_checkpoints = [saved_checkpoints[0]]
                    # else:
                    #     shutil.rmtree(saved_checkpoints[0], ignore_errors =True)
                    #     saved_checkpoints = [saved_checkpoints[1]]
                
                saved_checkpoints.append(opt.save_path + "/checkpoint-{}".format(train_step))
                assert len(saved_checkpoints) <= 2

                optimizer_scheduler = { 
                    'epoch': epoch,
                    'train_step': train_step,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    "saved_checkpoints": saved_checkpoints}
                torch.save(optimizer_scheduler, opt.save_path + "/checkpoint-{}/optimizer_scheduler.pth".format(train_step))

                rng_states = {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "cpu": torch.random.get_rng_state(),
                }
                if torch.cuda.is_available():
                    rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
                
                torch.save(rng_states, opt.save_path + "/checkpoint-{}/rng_states.pth".format(train_step))

                json.dump(spider_metric_result, open(opt.save_path + "/checkpoint-{}/eval_results.json".format(train_step), 'w'))
            
def _test(opt):
    set_seed(opt.seed)
    print(opt)

    import time
    start_time = time.time()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device
    
    if opt.target_type == "natsql":
        tables = json.load(open(opt.tables_for_natsql,'r'))
        table_dict = dict()
        for t in tables:
            table_dict[t["db_id"]] = t

    # initialize tokenizer
    tokenizer = T5TokenizerFast.from_pretrained(
        opt.save_path,
        add_prefix_space = True,
        cache_dir="/transformers_cache/"
    )
    
    if isinstance(tokenizer, T5TokenizerFast):
        tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])
    
    dev_dataset = Text2SQLDataset(
        dir_ = opt.dev_filepath,
        mode = opt.mode
    )

    dev_dataloder = DataLoader(
        dev_dataset, 
        batch_size = opt.batch_size, 
        shuffle = False,
        collate_fn = lambda x: x,
        drop_last = False
    )

    model_class = MT5ForConditionalGeneration if "mt5" in opt.save_path else T5ForConditionalGeneration

    # initialize model
    model = model_class.from_pretrained(opt.save_path, cache_dir="/transformers_cache/")
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    predict_sqls = []
    for batch in tqdm(dev_dataloder):
        batch_inputs = [data[0] for data in batch]
        batch_db_ids = [data[1] for data in batch]
        batch_tc_original = [data[2] for data in batch]

        tokenized_inputs = tokenizer(
            batch_inputs, 
            return_tensors="pt",
            padding = "max_length",
            max_length = 512,
            truncation = True
        )
        
        encoder_input_ids = tokenized_inputs["input_ids"]
        encoder_input_attention_mask = tokenized_inputs["attention_mask"]
        if torch.cuda.is_available():
            encoder_input_ids = encoder_input_ids.cuda()
            encoder_input_attention_mask = encoder_input_attention_mask.cuda()

        with torch.no_grad():
            model_outputs = model.generate(
                input_ids = encoder_input_ids,
                attention_mask = encoder_input_attention_mask,
                max_length = 256,
                decoder_start_token_id = model.config.decoder_start_token_id,
                num_beams = opt.num_beams,
                num_return_sequences = opt.num_return_sequences
            )

            model_outputs = model_outputs.view(len(batch_inputs), opt.num_return_sequences, model_outputs.shape[1])
            if opt.target_type == "sql":
                predict_sqls += decode_sqls(
                    opt.db_path, 
                    model_outputs, 
                    batch_db_ids, 
                    batch_inputs, 
                    tokenizer, 
                    batch_tc_original
                )
            elif opt.target_type == "natsql":
                predict_sqls += decode_natsqls(
                    opt.db_path, 
                    model_outputs, 
                    batch_db_ids, 
                    batch_inputs, 
                    tokenizer, 
                    batch_tc_original, 
                    table_dict
                )
            else:
                raise ValueError()
    
    new_dir = "/".join(opt.output.split("/")[:-1]).strip()
    if new_dir != "":
        os.makedirs(new_dir, exist_ok = True)
    
    # save results
    with open(opt.output, "w", encoding = 'utf-8') as f:
        for pred in predict_sqls:
            f.write(pred + "\n")
    
    end_time = time.time()
    print("Text-to-SQL inference spends {}s.".format(end_time-start_time))
    
    if opt.mode == "eval":
        # initialize evaluator
        evaluator = EvaluateTool()
        evaluator.register_golds(opt.original_dev_filepath, opt.db_path)
        spider_metric_result = evaluator.evaluate(predict_sqls, only_exec=True)
        # print('exact_match score: {}'.format(spider_metric_result["exact_match"]))
        print('exec score: {}'.format(spider_metric_result["exec"]))
    
        return spider_metric_result["exec"] #spider_metric_result["exact_match"], 
    
if __name__ == "__main__":
    opt = parse_option()
    if opt.mode in ["train"]:
        _train(opt)
    elif opt.mode in ["eval", "test"]:
        _test(opt)