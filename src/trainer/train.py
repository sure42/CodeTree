import sys
import os
import argparse

import codecs
import logging
import torch
import torch.nn as nn
from tqdm import tqdm  

from transformers import AutoTokenizer, T5EncoderModel, T5ForConditionalGeneration
# from transformers import RobertaTokenizer, RobertaForSequenceClassification

# os.path.dirname 返回给定文件路径的目录部分，也就是去掉最后一级文件名或者目录名之后的路径
# os.path.sep 替换直接写死的 '\\'
# os.getcwd() 工作目录    os.path.abspath(__file__)脚本的绝对路径
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
work_dir = os.path.dirname(src_dir)

model_dir = os.path.join(src_dir, "models")
sys.path.append(model_dir) 
dataloader_dir = os.path.join(src_dir, "dataloader")
sys.path.append(dataloader_dir)

# print(src_dir) # CoP\src
# print(work_dir) # CoP\src
# print(model_dir)
# print(dataloader_dir)

from dictionary import Dictionary
from MyDataloader import MyDataset
from example import CGModel


# 默认设置
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device_ids = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

def get_args():
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--dataset",  type=str, default='mix')
    argparser.add_argument("--model",  type=str, default='codet5-plus')
    argparser.add_argument("--train", action="store_false", help="use the training model (default: True) store_false")
    argparser.add_argument("--valid", action="store_false", help="use the validing model (default: False) store_true")

    argparser.add_argument("--idx", type=int, default=1)
    argparser.add_argument("--save_name", type=str, default="model_codet5p")

    argparser.add_argument("--batch_size", type=int, default=2)
    argparser.add_argument("--lr", type=float, default=5e-5)
    argparser.add_argument("--epoch", type=float, default=1)
    # argparser.add_argument("--epoch", type=float, default=3)
    argparser.add_argument("--dropout", type=float, default=0.01)
    argparser.add_argument("--weight_lm", type=float, default=0.1, help="the loss ratio of the model is encoded when the model is trained")
    # argparser.add_argument("--dropout", type=float, default=0.01)
    args = argparser.parse_args()
    return args


class Trainer():
    def __init__(self, args, tokenizer, hyper_parameter, emd_model, dictionary):
        self.args = args
        self.logger = self.log_config()
        self.logger.info(args)

        model_name = "{}_{}.pkl".format(self.args.save_name, self.args.idx)
        self.save_path = os.path.join(work_dir, 'data/models', model_name)

        train_dataset = None
        valid_dataset = None
        if args.train:
            train_file = os.path.join(work_dir, 'data/data', 'training_src.txt')
            train_dataset = MyDataset(train_file, dictionary)
            # print('len(train_labels)', len(train_file))
        if args.valid:
            valid_file = os.path.join(work_dir, 'data/data', 'validation_src.txt')
            valid_dataset = MyDataset(valid_file, dictionary)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        self.tokenizer = tokenizer
        self.model = CGModel(tokenizer, embed_dim=768, max_positions=1024,
            src_encoder_convolutions=hyper_parameter['src_encoder_convolutions'],
            ctx_encoder_convolutions=hyper_parameter['ctx_encoder_convolutions'],
            decoder_convolutions=hyper_parameter['decoder_convolutions'],
            dropout=hyper_parameter['dropout'], embed_model=emd_model,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        
    def log_config(self):
        # 创建一个logger
        logger = logging.getLogger('my_logger')
        logger.setLevel(logging.DEBUG)  # 设置日志级别

        # 创建一个handler，用于写入日志文件
        log_path = os.path.join(work_dir, "logs/output.log")
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)  # handler的级别也要设置，否则不会输出

        # 再创建一个handler，用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # 定义handler的输出格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') # 时间戳（%(asctime)s）、logger名称（%(name)s）、日志级别（%(levelname)s）以及日志信息本身（%(message)s）
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 给logger添加handler
        logger.addHandler(fh)
        logger.addHandler(ch)

        # # 记录一条日志
        # logger.info('This is an info message')
        # logger.debug('Debugging...')
        # logger.warning('Warning exists')
        # logger.error('An error occurred')
        # logger.critical('Critical error!')
        return logger
        
    def pad_at_rear(self, encodings):
        # 基于分词结果，填充
        # input_ids = encodings['input_ids']
        padding_tensor = torch.zeros(encodings['input_ids'].size(0), 1).to(device).long()
        encodings['input_ids'] = torch.cat((encodings['input_ids'], padding_tensor), dim=1) 
        # attention_mask = encodings['attention_mask']
        encodings['attention_mask'] = torch.cat((encodings['attention_mask'], padding_tensor), dim=1) 

        return encodings

    def tokenizer_with_pad(self, tokenizer, prevs, srcs, src_encodings_max_length):
        pads = []
        for src in srcs:
            src_ids = tokenizer.tokenize(src)
            # len_ids = src_encodings_max_length-len(src_ids)-2 # len_ids+2是对应的分词后的未填充的部分的长度 
            # pads.append(' '.join(['<pad>']*len_ids))
            len_ids = src_encodings_max_length-len(src_ids)-1 # len_ids+2是对应的分词后的未填充的部分的长度  这里尝试一下-1 再加一个</s>看看能不能有效？
            pads.append('</s>'+''.join(['<pad>']*len_ids))  # '<pad> <pad>' 这个字符串分词时有点问题，会划分出来一个255，但没有空格的话就还好
        # print(pads)
        src_with_prev_encodings = tokenizer([prev + '</s>' + src + pad for prev, src, pad in zip(prevs,srcs,pads)],
                                            return_tensors="pt", padding=True, truncation='longest_first').to(device)
        return src_with_prev_encodings

    def train(self):
        self.model.train()
        data_length = self.train_dataset.total_size
        batch_size = self.args.batch_size
        batch_num = data_length // batch_size

        ## 训练
        str_train = "-"*5 + "model train" + "-"*5
        self.logger.info(str_train) ## 训练数据数量，轮数 batchsize
        str_train = "train size: {}, epoch: {}, batch size: {}".format(data_length, self.args.epoch, batch_size)
        self.logger.info(str_train) 

        if data_length % batch_size != 0:
            batch_num += 1  # 如果有剩余的数据，则增加一个批次  
        weight_lm = self.args.weight_lm
        self.optimizer.zero_grad()
        oom = 0
        
        for i in range(self.args.epoch):
            desc = "epoch {}".format(i)
            val_loss, val_lm_loss = [], [], []
            for j in tqdm(range(batch_num), desc=desc):  
                # 计算当前批次的开始和结束索引  
                start_idx = j * batch_size
                end_idx = min(start_idx + batch_size, data_length)  
                self.optimizer.zero_grad()
                
                # 填充pad  <pad> 对应 0   但还是有一个问题，有没有可能连接后超过最大长度？被截断？
                src_encodings = self.tokenizer(self.train_dataset.src[start_idx:end_idx], return_tensors="pt", padding=True, truncation=True).to(device) # input_ids = tokenizer(train_dataset.src, return_tensors="pt")["input_ids"]
                ctx_encodings = self.tokenizer(self.train_dataset.ctx[start_idx:end_idx], return_tensors="pt", padding=True, truncation=True).to(device)
                tgt_encodings = self.tokenizer(self.train_dataset.tgt[start_idx:end_idx], return_tensors="pt", padding=True, truncation=True).to(device)
                
                src_encodings_max_length = src_encodings['input_ids'].size(1)
                src_with_prev_encodings = self.tokenizer_with_pad(self.tokenizer, self.train_dataset.prev[start_idx:end_idx], self.train_dataset.src[start_idx:end_idx], src_encodings_max_length)
                
                tgt_encodings_max_length = tgt_encodings['input_ids'].size(1)
                tgt_with_prev_encodings = self.tokenizer_with_pad(self.tokenizer, self.train_dataset.prev[start_idx:end_idx], self.train_dataset.tgt[start_idx:end_idx], tgt_encodings_max_length)
                try:
                    outputs = self.model(src_encodings, src_with_prev_encodings, ctx_encodings, tgt_encodings, tgt_with_prev_encodings, tgt_encodings)
                    logits, loss, lm_loss = outputs[:3]
                    loss = loss + weight_lm * lm_loss
                    loss.mean().backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 0.5, norm_type=2) # 用于梯度裁剪,防止梯度爆炸。将模型参数的梯度裁剪到指定的范数（norm）以下。包含模型参数的迭代器，裁剪的范数阈值，范数类型
                    self.optimizer.step()
                    # val_loss
                    # val_lm_loss
                except Exception as e:
                    oom += 1
                    torch.cuda.empty_cache()
        ## 训练
        str_train = "-"*5 + "model train" + "-"*5
        self.logger.info(str_train) ## 训练数据数量，轮数 batchsize
        str_train = "val loss: {}, val apr_loss: {}, oom: {}".format(data_length, self.args.epoch, oom)
        self.logger.info(str_train)

        self.save_model()

    def valid(self):

        self.load_model()

        self.model.eval()
        self.logger.info() ## 训练数据数量，轮数 batchsize
        data_length = self.valid_dataset.total_size
        batch_size = self.args.batch_size
        batch_num = data_length // batch_size
        if data_length % batch_size != 0:
            batch_num += 1  # 如果有剩余的数据，则增加一个批次  
        weight_lm = self.args.weight_lm
        self.optimizer.zero_grad()
        oom = 0
        
        self.logger.info(oom)
        for j in tqdm(range(batch_num)):  
            # 计算当前批次的开始和结束索引  
            start_idx = j * batch_size
            end_idx = min(start_idx + self.args.batch_size, data_length)  
            self.optimizer.zero_grad()
            
            # 填充pad  <pad> 对应 0   但还是有一个问题，有没有可能连接后超过最大长度？被截断？
            src_encodings = self.tokenizer(self.valid_dataset.src[start_idx:end_idx], return_tensors="pt", padding=True, truncation=True).to(device) # input_ids = tokenizer(train_dataset.src, return_tensors="pt")["input_ids"]
            ctx_encodings = self.tokenizer(self.valid_dataset.ctx[start_idx:end_idx], return_tensors="pt", padding=True, truncation=True).to(device)
            tgt_encodings = self.tokenizer(self.valid_dataset.tgt[start_idx:end_idx], return_tensors="pt", padding=True, truncation=True).to(device)
            
            src_encodings_max_length = src_encodings['input_ids'].size(1)
            src_with_prev_encodings = self.tokenizer_with_pad(self.tokenizer, self.valid_dataset.prev[start_idx:end_idx], self.valid_dataset.src[start_idx:end_idx], src_encodings_max_length)
            
            tgt_encodings_max_length = tgt_encodings['input_ids'].size(1)
            tgt_with_prev_encodings = self.tokenizer_with_pad(self.tokenizer, self.valid_dataset.prev[start_idx:end_idx], self.valid_dataset.tgt[start_idx:end_idx], tgt_encodings_max_length)
            try:
                outputs = self.model(src_encodings, src_with_prev_encodings, ctx_encodings, tgt_encodings, tgt_with_prev_encodings, tgt_encodings)
                logits, loss, lm_loss = outputs[:3]
                loss = loss + weight_lm * lm_loss
            except Exception as e:
                oom += 1
                torch.cuda.empty_cache()
        self.logger.info() ## 测试的结果
            
    def save_model(self):    
        self.logger.info() # 保存模型 路径
        torch.save(self.model.state_dict(), self.save_path)

    def load_model(self):
        self.logger.info() # 读取 路径
        self.model.load_state_dict(torch.load(self.save_path, map_location=device))

def main():
    args = get_args()
    

    # 一些配置    
    # 读取数据集    
    vocab_file = os.path.join(work_dir, 'data/vocabulary', 'vocabulary.txt')
    dictionary = Dictionary(vocab_file)
    model_name = "{}_{}.pkl".format(args.save_name, args.idx)
    
    
    if args.model == 'codet5-plus':
        model_path = os.path.join(work_dir, 'data/models', 'codet5p')
        tokenizer = AutoTokenizer.from_pretrained(model_path)  
        # This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
        # be encoded differently whether it is at the beginning of the sentence (without space) or not
    else:
        pass

    # emd_model = T5EncoderModel.from_pretrained(model_path)
    emd_model = T5ForConditionalGeneration.from_pretrained(model_path)
    # model = AutoModel.from_pretrained(model_path)
    hyper_parameter = {
        'src_encoder_convolutions': ((192, 5),) * 1,
        'ctx_encoder_convolutions': ((768, 5),) * 1,
        'decoder_convolutions': ((192, 5),) * 1,
        'dropout': 0.01,
    }
    trainer = Trainer(args, tokenizer, hyper_parameter, emd_model, dictionary)
    trainer.train()
    # trainer.valid()

if __name__ == "__main__":
    main()







            # print("src")
            # print(train_dataset.src[start_idx:end_idx])
            # print(src_encodings['input_ids'])
            # print(tokenizer.convert_ids_to_tokens(src_encodings['input_ids'].view(-1)))
            # # print("ctx")
            # # print(train_dataset.ctx[start_idx:end_idx])
            # # print(ctx_encodings['input_ids'])
            # print("tgt")
            # print(train_dataset.tgt[start_idx:end_idx])
            # print(tgt_encodings['input_ids'])
            # print(tokenizer.convert_ids_to_tokens(tgt_encodings['input_ids'].view(-1)))
            # # print("prev")
            # # print(train_dataset.prev[start_idx:end_idx])
            # print("src_with_prev")
            # print(tokenizer.convert_ids_to_tokens(src_with_prev_encodings['input_ids'].view(-1)))
            # print("tgt_with_prev")
            # print(tokenizer.convert_ids_to_tokens(tgt_with_prev_encodings['input_ids'].view(-1)))
            # print("-"*20)
            





    # # train_encodings = tokenizer(["请分析并修复以下代码中存在的安全漏洞：\n```java\n" + src for src in train_dataset.src], return_tensors="pt", padding=True, truncation=True)["input_ids"] # input_ids + attention_mask
    
    # 先 </s> 再 pad
    # src_encodings = tokenizer_with_pad(tokenizer, train_dataset.src[0]) 
    # ctx_encodings = tokenizer_with_pad(tokenizer, train_dataset.ctx[0])
    # src_with_prev_encodings = tokenizer_with_pad(tokenizer, train_dataset.prev[0]+train_dataset.src[0])
    # tgt_encodings = tokenizer_with_pad(tokenizer, train_dataset.tgt[0])
    # tgt_with_prev_encodings = tokenizer_with_pad(tokenizer, train_dataset.prev[0]+train_dataset.tgt[0])
    # # train_encodings = tokenizer(train_dataset.prev[0] + train_dataset.src[0], return_tensors="pt", padding=True, truncation=True) 
    # # train_encodings ：{'input_ids':tensor[], 'attention_mask':tensor[]}

    # unk_token_id = tokenizer.unk_token_id
    # oov_detected = train_encodings['input_ids'] == unk_token_id
    # any_oov = oov_detected.any()

    # if any_oov:
    #     print("存在OOV（未登录词）")
    # else:
    #     print("没有检测到OOV（未登录词）")
    # assert 1==2

    # input_ids = { # 这个思路完全不行
    #     'input_ids': train_encodings['input_ids'][0].unsqueeze(0),
    #     'attention_mask': train_encodings['attention_mask'][0].unsqueeze(0) 
    # }
    # # input_ids = train_encodings[0].unsqueeze(0)
    # if src_with_prev_encodings is not None: 
    #     src_encodings =   torch.tensor([[0]*len(train_dataset.prev[0]) + [1] * len(train_dataset.src[0])]) # 这里的长度计算的是字符的个数，而不是分词后的长度，现在仅有的思路是每一条都单独走一遍，或者prev_len + src的musk


    # with torch.no_grad():
    #     outputs = model(src_encodings, src_with_prev_encodings, ctx_encodings, tgt_encodings, tgt_with_prev_encodings, tgt_encodings)
    # features = outputs.last_hidden_state[:, 0, :]  # 提取[CLS] token的特征

    # code - tokenizer - tensor - dataset - dataloader(自动有) - input/mask - model - output