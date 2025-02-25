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

src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # ..\CodeTree\src
work_dir = os.path.dirname(src_dir) # ..\CodeTree\
sys.path.append(src_dir) 
sys.path.append(work_dir) 

from dataloader.dictionary import Dictionary
from dataloader.dataloader import MyDataset
from models.CGModel import CGModel
from utils import utils

# 默认设置
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device_ids = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

config = utils.read_config(work_dir)

class Trainer():
    def __init__(self, tokenizer, emd_model, dictionary):
        self.logger = utils.log_config(work_dir, config)
        self.logger.info([config.items(item) for item in config.sections()])

        train_file = os.path.join(work_dir, config['file.path']['train_file_path'])
        train_dataset = MyDataset(train_file, dictionary)
        # print('len(train_labels)', len(train_file))
        valid_file = os.path.join(work_dir, config['file.path']['valid_file_path'])
        valid_dataset = MyDataset(valid_file, dictionary)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        

        self.tokenizer = tokenizer

        model_name = "{}_{}.pkl".format(config['model']['save_name'], config['train']['model_idx'])
        self.save_path = os.path.join(work_dir, config['file.path']['models_path'], model_name)
        
        self.model = CGModel(tokenizer, embed_dim=int(config['model']['embed_dim']), max_positions=int(config['model']['max_positions']),
            src_encoder_convolutions=utils.ConvLayers(config['model.src_encoder']),
            ctx_encoder_convolutions=utils.ConvLayers(config['model.ctx_encoder']),
            decoder_convolutions=utils.ConvLayers(config['model.decoder']),
            dropout=float(config['DEFAULT']['dropout']), 
            embed_model=emd_model,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(config['DEFAULT']['lr']))
        
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
        data_length = 10
        batch_size = int(config['DEFAULT']['batch_size'])
        epoch = int(config['DEFAULT']['epoch'])
        weight_lm = float(config['DEFAULT']['weight_lm'])

        batch_num = data_length // batch_size
        if data_length % batch_size != 0:
            batch_num += 1  # 如果有剩余的数据，则增加一个批次  
        # batch_num = 12
        ## 训练
        self.logger.info("-"*5 + "model train start" + "-"*5) ## 训练数据数量，轮数 batchsize
        self.logger.info("train size: {}, epoch: {}, batch size: {}".format(data_length, epoch, batch_size)) 

        self.optimizer.zero_grad()
        oom_all = 0
        avg_val_loss, avg_val_lm_loss = [], []
        for i in range(epoch):
            desc = "epoch {}".format(i)
            oom = 0
            val_loss, val_lm_loss = [], []
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
                    val_loss.append(loss)
                    val_lm_loss.append(lm_loss)
                except Exception as e:
                    oom += 1
                    torch.cuda.empty_cache()
            avg_val_loss.append(sum(val_loss)/len(val_loss))
            avg_val_lm_loss.append(sum(val_lm_loss)/len(val_lm_loss))
            self.logger.info("epoch {} val loss: {}, val apr_loss: {}, oom: {}".format(i, sum(val_loss)/len(val_loss), sum(val_lm_loss)/len(val_lm_loss), oom))
            oom_all += oom
        ## 训练
        self.logger.info("-"*5 + "model train finished" + "-"*5) ## 训练数据数量，轮数 batchsize        
        self.logger.info("last val loss: {}, last val apr_loss: {}".format(avg_val_loss[-1], avg_val_lm_loss[-1]))
        self.logger.info("avg val loss: {}, avg val apr_loss: {}, oom: {}".format(sum(avg_val_loss)/len(avg_val_loss), sum(avg_val_lm_loss)/len(avg_val_lm_loss), oom_all))
        self.save_model()

    def valid(self):
        self.load_model()
        self.model.eval()

        data_length = self.valid_dataset.total_size
        # data_length = 10
        batch_size = int(config['DEFAULT']['batch_size'])
        weight_lm = float(config['DEFAULT']['weight_lm'])
        
        batch_num = data_length // batch_size
        if data_length % batch_size != 0:
            batch_num += 1  # 如果有剩余的数据，则增加一个批次  
        self.logger.info("-"*5 + "model valid start" + "-"*5) ## 训练数据数量，轮数 batchsize
        self.logger.info("valid size: {}, batch size: {}".format(data_length, batch_size)) ## 训练数据数量，轮数 batchsize

        self.optimizer.zero_grad()
        oom = 0
        val_loss, val_lm_loss = [], []
        for j in tqdm(range(batch_num)):  
            # 计算当前批次的开始和结束索引  
            start_idx = j * batch_size
            end_idx = min(start_idx + batch_size, data_length)  
            self.optimizer.zero_grad()

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
                val_loss.append(loss)
                val_lm_loss.append(lm_loss)
            except Exception as e:
                oom += 1
                torch.cuda.empty_cache()
        self.logger.info("-"*5 + "model valid finished" + "-"*5) ## 训练数据数量，轮数 batchsize        
        self.logger.info(" val loss: {}, val apr_loss: {}, oom: {}".format(sum(val_loss)/len(val_loss), sum(val_lm_loss)/len(val_lm_loss), oom))
    
    def save_model(self):    
        self.logger.info("model save path: {}".format(self.save_path)) # 保存模型 路径
        torch.save(self.model.state_dict(), self.save_path)

    def load_model(self):
        self.logger.info("model load path: {}".format(self.save_path)) # 读取 路径
        self.model.load_state_dict(torch.load(self.save_path, map_location=device))

def main():
    
    # 一些配置    
    # 字典不需要继续保存，所以放到外面应该也可以？
    vocab_file = os.path.join(work_dir, config['file.path']['vocab_path'])
    dictionary = Dictionary(vocab_file)
    
    if config['train']['model'] == 'codet5-plus':
        model_path = os.path.join(work_dir, config['file.path']['models_path'], 'codet5p')
        tokenizer = AutoTokenizer.from_pretrained(model_path)  
        # This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
        # be encoded differently whether it is at the beginning of the sentence (without space) or not
    else:
        pass

    # emd_model = T5EncoderModel.from_pretrained(model_path)
    emd_model = T5ForConditionalGeneration.from_pretrained(model_path)
    # model = AutoModel.from_pretrained(model_path)

    trainer = Trainer(tokenizer, emd_model, dictionary)
    
    if config['train']['train']:
        trainer.train()
    if config['train']['valid']:
        trainer.valid()

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