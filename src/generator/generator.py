import codecs
import torch
import sys
import os
import javalang
import argparse
import time
import configparser

from transformers import AutoTokenizer, T5EncoderModel, T5ForConditionalGeneration


src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # ..\CodeTree\src
work_dir = os.path.dirname(src_dir) # ..\CodeTree\
sys.path.append(src_dir) 
sys.path.append(work_dir) 
print(src_dir)
print(work_dir)

# model_dir = os.path.join(src_dir, "models")
# sys.path.append(model_dir) 
# dataloader_dir = os.path.join(src_dir, "dataloader")
# sys.path.append(dataloader_dir)

from dataloader.dictionary import Dictionary
from dataloader.dataloader import MyDataset
from dataloader.identifier_data_loader import IdentifierDataLoader
from models.CGModel import CGModel
from utils import utils
from beamsearch import BeamSearch


# 默认设置
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device_ids = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
config = utils.read_config(work_dir)

def check_parse(tmp_str):
    flag_src = True
    if '/ /' in tmp_str or '\\' in tmp_str:
        flag_src = False
        return flag_src
    tmp_str = tokenization.token2statement(tmp_str.split(' '), ['3'], ['tmp_str'])
    tmp_str  = tmp_str[0].replace('@@ ', '')
    try:
        tree = javalang.parse.parse(tmp_str)# 前+h+后  javalangparser. Parsebxception
    except javalang.parser.JavaSyntaxError as e:#  javalang.parser.ParseException
        flag_src = False
    return flag_src

class Generator():
    def __init__(self, tokenizer, emd_model, dictionary):
        # 日志
        self.logger = utils.log_config(work_dir, config)
        self.logger.info([config.items(item) for item in config.sections()])

        # 地址设置
        model_name = "{}_{}.pkl".format(config['model']['save_name'], config['train']['model_idx'])
        self.model_path = os.path.join(work_dir, config['file.path']['models_path'], model_name)
        self.identifier_token_file = os.path.join(work_dir, config['file.path']['models_path'], model_name)
        self.identifier_txt_file = os.path.join(work_dir, config['file.path']['models_path'], model_name)

        self.tokenizer = tokenizer
        self.dictionary = dictionary

        self.model = CGModel(
            dictionary=dictionary, embed_dim=config['embed_dim'],
            max_positions=config['max_positions'],
            src_encoder_convolutions=config['src_encoder_convolutions'],
            ctx_encoder_convolutions=config['ctx_encoder_convolutions'],
            decoder_convolutions=config['decoder_convolutions'],
            dropout=0, embed_model=emd_model,
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(config['DEFAULT']['lr']))
        self.load()
        # self.beamsearch = BeamSearch(self.model, dictionary, config['beam_size'])
        # print(self.model, beam_size)

    def load(self):
        '''
        加载模型和数据
        '''
        # 模型加载 
        loaded = torch.load(
            self.model_path, map_location='cpu'
        )
        # 加载GPT模型配置
        config = loaded['config']
        self.model.load_state_dict(torch.load(self.model_path, map_location=device))
        # self.model.load_state_dict(loaded['model'])
        self.identifier_loader = IdentifierDataLoader(
            self.dictionary, self.identifier_token_file, self.identifier_txt_file
        )
        bug_file = os.path.join(work_dir, config['file.path']['bug_file_path'])
        self.bug_dataset = MyDataset(bug_file, self.dictionary, self.identifier_loader)

    def load_model(self):
        self.logger.info("model load path: {}".format(self.save_path)) # 读取 路径
        self.model.load_state_dict(torch.load(self.save_path, map_location=device))

    # def generate(self, output_path, output_file_parse, model_id):

    #     wp = codecs.open(output_path, 'w', 'utf-8')
    #     wp_parse = codecs.open(output_file_parse, 'w', 'utf-8')
    #     self.data_loader.load_data(0, self.data_loader.total_size)
    #     for i in range(self.data_loader.total_size):
    #         start_time_ge = time.time()
    #         print(i, '/', self.data_loader.total_size)
    #         data = self.data_loader.dataset[i]
    #         # try:
    #         self.beamsearch.beam_size = self.beam_size
    #         sample = self.data_loader.dataset.collater([data])
    #         with torch.no_grad():
    #             # if isinstance(self.model, GPTCoNuTModel):
    #             #     hypothesis = self.beamsearch.generate_gpt_conut(sample)
    #             # elif isinstance(self.model, GPTFConvModel):
    #             #     hypothesis = self.beamsearch.generate_gpt_fconv(sample)
    #             if isinstance(self.model, GPTCoNuTModel):
    #                 hypothesis = self.beamsearch.generate_gpt_conut_with_detect(sample, model_id)

    #         # except Exception as e:# 这个格式可以参考
    #         #    print(e)
    #         #    continue

    #         id = str(sample['id'].item())
    #         wp.write('S-{}\t'.format(id))# 源代码
    #         wp.write(self.dictionary.string(data['source']) + '\n')
    #         wp.write('T-{}\t'.format(id))# 目标补丁
    #         wp.write(self.dictionary.string(data['target']) + '\n')
    #         for h in hypothesis:# 生成的候选
    #             wp.write('H-{}\t{}\t'.format(id, str(h['final_score'])))
    #             wp.write(self.dictionary.string(h['hypo']) + '\n')
    #             wp.write('P-{}\t'.format(id))
    #             wp.write(' '.join(str(round(s.item(), 4)) for s in h['score']) + '\n')# round小数位数

    #         wp_parse.write('S-{}\t'.format(id))# 源代码
    #         wp_parse.write(self.dictionary.string(data['source']) + '\n')
    #         wp_parse.write('T-{}\t'.format(id))# 目标补丁
    #         wp_parse.write(self.dictionary.string(data['target']) + '\n')
    #         wp_parse.write('skip_flag-{}\t'.format(id))
    #         wp_parse.write(str(skip_flag) + '\n')
    #         for h in hypothesis_parse:# 生成的候选
    #             wp_parse.write('H-{}\t{}\t'.format(id, str(h['final_score'])))
    #             wp_parse.write(self.dictionary.string(h['hypo']) + '\n')
    #             wp_parse.write('P-{}\t'.format(id))
    #             wp_parse.write(' '.join(str(round(s.item(), 4)) for s in h['score']) + '\n')# round小数位数
            
    #         end_time_ge = time.time()
    #         generate_time = end_time_ge - start_time_ge
    #         print('generate test time:',generate_time)
    #         time_file = './data/models/time_ge.txt'
    #         with open(time_file, 'a+') as f:
    #             info = 'generate : gpt:{}, detector:{}, id:{}, time:{}s, dir:{}\n'.\
    #                 format('1', '1', generate_time, id, output_file )
    #             f.write(info)
    #         wp_parse.write('time-{}\t'.format(id))
    #         wp_parse.write(str(generate_time) + '\n')
    #         wp.write('time-{}\t'.format(id))
    #         wp.write(str(generate_time) + '\n')
    #     with open(time_file, 'a+') as f:
    #         info = '****************************************\n'
    #         f.write(info)                          
    #     wp.close()
    #     wp_parse.close()
    def generate(self):
        self.logger.info("-"*5 + "patches generation start" + "-"*5) 


    def patches_save():
        # 保存路径的生成

        # 文件写入
        pass
        
def generate(vocab_file, model_file, input_file, identifier_txt_file, identifier_token_file, output_file, output_file_parse,  beam_size, model_id):

    generator.generate(output_file, output_file_parse, model_id)


def main(): 

    # 1. read config
    vocab_file = os.path.join(work_dir, config['file.path']['vocab_path'])
    dictionary = Dictionary(vocab_file, min_cnt=0)

    # 2. create model
    if config['generat']['model'] == 'codet5-plus':
        model_path = os.path.join(work_dir, config['file.path']['models_path'], 'codet5p')
        tokenizer = AutoTokenizer.from_pretrained(model_path)  
        # This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
        # be encoded differently whether it is at the beginning of the sentence (without space) or not
    else:
        pass

    emd_model = T5ForConditionalGeneration.from_pretrained(model_path)


    # 2.1 生成类
    generator = Generator(tokenizer, emd_model, dictionary)
    # 2.2 根据选择的模型进行生成
    patches = generator.generate()
    # 2.3 保存结果
    # 类.save 

        # 2.1. generation


        # 2.2. save

    # model_file = GENERATOR_DIR + '..\..\data\models\gpt_fconv_1.pt'
    # output_file = GENERATOR_DIR + '..\..\data\patches\gpt_fconv_2.txt'
    # generate_gpt_fconv(vocab_file, model_file, input_file, identifier_txt_file, identifier_token_file, output_file, beam_size)


if __name__ == "__main__":
    main()