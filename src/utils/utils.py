import sys
import os
import argparse
import configparser
import torch
import logging

def read_config(work_dir):
    config = configparser.ConfigParser()    
    config.read(os.path.join(work_dir, './config.ini'), encoding='utf-8')
    return config


def get_args():
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')

    # base
    argparser.add_argument("--dataset",  type=str, choices=['quixbugs', 'defects4j'], default='quixbugs')
    argparser.add_argument("--model",  type=str, default='codet5-plus')
    argparser.add_argument("--train", action="store_true", help="use the training model (default: True) store_false")
    argparser.add_argument("--valid", action="store_false", help="use the validing model (default: False) store_true")

    # model
    argparser.add_argument("--idx", type=int, default=1)
    argparser.add_argument("--save_name", type=str, default="model_codet5p")

    # genetration
    argparser.add_argument("--batch_size", type=int, default=2)
    argparser.add_argument("--lr", type=float, default=5e-5)
    argparser.add_argument("--beam_size", type=float, default=3)
    args = argparser.parse_args()
    return args

def ConvLayers(noder):
    convLayers = tuple(((int(noder['in_channels']),int(noder['out_channels'])),)*int(noder['kernel_size']))
    return convLayers

def log_config(work_dir, config):
    # 创建一个loggerq
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)  # 设置日志级别

    # 创建一个handler，用于写入日志文件
    log_path = os.path.join(work_dir, config['file.path']['log_path'])
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