
import torch
import torch.nn as nn
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class XX_trainer():
    def __init__(self, train_loader, valid_loader, dictionary):
        pass

    def train(self, ):
        self.model.train()
        pass


    def valid(self, ): # 训练结束后调用的
        self.model.eval()
        pass

    def test(self, ): # 单独调用
        self.model.eval()
        pass

    def save(self, ): # 单独调用
        self.model.eval()
        pass


def main():
    # 这里需要看一下获取参数的代码

    # 记录log，参考pwws的结构，生成一个随机编码

    # 加载模型

    # 加载数据集==判断是哪个数据，进行对应的操作？这个可能不需要

    # 定义训练类

    # 根据参数判断具体的操作
    runmodel = "False"
    if runmodel == "train":
        pass
    elif runmodel == "train":
        pass
    

    # dataloader

    # 




    pass

if __name__ == '__main__':

    # test_src()
    main()