import codecs # 标准 Python 编解码器
import torch
# from gpt_conut_dataset import GPTCoNuTDataset


def find_sublist(ctx, src):
    start = -1
    for i in range(0, len(ctx) - len(src) + 1):
        if ctx[i: i + len(src)] == src:
            start = i
            break
    return start

def read_data(datafile, dictionary):
    fp = codecs.open(datafile, 'r', 'utf-8')
    src_list = []
    tgt_list = []
    with open(datafile, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            if line.strip() == '':
                continue
            
            src, tgt = line.split('\t')
            src = src.strip().split() + [dictionary.eos_word]
            tgt = tgt.strip().split() + [dictionary.eos_word]

            
            # 取漏洞行和函数
            for i in range(len(src)):
                if src_item[i] == dictionary.ctx_word:
                    ctx_index = i
            ctx_item = src_item[ctx_index + 1:]
            src_item = src_item[: ctx_index]

            # 按漏洞行将函数进行划分 ctx = prev + src + rear
            start = find_sublist(ctx_item, src_item)
            if start <= 0:
                ctx_item = [dictionary.eos_word] + ctx_item
                start = 1
            assert start > 0
            prev_context = ctx_item[: start]
            rear_context = ctx_item[start + len(src_item):]


            src_list.append(src)
            tgt_list.append(tgt)
            
    return src_list, tgt_list

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, datafile, dictionary, identifier_loader=None):
        self.datafile = datafile # file_path
        self.dictionary = dictionary
        self.total_size = 0

        self.src = []
        self.tgt = []
        self.ctx = [] # ctx = prev + src + rear
        self.prev = []
        self.rear = []

        # self.src_tokens = []
        # self.tgt_tokens = []

        self.identifier_loader = identifier_loader
        self.dataset = None

        self.load_data() # 读数据，src = 漏洞行 + <CTX> + 完整代码
        self.ctx_split() # 将上面这个src进行划分 

    def get_total_size(self): # 统计数据条数
        # fp = codecs.open(self.datafile, 'r', 'utf-8')
        # self.total_size = len(fp.readlines())
        # fp.close()
        self.total_size = len(self.src)
        assert self.total_size == len(self.tgt)

    def reinitialize(self):
        self.src = []
        self.tgt = []
        self.dataset = None

    def load_data(self):
        self.reinitialize()
        fp = codecs.open(self.datafile, 'r', 'utf-8')
        while True:
            line = fp.readline()
            if not line:
                break
            if line.strip() == '':
                continue
            
            src, tgt = line.split('\t')
            src = src.strip().split()
            tgt = tgt.strip().split()
            self.src.append(' '.join(src))
            self.tgt.append(' '.join(tgt))

            # src_tokens = self.dictionary.index(src)# 切分好的字符表示对应的编号，如'int a=1' src里的token分别是int a = 1 四部分对应的编号
            # tgt_tokens = self.dictionary.index(tgt)
            # src_tokens = src_tokens + [self.dictionary.eos()]# 序列结束
            # tgt_tokens = tgt_tokens + [self.dictionary.eos()]
            # self.src_tokens.append(src_tokens)
            # self.tgt_tokens.append(tgt_tokens)
        self.get_total_size()

    def ctx_split(self):
        src = []
        ctx = []
        prev = []
        rear = []
        for j in range(self.total_size):
            ctx_index = 0
            line_list = self.src[j].strip().split()
            for i in range(len(line_list)): # 划分出 <CTX>
                if line_list[i] == self.dictionary.ctx_word:
                    ctx_index = i
            ctx_item = line_list[ctx_index + 1:]
            src_item = line_list[: ctx_index]

            start = find_sublist(ctx_item, src_item) # 对CTX做划分，
            if start <= 0:
                ctx_item = [self.dictionary.eos_word] + ctx_item
                start = 1
            assert start > 0

            prev_context = ctx_item[: start]
            behind_context = ctx_item[start + len(src_item):]

            src.append(' '.join(src_item))
            ctx.append(' '.join(ctx_item)) # ctx = prev + src +rear
            prev.append(' '.join(prev_context)) 
            rear.append(' '.join(behind_context))
        self.src = src
        self.ctx = ctx
        self.prev = prev
        self.rear = rear

    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, index):

        src_item, tgt_item = self.src[index], self.tgt[index]
        ctx_index = 0
        for i in range(len(src_item)):
            if src_item[i] == self.dictionary.ctx():
                ctx_index = i
        ctx_item = src_item[ctx_index + 1:]
        src_item = src_item[: ctx_index]

        start = find_sublist(ctx_item, src_item)
        if start <= 0:
            ctx_item = [self.dictionary.eos()] + ctx_item
            start = 1
        assert start > 0
        prev_context = ctx_item[: start]
        behind_context = ctx_item[start + len(src_item):]

        item = {"id": index}
        item["src"] = src_item
        # item["src_len"] = get_statement_length(len(src_item))
        item["ctx"] = ctx_item
        item["tgt"] = tgt_item
        item["prev"] = prev_context
        item["rear"] = behind_context
        # item["identifier"] = self.identifier[index] if self.identifier is not None else None

        return item

class mdoelDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)