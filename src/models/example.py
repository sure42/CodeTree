import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import T5EncoderModel, T5ForConditionalGeneration

from conv_tbc import ConvTBC
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def find_subsequence_indices(src, src_with_prev):
    # 这里需要标注出src+1，这里的1是指生成的下一个单词   需要测试一下放到取编码时会不会出问题

    # 例子：src是 '<s>if ( model!= null )</s>' 长度为8 src[j, 1:-1]是'if ( model!= null )' 长度为6
    #  
    # 
    # 获取 a 和 b 的维度
    num_src_rows = src.size(0)
    num_src_with_prev_rows= src_with_prev.size(0)
    assert num_src_rows == num_src_with_prev_rows
    # 初始化标注张量，形状与 b 相同
    musk = torch.zeros_like(src_with_prev, dtype=torch.int32)

    # CodeT5用使用 <s> 作为序列的起始符号，</s> 作为序列的结束符号,对应的id分别是1和2
    # src[0][1:src[0].nonzero()[:, 0].max()+1] 所有非零元素的索引 -> 索引中最大的一个 -> 切片取得从第二个元素开始除了末尾连续0以外的部分
    # 遍历 b 的每一行
    for i in range(num_src_with_prev_rows):
        # 遍历 a 的每一行
        musk[i, 0] = 1  
        len_src_with_prev = len(src_with_prev[i])
        # musk[i, 0] = musk[i, -1] = 1  
        # for j in range(num_src_rows):
            # 遍历 b[i] 中的可能起始位置
        # filtered_src = src[i][1:src[i].nonzero()[:, 0].max()+1] # 排除首尾的标识
        filtered_src = src[i, 1:] # 排除首部的标识，尾不删除  # 这里还是有问题，现在遇到的就是src填充后的0 比 src_with_prev 的要多   k=6时有这个问题
        len_filtered_src = len(filtered_src)
        for k in range(len_src_with_prev - len_filtered_src+1):
            # 检查从位置 k 开始的子序列是否与 a[j] 匹配
            subseq = src_with_prev[i, k:k+len_filtered_src]
            # print(subseq)
            if torch.equal(subseq, filtered_src):
                musk[i, k-1:k+len_filtered_src] = 1    # +1 指生成的位置
                break
    return musk


class CGModel(nn.Module):
    def __init__(
            self, dictionary,  embed_dim=768, max_positions=1024,
            src_encoder_convolutions=((192, 5),) * 5,
            ctx_encoder_convolutions=((768, 5),) * 7,
            decoder_convolutions=((192, 5),) * 5,
            dropout=0.1, embed_model=None,
    ):
        super(CGModel, self).__init__()

        # 模型结构

        self.embed_dim = embed_dim
        self.max_positions = max_positions
        self.dictionary = dictionary
        self.src_encoder_convolutions = src_encoder_convolutions
        self.ctx_encoder_convolutions = ctx_encoder_convolutions
        self.decoder_convolutions = decoder_convolutions
        self.embed_model = embed_model
        # 编码器
        # self.encoder = T5EncoderModel.from_pretrained(model_path)

        # self.decoder = DecoderNode()

        self.encoder = Encoder(
            dictionary, embed_dim, max_positions,
            src_encoder_convolutions, ctx_encoder_convolutions, dropout
        )
        self.decoder = Decoder(# 改成CNN
            dictionary, embed_dim, max_positions,
            decoder_convolutions, dropout,
        )
        # 解码器
        pass

    def config(self):
        info = dict()

        info[""] = "XX"
        # 定义一些参数
        return info

    def forward(self, src_encodings, src_with_prev_encodings, ctx_encodings,
                tgt_encodings, tgt_with_prev_encodings=None, labels=None):
        a = 1
        encoder_out = self.encoder(
            src_encodings, src_with_prev_encodings,
            ctx_encodings, share_embed_model=self.embed_model
        )
        decoder_out = self.decoder(
            tgt_encodings, encoder_out,
            tgt_with_prev_encodings,
            share_embed_model=self.embed_model,
            output_lm_logits=True,
        )

        if labels is not None:
            logits, lm_logits = decoder_out
            # logits, avg_attn_scores, lm_logits = decoder_out
            loss_fct = nn.NLLLoss()# 损失函数
            
            shift_logits = logits[..., :-1, :].contiguous() # 1 x 26 x 32100 这里对应 prev_tokens_index
            # filtered_src = labels['input_ids'][i][1:labels['input_ids'][i].nonzero()[:, 0].max()+1] # 排除首尾的标识
            shift_labels = labels['input_ids'][..., :].contiguous() # 1 x 27 这里应该是 1 x 26    这里可能跟前后的标识有关系？在对应一下CURE的内容
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            shift_lm_logits = lm_logits[..., :-2, :].contiguous() # 这里的 -2 和下面的 1:-1 没有理解为什么
            shift_lm_labels = tgt_with_prev_encodings['input_ids'][:, 1:-1].contiguous()
            lm_loss = loss_fct(shift_lm_logits.view(-1, shift_lm_logits.size(-1)), shift_lm_labels.view(-1))

            # decoder_out = (logits, avg_attn_scores, loss, lm_loss)
            decoder_out = (logits, loss, lm_loss)
        return decoder_out

class Encoder(nn.Module):# 整体的encoder
    def __init__(
            self, dictionary, embed_dim=768, max_positions=1024,
            src_convolutions=((192, 5),) * 5, ctx_convolutions=((768, 5),) * 7,
            dropout=0.1,
    ):
        super(Encoder, self).__init__()
        self.src_encoder = EncoderNode(
            dictionary, embed_dim, max_positions, src_convolutions, dropout,
        )
        self.context_encoder = EncoderNode(
            dictionary, embed_dim, max_positions, ctx_convolutions, dropout,
        )

    def forward(self, src_encodings, src_with_prev_encodings, ctx_encodings, share_embed_model=None):
        # encode the buggy lines
        src_output = self.src_encoder.forward(
            src_encodings,
            src_with_prev_encodings=src_with_prev_encodings,
            share_embed_model=share_embed_model,
        )
        # encode the context lines
        ctx_output = self.context_encoder.forward(
            ctx_encodings,
            src_with_prev_encodings=None,
            share_embed_model=share_embed_model,
        )
        if src_output['encoder_padding_mask'] is None or ctx_output['encoder_padding_mask'] is None:
            encoder_padding_mask = None
        else:
            encoder_padding_mask = torch.cat(
                [src_output['encoder_padding_mask'],
                 ctx_output['encoder_padding_mask']], 1
            )
        return {
            'src_tokens': torch.cat([src_output['src_tokens'], ctx_output['src_tokens']], 1),
            'encoder_out': (torch.cat([src_output['encoder_out'][0], ctx_output['encoder_out'][0]],1),
                            torch.cat([src_output['encoder_out'][1], ctx_output['encoder_out'][1]],1)),
            'encoder_padding_mask': encoder_padding_mask
        }


class EncoderNode(nn.Module):# encoder内部的编码
    def __init__(
            self, dictionary, embed_dim=768, max_positions=1024, 
            convolutions=((192, 5),) * 5, dropout=0.1,
    ):
        super(EncoderNode, self).__init__()
        self.dictionary = dictionary
        self.dropout = dropout

        self.embed_norm = nn.LayerNorm(embed_dim)

        convolutions = extend_conv_spec(convolutions)
        in_channels = convolutions[0][0]
        self.fc1 = linear(embed_dim, in_channels, dropout=dropout)
        self.convolutions = nn.ModuleList()
        self.attentions = nn.ModuleList()

        layer_in_channels = [in_channels]
        for i, (out_channels, kernel_size, residual) in enumerate(convolutions):
            if kernel_size % 2 == 1:
                padding = kernel_size // 2# “//”在Python中表示整数除法，返回不大于结果的一个最大的整数，即除法结果向下取整
            else:
                padding = 0
            self.convolutions.append(
                convtbc(in_channels, out_channels * 2, kernel_size,
                        dropout=dropout, padding=padding)
            )
            in_channels = out_channels
            layer_in_channels.append(out_channels)
        self.fc2 = linear(in_channels, embed_dim)

    def forward(self, src_encodings, src_with_prev_encodings=None, share_embed_model=None):
        assert share_embed_model is not None
        
        if src_with_prev_encodings is not None:
            outputs = share_embed_model(input_ids = src_with_prev_encodings['input_ids'],
                                        attention_mask  = src_with_prev_encodings['attention_mask'],
                                        labels = src_with_prev_encodings['input_ids'],)
            # outputs2 = share_embed_model(input_ids = src_with_prev_encodings['input_ids'],
            #                             attention_mask  = src_with_prev_encodings['attention_mask'],
            #                             labels = src_encodings['input_ids'],)
            # assert (outputs.encoder_last_hidden_state == outputs2.encoder_last_hidden_state).all() 这里是相等的

            src_with_prev_ids = src_with_prev_encodings['input_ids']     
            embed = outputs.encoder_last_hidden_state 

            bsz = embed.size(0) # 这里含义是一共 B 条数据
            embed = embed.view(-1, embed.size(-1))      # B x context_src, H 将所有的数据拼成一条

            src_ids = find_subsequence_indices(src_encodings['input_ids'], src_with_prev_encodings['input_ids'])
            mask = src_ids.view(-1)                  # 将张量重塑为一维 B, context_src 变为 B x context_src  这里的src_token并不是直观上理解的src序列，而是mask，指明那一部分是prev，那一部分是src
            mask = mask.eq(1) # 取出bug行  得到bug行对应的位置，这个需要注意的是src_token并不是直观上理解的
            # torch.eq(input, other, *, out=None) 对两个张量Tensor进行逐元素的比较，若相同位置的两个元素相同，则返回True
            
            x = embed[mask, :]          # B x src, H 取出src部分的编码
            x = x.view(bsz, -1, x.size(1))      # B, src, H  恢复形状

            src_with_prev_ids = src_with_prev_ids.view(-1)  # B x context_src
            src_tokens = src_with_prev_ids[mask]      # B x src  这里的src_tokens又恢复成src序列了
            src_tokens = src_tokens.view(bsz, -1)               # B, src
            # 130行到这里都是调整size
        else:# 如果只有bug行，所以不用再提取了 在这个支线中，src_tokens应该表示的是src序列
            outputs = share_embed_model(input_ids = src_encodings['input_ids'],
                                        attention_mask  = src_encodings['attention_mask'],
                                        labels = src_encodings['input_ids'],)        
            x = outputs.encoder_last_hidden_state 
            src_tokens = src_encodings['input_ids']   

        # normalize the embedding of buggy/context lines
        x = self.embed_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        input_embedding = x

        # project to size of convolution / linear layer

        x = self.fc1(x)

        # used to mask padding in input
        encoder_padding_mask = src_tokens.eq(0).t()  # -> T x B

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # residuals = [x]
        #这里才开始本模型的训练
        # temporal convolutions / convolutional layer / fconv part 时域卷积/卷积层/ fconv部分
        # for conv, attention, res_layer, norm in zip(self.convolutions, self.attentions,
        #                                             self.residuals, self.norms):
        for conv in self.convolutions:
            # if res_layer > 0:
            #     residual = residuals[-res_layer]
            # else:
            #     residual = None

            if encoder_padding_mask is not None:
                x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)
            target_embedding = x.transpose(0, 1)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if conv.kernel_size[0] % 2 == 1:
                # padding is implicit（含蓄的，未言明的） in the conv
                x = conv(x)
            else:
                padding_l = (conv.kernel_size[0] - 1) // 2
                padding_r = conv.kernel_size[0] // 2
                x = F.pad(x, (0, 0, 0, 0, padding_l, padding_r))
                x = conv(x)
            x = F.glu(x, dim=2)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # project back to size of embedding / linear layer
        x = self.fc2(x)

        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.t()  # -> B x T
            x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)# unsqueeze起升维的作用,参数表示在哪个地方加一个维度,在第一个维度(中括号)的每个元素加中括号，0表示在张量最外层加一个中括号变成第一维

        # add output to input embedding for attention
        y = (x + input_embedding) * math.sqrt(0.5)

        return {
            'src_tokens': src_tokens,
            'encoder_out': (x, y),
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }


class Decoder(nn.Module):
    def __init__(
            self, dictionary, embed_dim=768, max_positions=1024,
            convolutions=((192, 5),) * 5, dropout=0.1,
    ):
        super(Decoder, self).__init__()
        self.dropout = dropout

        self.embed_norm = nn.LayerNorm(embed_dim)

        convolutions = extend_conv_spec(convolutions)
        in_channels = convolutions[0][0]
        self.fc1 = linear(embed_dim, in_channels, dropout=dropout)
        self.convolutions = nn.ModuleList()
        self.attentions = nn.ModuleList()
        # self.norms = nn.ModuleList()
        # self.residuals = []

        layer_in_channels = [in_channels]
        for i, (out_channels, kernel_size, residual) in enumerate(convolutions):
            self.convolutions.append(
                convtbc(in_channels, out_channels * 2, kernel_size,
                        padding=(kernel_size - 1), dropout=dropout, remove_future=True)
            )
            # self.norms.append(nn.LayerNorm(out_channels))
            # self.residuals.append(residual)
            in_channels = out_channels
            layer_in_channels.append(out_channels)

        self.fcg = linear(embed_dim + out_channels, 1)
        self.fc2 = linear(in_channels, len(dictionary))

    def forward(self, tgt_encodings, encoder_out_dict,
                prev_tokens_with_encodings=None, share_embed_model=None, output_lm_logits=False):
        # tgt_encodings---tgt经过tokenizer后的结果
        # encoder_out_dict---encoder的输出
        # prev_tokens_with_context---target_with_prev_context经过tokenizer后的结果

        src_tokens = encoder_out_dict['src_tokens']
        encoder_out = encoder_out_dict['encoder_out']
        encoder_a, encoder_b = self._split_encoder_out(encoder_out)# x,y

        assert prev_tokens_with_encodings is not None
 
        ## 这里需要考虑一下怎么去获取下一个单词的概览或者说这个概览跟 labels 的关系 这里暂时认为 labels 与
        # outputs = share_embed_model(**prev_tokens_with_context)           
        # embed = outputs.last_hidden_state
        outputs = share_embed_model(input_ids = prev_tokens_with_encodings['input_ids'],
                            attention_mask  = prev_tokens_with_encodings['attention_mask'],
                            labels = prev_tokens_with_encodings['input_ids'],)
        embed = outputs.encoder_last_hidden_state 
 
        lm_logits = None
        if output_lm_logits: 
            lm_logits = share_embed_model.lm_head(embed) # 这里可以运行，但需要看一下这里的对应的函数是什么，和下面的logits有什么区别----两个值size上一致，但下面那个不是概率，不符合要求。
            # lm_logits = outputs.logits  # 这两条貌似都可以？尝试一下，看哪个效果更好
            # self.lm_head()输出层将GPT2Model(config)计算输出的hidden_states张量的最后一个维度由768维(config.n_embd)
            # 投影为词典大小维度(config.vocab_size)的输出层, 此时hidden_states张量的形状将会由(batch_size, 1, n_embed)投影变为
            # lm_logits张量的(batch_size, 1, vocab_size)
            lm_logits = F.log_softmax(lm_logits, dim=-1)    # B, context_tgt, H
        bsz = embed.size(0)

        tgt_ids = find_subsequence_indices(tgt_encodings['input_ids'], prev_tokens_with_encodings['input_ids'])
        mask = tgt_ids.view(-1)   # B x context_tgt
        mask = mask.eq(1)
        embed = embed.view(-1, embed.size(-1))  # B x context_tgt, H

        # take out the target part / exclude context before
        x = embed[mask, :]  # B x tgt, H

        x = x.view(bsz, -1, x.size(1))  # B, tgt, H

        x = self.embed_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        target_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # avg_attn_scores = None
        copy_scores = None
        num_attn_layers = len(self.attentions)
        for conv, attention in zip(self.convolutions, self.attentions):        
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x)# 对应184-191
            x = F.glu(x, dim=2)

            # attention     T x B x C -> B x T x C
            x = x.transpose(0, 1)

        x = x.transpose(0, 1)

        # B x T x [C + E]
        h = torch.cat([x, target_embedding], dim=-1)
        p_gen = torch.sigmoid(self.fcg(h))

        # project back to size of vocabulary
        # B x T x C
        x = self.fc2(x)

        x = F.softmax(x, dim=-1)

        x = x * p_gen

        # x = x.scatter_add(
        #     2, src_tokens.unsqueeze(1).repeat(1, x.size(1), 1),
        #     copy_scores * (1 - p_gen)
        # )

        x = torch.log(x + 1e-32)
        return x, lm_logits
        # return x, avg_attn_scores, lm_logits  

    def _split_encoder_out(self, encoder_out):
        """Split and transpose encoder outputs."""
        # transpose only once to speed up attention layers
        encoder_a, encoder_b = encoder_out
        #print(encoder_a.size(), encoder_b.size())
        encoder_a = encoder_a.transpose(1, 2).contiguous()
        result = (encoder_a, encoder_b)
        return result


def extend_conv_spec(convolutions):# 把卷积核调整为三维
    """
    Extends convolutional spec that is a list of tuples of 2 or 3 parameters
    (kernel size, dim size and optionally how many layers behind to look for residual)
    to default the residual propagation param if it is not specified
    """
    extended = []
    for spec in convolutions:
        if len(spec) == 3:
            extended.append(spec)
        elif len(spec) == 2:
            extended.append(spec + (1,))# 默认由（192,5）变为（192,5,1）
        else:
            raise Exception('invalid number of parameters in convolution spec ' + str(spec) + '. expected 2 or 3')
    return tuple(extended)
def embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, 0.1)
    return m


def linear(in_features, out_features, dropout=0.):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return m


def convtbc(in_channels, out_channels, kernel_size, dropout=0., **kwargs):
    """Weight-normalized Conv1d layer"""
    m = ConvTBC(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return m
