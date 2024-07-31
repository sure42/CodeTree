XX_encodings ---- tokenizer后的结果
XX_tokens ---- tokenizer后结果中的input_ids部分



### EncodeNode
1. src_encodings src经过tokenizer后的结果
2. src_with_prev_encodings=None prev+src经过tokenizer后的结果
3. share_embed_model 特征提取模型
4. src_ids->musk 分析哪一部分是src，哪一部分对应prev。作用是对src_with_prev_encodings经过特征提取后的结果做处理，提取出src部分。
5. src_tokens src_encodings['input_ids'],src对应的分词结果