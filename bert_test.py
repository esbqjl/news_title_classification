import torch    
from transformers import BertTokenizer, BertModel

'''
classifier or sequence label task,
single sentence input bert persentation
'''

texts=[
    '你好呀',
    '我不是很好，good123'
]
# load the bert tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('./model/bert-base-chinese')

# to obtain input_ids and att_mask
# text of how input_ids store their ids, none pad part of att_mask is 1, otherwise 0
batch_input_ids,batch_att_mask = [], []
for text in texts:
    encoded_dict = bert_tokenizer.encode_plus( #tokenzier get id format tensor, and pad
        text, max_length = 10, padding = 'max_length', return_tensors='pt',truncation=True
    )
    batch_input_ids.append(encoded_dict['input_ids'])
    batch_att_mask.append(encoded_dict['attention_mask'])    
batch_input_ids = torch.cat(batch_input_ids)
batch_att_mask = torch.cat(batch_att_mask)

bert_model = BertModel.from_pretrained('./model/bert-base-chinese')
with torch.no_grad():
    last_hidden_state, pooled_output = bert_model(input_ids = batch_input_ids, attention_mask = batch_att_mask, return_dict=False)
    
    print('last_hidden_state',last_hidden_state,last_hidden_state.shape)
    print('\n')
    print('pooled_output',pooled_output,pooled_output.shape)