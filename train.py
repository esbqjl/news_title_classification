import numpy as np
import torch
from sklearn.metrics import accuracy_score

from classification.bert_fc.bert_fc_predictor import BertFCPredictor
from classification.bert_fc.bert_fc_trainer import BertFCTrainer

# setup random seed
seed =0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


def read_data(data_path):
    '''
    read original data, return titles, labels
    
    '''
    titles,labels = [],[]
    with open(data_path,'r',encoding='utf-8') as f:
        print('current_file',data_path)
        
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            _,_,label,title,_ = line.split('_!_')
            
            titles.append(list(title)),labels.append([label])
            
        print(data_path,'finish')
        
    return titles, labels

# read train dev test 
train_path, dev_path, test_path = \
    'data/toutiao_cat_data.train.txt','data/toutiao_cat_data.dev.txt','data/toutiao_cat_data.test.txt'
(train_texts, train_labels), (dev_texts, dev_labels), (test_texts, test_labels) = \
    read_data(train_path),read_data(dev_path),read_data(test_path)
    
# trainer= BertFCTrainer(
#     pretrainerd_model_dir='./model/bert-base-chinese',model_dir='./temp/bertfc',learning_rate=5e-5
# )

# trainer.train(
#     train_texts,train_labels,validate_texts = dev_texts, validate_labels = dev_labels, batch_size=64,epoch=4
# )

predictor= BertFCPredictor(
    pretrained_model_dir='./model/bert-base-chinese',model_dir='./temp/bertfc'
)

predict_labels = predictor.predict(test_texts, batch_size=64)

# validation
test_acc = accuracy_score(test_labels, predict_labels)
print('test acc', test_acc)

