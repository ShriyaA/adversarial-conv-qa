from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from quac_dataset import QuACDataset
from easydict import EasyDict

class QuACQGDataset(QuACDataset):
    def __init__(self, config, split):
        super().__init__(config, split)

    def __getitem__(self, idx):
        item = self.flattened_data[idx]
        context = item['context']
        answer = item['orig_answer']['text']
        question = item['question']
        inputs = self.tokenizer(context, answer, truncation='longest_first', padding='max_length', max_length=self.config.max_len, return_tensors='pt')
        labels = self.tokenizer(question, truncation='longest_first', padding='max_length', max_length=self.config.max_len, return_tensors='pt')
        labels['input_ids'][labels['input_ids']==1] = -1
        return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], 'question': question, 'labels':labels['input_ids'], 'qid': item['turn_id']}

if __name__=='__main__':
    data = QuACQGDataset(EasyDict({'model_name': 'distilbert-base-uncased', 'num_prev_turns': '2', 'max_len': 256}), 'train')
    l = len(data)
    sample = data[25]
