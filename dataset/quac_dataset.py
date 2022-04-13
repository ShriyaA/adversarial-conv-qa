from distutils.command.config import config
from struct import pack_into
from tracemalloc import start
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from easydict import EasyDict

class QuACTrainDataset(Dataset):
    def __init__(self, config):
        
        self.config = config
        self.data = load_dataset('quac', split='train')
        self.flattened_data, self.turn_id_idx_map = self.flatten_data()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=True)
        self.num_prev_turns = int(self.config.num_prev_turns)

    def flatten_data(self):
        flattened_data = []
        turn_id_idx_map = {}
        single_keys = ['dialogue_id', 'wikipedia_page_title', 'background', 'section_title', 'context']
        multi_keys = ['turn_ids', 'questions', 'followups', 'yesnos']
        idx = 0
        for row in self.data:
            for i in range(len(row['questions'])):
                row_dict = {}
                for k in single_keys:
                    row_dict[k] = row[k]
                for k in multi_keys:
                    row_dict[k[:-1]] = row[k][i]
                row_dict['orig_answer'] = {'answer_start': int(row['orig_answers']['answer_starts'][i]), 'answer_end': int(row['orig_answers']['answer_starts'][i]) + len(row['orig_answers']['texts'][i]), 'text':row['orig_answers']['texts'][i]}
                flattened_data.append(row_dict)
                turn_id_idx_map[row_dict['turn_id']] = idx
                idx += 1
        return flattened_data, turn_id_idx_map
        

    def __len__(self):
        return len(self.flattened_data)

    def __getitem__(self, idx):
        item = self.flattened_data[idx]
        turn_no = int(item['turn_id'].split('#')[-1])
        turn_history = ""
        if turn_no > 0:
            for i in range(max(turn_no-self.num_prev_turns, 0), turn_no):
                turn_history += " " + self.flattened_data[self.turn_id_idx_map[item['turn_id'].split('#')[0] + '#' + str(i)]]['question']
                turn_history += " " + self.flattened_data[self.turn_id_idx_map[item['turn_id'].split('#')[0] + '#' + str(i)]]['orig_answer']['text']
        inputs = self.tokenizer(item['context'], turn_history.strip() + " " + item['question'], truncation='only_first', padding='max_length', max_length=self.config.max_len, return_tensors='pt')
        start_token = inputs.char_to_token(item['orig_answer']['answer_start'])
        if start_token is None:
            start_token = self.config.max_len
        end_token = inputs.char_to_token(item['orig_answer']['answer_end'])
        if end_token is None:
            end_token = self.config.max_len
        return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], 'start_positions': start_token, 'end_positions': end_token}

if __name__=='__main__':
    data = QuACTrainDataset(EasyDict({'model_name': 'distilbert-base-uncased', 'num_prev_turns': '2', 'max_len': 256}))
    print(len(data))
    from torch.utils.data import DataLoader
    dataloader = DataLoader(data, batch_size=3, shuffle=True)
    batch = next(iter(dataloader))
    example = data[5]