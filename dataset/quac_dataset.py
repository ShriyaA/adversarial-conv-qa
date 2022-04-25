from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from easydict import EasyDict

class QuACDataset(Dataset):
    def __init__(self, config, split):
        
        self.config = config
        self.data = load_dataset('quac', split=split)
        self.flattened_data, self.turn_id_idx_map = self.flatten_data()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=True)

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
                row_dict['answers'] = {'answer_starts': [int(x) for x in row['answers']['answer_starts'][i]], 'texts': row['answers']['texts'][i]}
                row_dict['answer_ends'] = [x + len(row_dict['answers']['texts'][j]) for j,x in enumerate(row_dict['answers']['answer_starts'])]
                flattened_data.append(row_dict)
                turn_id_idx_map[row_dict['turn_id']] = idx
                idx += 1
        return flattened_data, turn_id_idx_map
        

    def __len__(self):
        return len(self.flattened_data)

    def __getitem__(self, idx):
        raise NotImplementedError()