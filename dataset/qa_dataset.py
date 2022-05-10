import torch
import json
from torch.utils.data import DataLoader
from quac_dataset import QuACDataset
from easydict import EasyDict
from transformers import AutoTokenizer

class QuACQADataset(QuACDataset):
    def __init__(self, config, split):

        if not config.generated_questions:
            super().__init__(config, split)
        else:
            self.config = config
            self.split = 'train'
            self.input_file = config.input_file
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=True)
            self.num_prev_turns = int(self.config.num_prev_turns)
            self.turn_id_idx_map = {}

            with open(self.input_file) as f:
                self.flattened_data = json.load(f)
            
            for i,item in enumerate(self.flattened_data):
                self.turn_id_idx_map[item['turn_id']] = i

        self.chunked_data = self.chunk_data()

    def chunk_data(self):
        chunked_data = []
        for item in self.flattened_data:
            
            item['orig_answer']['answer_start'] = item['orig_answer']['answer_start'] if not item['context'][item['orig_answer']['answer_start']].isspace() else item['orig_answer']['answer_start'] - 1
            item['orig_answer']['answer_end'] = item['orig_answer']['answer_end'] if not item['context'][item['orig_answer']['answer_end']].isspace() else item['orig_answer']['answer_end'] - 1

            
            chunked_item, _ = self.chunk_passage(item)
        
            for chunk in chunked_item:
                chunk_dict = {}

                for k in item.keys():
                    chunk_dict[k] = item[k]

                chunk_dict['input_ids'] = chunk['input_ids']
                chunk_dict['attention_mask'] = chunk['attention_mask']
                chunk_dict['token_type_ids'] = chunk['token_type_ids']
                chunk_dict['start_position'] = chunk['start_position']
                chunk_dict['end_position'] = chunk['end_position']

                if chunk_dict['start_position'] is not None and chunk_dict['end_position'] is not None:
                    chunked_data.append(chunk_dict)

                elif self.split != 'train':
                    chunk_dict['start_position'] = 0
                    chunk_dict['end_position'] = 0
                    chunked_data.append(chunk_dict)

        return chunked_data

    def chunk_passage(self, item):
        max_query_len = self.config.max_query_len
        question_with_history = self.get_history_turns(item)
        max_seq_length = self.config.max_len
        
        tokenized_question = self.tokenizer(question_with_history, truncation=False, padding=False)
        
        if len(tokenized_question['input_ids']) > max_query_len:
            tokenized_question['input_ids'] = [tokenized_question['input_ids'][0]] + tokenized_question['input_ids'][len(tokenized_question['input_ids']) - max_query_len + 1:]
            tokenized_question['attention_mask'] = [tokenized_question['attention_mask'][0]] + tokenized_question['attention_mask'][len(tokenized_question['attention_mask']) - max_query_len + 1:]
        
        query_len = len(tokenized_question['input_ids'])
        context_len = max_seq_length - query_len - 1 # -1 for final [SEP] token

        context_tokenized = self.tokenizer(item['context'], max_length=context_len, truncation=True, padding=False, return_overflowing_tokens=True, add_special_tokens=False, return_offsets_mapping=True, stride=self.config.context_stride)
        chunked_items = []

        for i in range(len(context_tokenized['input_ids'])):

            offsets_mapping = context_tokenized['offset_mapping'][i]
            start_token, end_token = (None, None)

            for j, offset in enumerate(offsets_mapping):
                if offset[0] <= item['orig_answer']['answer_start'] and offset[1] >= item['orig_answer']['answer_start']:
                    start_token = j
                if offset[0] <= item['orig_answer']['answer_end'] and offset[1] >= item['orig_answer']['answer_end']:
                    end_token = j

            if start_token is not None and end_token is not None:
                start_token += len(tokenized_question['input_ids'])
                end_token += len(tokenized_question['input_ids'])

            input_ids = tokenized_question['input_ids'] + context_tokenized['input_ids'][i] + [self.tokenizer.convert_tokens_to_ids('[SEP]')]
            attention_mask = tokenized_question['attention_mask'] + context_tokenized['attention_mask'][i] + [1]
            token_type_ids = [0] * len(tokenized_question['input_ids']) + [1] * (len(context_tokenized['input_ids'][i]) + 1)
            
            if i == len(context_tokenized['input_ids']) - 1:
                input_ids = input_ids + ([self.tokenizer.convert_tokens_to_ids('[PAD]')] * (max_seq_length-len(input_ids)))
                attention_mask = attention_mask + ([0] * (max_seq_length-len(attention_mask)))
                token_type_ids = token_type_ids + ([1] * (max_seq_length-len(token_type_ids)))

            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)
            token_type_ids = torch.tensor(token_type_ids)

            chunked_items.append({'input_ids':input_ids, 'attention_mask':attention_mask, 'token_type_ids':token_type_ids, 'start_position':start_token, 'end_position': end_token})
        return chunked_items, context_len

    def __len__(self):
        return len(self.chunked_data)
        
    def __getitem__(self, idx):
        item = self.chunked_data[idx]
        return {'input_ids': item['input_ids'], 'attention_mask': item['attention_mask'], 'token_type_ids': item['token_type_ids'], 'start_positions': item['start_position'], 'end_positions': item['end_position'], 'qid': item['turn_id'], 'yesno': item['yesno'], 'followup': item['followup']}

if __name__=='__main__':
    data = QuACQADataset(EasyDict({'model_name': 'distilbert-base-uncased', 'num_prev_turns': 3, 'max_len': 256, 'max_query_len': 50, 'context_stride': 128, 'generated_questions':True, 'input_file':'./data/generated_questions_filtered.json'}), 'validation')
    l = len(data)
    sample = data[25]
