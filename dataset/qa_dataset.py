import torch
from torch.utils.data import DataLoader
from quac_dataset import QuACDataset
from easydict import EasyDict

class QuACQADataset(QuACDataset):
    def __init__(self, config, split):
        super().__init__(config, split)
        self.chunked_data = self.chunk_data()

    def chunk_data(self):
        chunked_data = []
        for item in self.flattened_data:
            all_tokens = self.tokenizer(item['context'])
            
            answer_start = item['orig_answer']['answer_start'] if not item['context'][item['orig_answer']['answer_start']].isspace() else item['orig_answer']['answer_start'] - 1
            answer_end = item['orig_answer']['answer_end'] if not item['context'][item['orig_answer']['answer_end']].isspace() else item['orig_answer']['answer_end'] - 1

            start_position, end_position = all_tokens.char_to_token(answer_start), all_tokens.char_to_token(answer_end)
            chunked_item, chunk_len = self.chunk_passage(item)
        
            for chunk in chunked_item:
                chunk_dict = {}

                for k in item.keys():
                    chunk_dict[k] = item[k]

                chunk_dict['input_ids'] = chunk['input_ids']
                chunk_dict['attention_mask'] = chunk['attention_mask']

                if chunk['start_offset'] + chunk_len > start_position and chunk['start_offset'] < end_position:
                    if chunk['start_offset'] <= start_position:
                        chunk_dict['start_position'] = start_position - chunk['start_offset']
                    else:
                        chunk_dict['start_position'] = 0
                    if chunk['start_offset'] + chunk_len >= end_position:
                        chunk_dict['end_position'] = end_position - chunk['start_offset']
                    else:
                        chunk_dict['end_position'] = chunk_len - 1
                    chunked_data.append(chunk_dict)

                elif self.split != 'train':
                    chunk_dict['start_position'] = self.config.max_len
                    chunk_dict['end_position'] = self.config.max_len
                    chunked_data.append(chunk_dict)

        return chunked_data

    def chunk_passage(self, item):
        max_query_len = self.config.max_query_len
        question_with_history = self.get_history_turns(item)
        max_seq_length = self.config.max_len
        
        tokenized_question = self.tokenizer(question_with_history, max_length=max_query_len, truncation=True, padding=False)
        query_len = min(len(tokenized_question['input_ids']), max_query_len)
        context_len = max_seq_length - query_len - 1 # -1 for final [SEP] token

        context_tokenized = self.tokenizer(item['context'], max_length=context_len, truncation=True, padding=False, return_overflowing_tokens=True, add_special_tokens=False, stride=self.config.context_stride)
        chunked_items = []
        start_offset = 0

        for i in range(len(context_tokenized['input_ids'])):
            input_ids = tokenized_question['input_ids'] + context_tokenized['input_ids'][i] + [self.tokenizer.convert_tokens_to_ids('[SEP]')]
            attention_mask = tokenized_question['attention_mask'] + context_tokenized['attention_mask'][i] + [1]
            
            if i == len(context_tokenized['input_ids']) - 1:
                input_ids = input_ids + ([self.tokenizer.convert_tokens_to_ids('[PAD]')] * (max_seq_length-len(input_ids)))
                attention_mask = attention_mask + ([0] * (max_seq_length-len(attention_mask)))

            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)

            chunked_items.append({'input_ids':input_ids, 'attention_mask':attention_mask, 'start_offset':start_offset})
            start_offset += context_len
        return chunked_items, context_len

    def __len__(self):
        return len(self.chunked_data)
        
    def __getitem__(self, idx):
        item = self.chunked_data[idx]
        return {'input_ids': item['input_ids'], 'attention_mask': item['attention_mask'], 'start_positions': item['start_position'], 'end_positions': item['end_position'], 'qid': item['turn_id'], 'yesno': item['yesno'], 'followup': item['followup']}

if __name__=='__main__':
    data = QuACQADataset(EasyDict({'model_name': 'distilbert-base-uncased', 'num_prev_turns': '2', 'max_len': 256, 'max_query_len': 50}), 'train')
    l = len(data)
    sample = data[25]
