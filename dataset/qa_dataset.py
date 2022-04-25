from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from quac_dataset import QuACDataset
from easydict import EasyDict

class QuACQADataset(QuACDataset):
    def __init__(self, config, split):
        super().__init__(config, split)
        self.num_prev_turns = int(self.config.num_prev_turns)

    def __getitem__(self, idx):
        item = self.flattened_data[idx]

        turn_no = int(item['turn_id'].split('#')[-1])
        turn_history = ""
        if turn_no > 0:
            for i in range(max(turn_no-self.num_prev_turns, 0), turn_no):
                try:
                    history_turn_id = item['turn_id'].split('#')[0] + '#' + str(i)
                    history_idx = self.turn_id_idx_map[history_turn_id]
                    turn_history += " " + self.flattened_data[history_idx]['question']
                    turn_history += " " + self.flattened_data[history_idx]['orig_answer']['text']
                except KeyError:
                    pass
        question_with_history = (turn_history.strip() + " " + item['question']).strip()

        inputs = self.tokenizer(question_with_history, item['context'], truncation='longest_first', padding='max_length', max_length=self.config.max_len, return_tensors='pt')
        
        start_token = inputs.char_to_token(item['orig_answer']['answer_start'], sequence_index=1)
        if start_token is None:
            start_token = self.config.max_len - 1
        end_token = inputs.char_to_token(item['orig_answer']['answer_end'], sequence_index=1)
        if end_token is None:
            end_token = self.config.max_len - 1

        return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], 'start_positions': start_token, 'end_positions': end_token, 'qid': item['turn_id'], 'yesno': item['yesno'], 'followup': item['followup']}

if __name__=='__main__':
    data = QuACQADataset(EasyDict({'model_name': 'distilbert-base-uncased', 'num_prev_turns': '2', 'max_len': 256}), 'train')
    l = len(data)
    sample = data[25]