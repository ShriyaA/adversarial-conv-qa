import os
import torch
import math
import wandb

from json import load
from agents.base_agent import BaseAgent
from transformers import optimization, AutoTokenizer, AutoModelForQuestionAnswering
from torch.utils.data import DataLoader
from dataset.qa_dataset import QuACQADataset
from utils.misc import print_cuda_statistics
from utils.scorer import external_call

class FinetuneQAModelAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)
        # define models
        self.model_name = config.model_name
        self.checkpoint_path = config.checkpoint_path
        self.model = self.load_checkpoint(self.checkpoint_path)
        wandb.watch(self.model)

        # define tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # define data_loader
        self.train_dataset = QuACQADataset(config, 'train')
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=config.shuffle_data)
        self.valid_dataset = QuACQADataset(config, 'validation')
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=config.valid_batch_size, shuffle=False)

        # initialize counter
        self.current_epoch = 1
        self.current_iteration = 0
        self.best_metric = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            torch.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            self.model = self.model.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA*****\n")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        # define optimizer

        no_decay = []
        for name, _ in self.model.named_parameters():
            if 'bias' in name or 'layer_norm' in name or 'LayerNorm' in name:
                no_decay.append(name)

        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in no_decay],
                    "weight_decay": config.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n in no_decay],
                    "weight_decay": 0.0,
                },
            ]
        
        self.optimizer = optimization.AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
        
        # define scheduler
        num_training_steps = len(self.train_dataloader) * self.config.max_epoch
        self.scheduler = optimization.get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=int(0.1*num_training_steps), num_training_steps=num_training_steps)

        # Summary Writer
        self.summary_writer = None


    def load_checkpoint(self, path):
        if len(path) > 0 and os.path.exists(path):
            model = AutoModelForQuestionAnswering.from_pretrained(path)
        else:
            model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        return model

    def save_checkpoint(self, is_best=False):
        checkpoint_dir = self.config.checkpoint_dir
        if not is_best:
            model_dir = os.path.join(checkpoint_dir, 'step_{}'.format(self.current_iteration))
        else:
            model_dir = os.path.join(checkpoint_dir, 'best_model')
        os.makedirs(model_dir, exist_ok=True)
        self.model.save_pretrained(model_dir)

    def run(self):
        self.train()

    def train(self):
        for epoch in range(1, self.config.max_epoch + 1):
            self.train_one_epoch()
            if self.config.validate_during_training:
                self.validate()
            self.current_epoch += 1

    def train_one_epoch(self):
        
        self.model.train()
        for batch_idx, batch in enumerate(self.train_dataloader):
            input_ids = batch['input_ids'].to(self.device).squeeze()
            attention_mask = batch['attention_mask'].to(self.device).squeeze()
            start_positions = batch['start_positions'].to(self.device)
            end_positions = batch['end_positions'].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)

            loss = outputs.loss
            loss.backward()
            assert math.isnan(loss) == False, "Current step {}".format(self.current_iteration)
            self.optimizer.step()
            self.scheduler.step()

            if batch_idx % self.config.log_interval == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.current_epoch, batch_idx, len(self.train_dataloader),
                           100. * batch_idx / len(self.train_dataloader), loss.item()))
                wandb.log({'train_loss': loss.item(), 'epoch': self.current_epoch, 'step': self.current_iteration}, step=self.current_iteration)
                        
            if self.config.validate_during_training and batch_idx % self.config.validate_every == 0:
                self.validate()

            if batch_idx % self.config.save_every == 0:
                self.logger.info('Saving model at step {} with Loss {}'.format(self.current_iteration, loss.item()))
                self.save_checkpoint()

            self.current_iteration += 1

    def validate(self):
        
        self.model.eval()
        predictions = []
        for batch_idx, batch in enumerate(self.valid_dataloader):

            input_ids = batch['input_ids'].to(self.device).squeeze()
            attention_mask = batch['attention_mask'].to(self.device).squeeze()
            start_positions = batch['start_positions'].to(self.device)
            end_positions = batch['end_positions'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            
            loss = outputs.loss.item()
            answer_start_index = outputs.start_logits.argmax(dim=1).cpu().tolist()
            answer_end_index = outputs.end_logits.argmax(dim=1).cpu().tolist()

            if batch_idx % self.config.validation_log_interval == 0:
                self.logger.info('Validation Batch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx, batch_idx, len(self.valid_dataloader),
                           100. * batch_idx / len(self.valid_dataloader), loss))
            
            for i, start_idx in enumerate(answer_start_index):
                end_idx = answer_end_index[i]
                best_span_str = self.tokenizer.decode(input_ids[i,start_idx:end_idx].cpu())
                qid = batch['qid'][i]
                yesno = batch['yesno'][i]
                followup = batch['followup'][i]
                predictions.append({'best_span_str':[best_span_str], 'qid':[qid], 'yesno':[yesno.item()], 'followup':[followup.item()]})

        metrics = external_call(predictions)
        self.logger.info('Validation F1 Score: {}'.format(metrics['f1']))
        wandb.log({'valid_loss':loss, 'epoch':self.current_epoch, 'step':self.current_iteration}, step=self.current_iteration)
        wandb.log(metrics, step=self.current_iteration)

        if metrics['f1'] > self.best_metric:
            self.logger.info('Saving best model at step {} with F1 Score {}'.format(self.current_iteration, metrics['f1']))
            wandb.log({'best_f1': metrics['f1']}, step=self.current_iteration)
            self.save_checkpoint(is_best=True)
            

    def finalize(self):
        pass