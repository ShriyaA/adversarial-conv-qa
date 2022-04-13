import os
import torch

from json import load
from agents.base_agent import BaseAgent
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from torch import optim, nn
from torch.utils.data import DataLoader
from dataset.quac_dataset import QuACTrainDataset
from utils.misc import print_cuda_statistics

class FinetuneQAModelAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)
        # define models
        self.model_name = config.model_name
        self.checkpoint_path = config.checkpoint_path
        self.model = self.load_checkpoint(self.checkpoint_path)
        
        # define data_loader
        self.dataset = QuACTrainDataset(config)
        self.data_loader = DataLoader(self.dataset, batch_size=config.batch_size, shuffle=config.shuffle_data)

        # define loss
        self.loss = nn.NLLLoss()

        # define optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

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
            self.loss = self.loss.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA*****\n")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_path)
        # Summary Writer
        self.summary_writer = None


    def load_checkpoint(self, path):
        if len(path) > 0 and os.path.exists(path):
            model = AutoModelForQuestionAnswering.from_pretrained(path)
        else:
            model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        return model

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
        for batch_idx, batch in enumerate(self.data_loader):
            batch['input_ids'].to(self.device)
            batch['attention_mask'].to(self.device)
            batch['start_positions'].to(self.device)
            batch['end_positions'].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(batch['inputs'], attention_mask=batch['attention_mask'], start_positions=batch['start_positions'], end_positions=batch['end_positions'])

            loss = outputs.loss
            loss.backward()
            self.optimizer.step()

            if batch_idx % self.config.log_interval == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.current_epoch, batch_idx * len(batch), len(self.data_loader.dataset),
                           100. * batch_idx / len(self.data_loader), loss.item()))
            self.current_iteration += 1

    def validate(self):
        pass

    def finalize(self):
        pass