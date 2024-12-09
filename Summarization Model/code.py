!pip install datasets==2.9.0
!pip install transformers==4.26.1
!pip install pytorch_lightning==1.9.1
!pip install torch==1.13.1+cu116
!pip install scikit-learn==1.0.2
!pip install pandas==1.3.5

from datasets import load_dataset
from transformers import BartForConditionalGeneration, BartTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
import pytorch_lightning as pl
from transformers import AdamW

!nvidia-smi

from google.colab import drive
drive.mount('/content/drive')

!pip install --upgrade datasets

!pip install --upgrade datasets
from datasets import load_dataset

dataset_ = load_dataset('ccdv/cnn_dailymail', '3.0.0', split='validation')

dataset_ = dataset_.select(range(15)) 
print(dataset_)
print(f"Dataset len(dataset): {len(dataset_)}")
print("\nFirst item 'dataset[0]':")
from pprint import pprint
pprint(dataset_[0])
from datasets import load_dataset
from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __init__(self, texts, summaries, tokenizer, source_len, summ_len):
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.summ_len = summ_len

    def __len__(self):
        return len(self.summaries)

    def __getitem__(self, index):
        text = ' '.join(str(self.texts[index]).split())
        summary = ' '.join(str(self.summaries[index]).split())

        source = self.tokenizer.batch_encode_plus(
            [text],
            max_length=self.source_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        target = self.tokenizer.batch_encode_plus(
            [summary],
            max_length=self.summ_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return (
            source['input_ids'].squeeze(),
            source['attention_mask'].squeeze(),
            target['input_ids'].squeeze(),
            target['attention_mask'].squeeze()
        )


import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import Dataset
class BARTDataLoader(pl.LightningDataModule):
    def __init__(self, tokenizer, text_len, summarized_len, file_path, corpus_size, columns_name, train_split_size, batch_size):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_len = text_len
        self.summarized_len = summarized_len
        self.file_path = file_path
        self.corpus_size = corpus_size
        self.columns = columns_name
        self.train_split_size = train_split_size
        self.batch_size = batch_size

    def prepare_data(self):
        data = pd.read_csv(self.file_path, nrows=self.corpus_size, encoding='latin-1')
        data = data[self.columns]
        data.iloc[:, 1] = 'summarize: ' + data.iloc[:, 1]
        self.text = list(data.iloc[:, 0].values)
        self.summary = list(data.iloc[:, 1].values)

    def setup(self, stage=None):
        X_train, X_val, y_train, y_val = train_test_split(self.text, self.summary, train_size=self.train_split_size)
        self.train_dataset = CustomDataset(X_train, y_train, self.tokenizer, self.text_len, self.summarized_len)
        self.val_dataset = CustomDataset(X_val, y_val, self.tokenizer, self.text_len, self.summarized_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

class AbstractiveSummarizationBARTFineTuning(pl.LightningModule):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask, decoder_input_ids, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        input_ids, source_mask, decoder_input_ids, decoder_mask = batch
        outputs = self(input_ids=input_ids, attention_mask=source_mask, decoder_input_ids=decoder_input_ids, labels=decoder_input_ids)
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, source_mask, decoder_input_ids, decoder_mask = batch
        outputs = self(input_ids=input_ids, attention_mask=source_mask, decoder_input_ids=decoder_input_ids, labels=decoder_input_ids)
        loss = outputs.loss
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=5e-5)


from transformers import BartForConditionalGeneration, BartTokenizer 

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import Dataset
import pandas as pd # Import pandas here with alias pd

class BARTDataLoader(pl.LightningDataModule):
    def __init__(self, tokenizer, text_len, summarized_len, file_path, corpus_size, columns_name, train_split_size, batch_size):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_len = text_len
        self.summarized_len = summarized_len
        self.file_path = file_path
        self.corpus_size = corpus_size
        self.columns = columns_name
        self.train_split_size = train_split_size
        self.batch_size = batch_size

    def prepare_data(self):
        data = pd.read_csv(self.file_path, nrows=self.corpus_size, encoding='latin-1')
        data = data[self.columns]
        data.iloc[:, 1] = 'summarize: ' + data.iloc[:, 1]
        self.text = list(data.iloc[:, 0].values)
        self.summary = list(data.iloc[:, 1].values)

    def setup(self, stage=None):
        X_train, X_val, y_train, y_val = train_test_split(self.text, self.summary, train_size=self.train_split_size)
        self.train_dataset = CustomDataset(X_train, y_train, self.tokenizer, self.text_len, self.summarized_len)
        self.val_dataset = CustomDataset(X_val, y_val, self.tokenizer, self.text_len, self.summarized_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

from torch.utils.data import DataLoader
from datasets import load_dataset

dataset = load_dataset('cnn_dailymail', '3.0.0', split='train')
def tokenize_function(examples):
    return tokenizer(examples["article"], truncation=True, padding="max_length", max_length=1024)

dataset = dataset.map(tokenize_function, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "highlights"])

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

def summarize_article(article):
    model_name = 'facebook/bart-large-cnn'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    inputs = tokenizer.encode(article, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, num_beams=4, max_length=150, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

article = """Ever noticed how plane seats appear to be getting smaller and smaller? With increasing numbers of people taking to the skies, some experts are questioning if having such packed out planes is putting passengers at risk. They say that the shrinking space on aeroplanes is not only uncomfortable - it's putting our health and safety in danger. More than squabbling over the arm rest, shrinking space on planes putting our health and safety in danger? This week, a U.S consumer advisory group set up by the Department of Transportation said at a public hearing that while the government is happy to set standards for animals flying on planes, it doesn't stipulate a minimum amount of space for humans. 'In a world where animals have more rights to space and food than humans,' said Charlie Leocha, consumer representative on the committee. 'It is time that the DOT and FAA take a stand for humane treatment of passengers.' But could crowding on planes lead to more serious issues than fighting for space in the overhead lockers, crashing elbows and seat back kicking? Tests conducted by the FAA use planes with a 31 inch pitch, a standard which on some airlines has decreased . Many economy seats on United Airlines have 30 inches of room, while some airlines offer as little as 28 inches . Cynthia Corbertt, a human factors researcher with the Federal Aviation Administration, that it conducts tests on how quickly passengers can leave a plane. But these tests are conducted using planes with 31 inches between each row of seats, a standard which on some airlines has decreased, reported the Detroit News. The distance between two seats from one point on a seat to the same point on the seat behind it is known as the pitch. While most airlines stick to a pitch of 31 inches or above, some fall below this. While United Airlines has 30 inches of space, Gulf Air economy seats have between 29 and 32 inches, Air Asia offers 29 inches and Spirit Airlines offers just 28 inches. British Airways has a seat pitch of 31 inches, while easyJet has 29 inches, Thomson's short haul seat pitch is 28 inches, and Virgin Atlantic's is 30-31."""
summary = summarize_article(article)
print("Summary:")
print(summary)