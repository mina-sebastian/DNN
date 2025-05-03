from typing import List
import torch
from torch.utils.data import DataLoader
from utils.base_model import BaseModel
from utils.datasets_class import MultipleChoiceCombinedDataset, MultipleChoicePointwiseCached, MultipleChoiceSeparatedDataset, TitleContentDataset
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim

from collections import Counter
from transformers import AutoModel, AutoTokenizer

from utils.models_dnn import ArcMLP, DualInputMLP, MultiOptionMLP, OneOptionMLP, PairwiseQuadricMLP, PairwiseQuadricWithQueryMLP, QuadricInputMLP, SharedInputMLP
from utils.pairwise_loss import PairwiseRankingLoss
from utils.train_util import test_model, train_and_evaluate
from utils.get_embeddings import get_embedding
from utils.base_model import LAROSEDA, ARC, LAROSEDA_TRAIN, LAROSEDA_TEST, ARC_TRAIN, ARC_TEST

device = "cuda" if torch.cuda.is_available() else "cpu"
# model_id = "faur-ai/LLMic"

# # HYPERPARAMS
# BATCH_SIZE = 32

# tokenizer = None
# model_llmic = None


# def get_embedding_llmic(text):
#     inputs = tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         padding=True,
#         max_length=512
#     ).to(device)

#     global tokenizer, model_llmic
#     if tokenizer is None:
#         tokenizer = AutoTokenizer.from_pretrained(model_id)
#     if model_llmic is None:
#         model_llmic = AutoModel.from_pretrained(model_id).to(device)

#     return get_embedding(
#         model=model_llmic,
#         tokenizer=tokenizer,
#         text=text,
#         strategy="mean"
#     )
    


    

# full_dataset = TitleContentDataset(
#     csv_file="data/laroseda/laroseda_train.csv",
#     get_embedding=get_embedding,  
#     type="train",
#     name="mGPT-1-3B-romanian",                 
#     emb_dim=2048,                
#     save_interval=200            
# )

# test_dataset = TitleContentDataset(
#     csv_file="data/laroseda/laroseda_test.csv",
#     get_embedding=get_embedding,
#     type="test",
#     name="mGPT-1-3B-romanian",
#     emb_dim=2048,
#     save_interval=200
# )

# train_ratio = 0.8
# train_size = int(train_ratio * len(full_dataset))
# val_size = len(full_dataset) - train_size

# generator = torch.Generator().manual_seed(42)

# train_dataset, val_dataset = random_split(
#     full_dataset,
#     lengths=[train_size, val_size],
#     generator=generator
# )

# print(f"Full dataset size: {Counter(full_dataset.labels)}")
# print(f"Train dataset size: {Counter(train_dataset.dataset.labels[train_dataset.indices])}")
# print(f"Val dataset size: {Counter(val_dataset.dataset.labels[val_dataset.indices])}")
# print(f"Test dataset size: {Counter(test_dataset.labels)}")

# train_dataloader = DataLoader(
#         train_dataset, 
#         batch_size=BATCH_SIZE, 
#         shuffle=True)

# val_dataloader = DataLoader(
#         val_dataset, 
#         batch_size=BATCH_SIZE, 
#         shuffle=False)


# llmic_mlp = DualInputMLP(input_dim=2048, hidden_dim=512).to(device)

# history, best_model = train_and_evaluate(
#     model=llmic_mlp,
#     train_loader=train_dataloader,
#     val_loader=val_dataloader,
#     criterion=nn.BCEWithLogitsLoss(),
#     optimizer=optim.Adam(llmic_mlp.parameters(), lr=1e-3),
#     num_epochs=10,
#     device=device,
#     name="llmic",
#     save=True,
#     do_all_metrics=True,
# )
# test_model(llmic_mlp, 'llmic', test_dataset, device=device)


class LLMicModel(BaseModel):
    """
    LLMic model class
    """
    def __init__(self, strategy: str, dataset: str,
                 save_interval: int = 200, train_ratio: float = 0.8,
                 BATCH_SIZE: int = 32):
        super().__init__("LLMic", strategy, dataset, 2560,
                         save_interval, train_ratio, BATCH_SIZE)

    def get_embeddings_model(self, text: str) -> List[float]:
        """
        Get the embedding for the given text using LLMic model.
        """
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("faur-ai/LLMic")
        if self.model is None:
            self.model = AutoModel.from_pretrained("faur-ai/LLMic").to(self.device)
        
        return get_embedding(
            model=self.model,
            tokenizer=self.tokenizer,
            text=text,
            strategy=self.strategy
        )

    def load_datasets(self):
        """
        Load the datasets for training and testing.
        """
        generator = torch.Generator().manual_seed(42)
        if self.dataset == LAROSEDA:
            self.full_dataset = TitleContentDataset(
                csv_file=LAROSEDA_TRAIN,
                get_embedding=self.get_embeddings_model,
                type="train",
                name=self.model_name,
                emb_dim=self.emb_dim,
                save_interval=self.save_interval
            )

            self.test_dataset = TitleContentDataset(
                csv_file=LAROSEDA_TEST,
                get_embedding=self.get_embeddings_model,
                type="test",
                name=self.model_name,
                emb_dim=self.emb_dim,
                save_interval=self.save_interval
            )

            train_size = int(self.train_ratio * len(self.full_dataset))
            val_size = len(self.full_dataset) - train_size

            self.train_dataset, self.val_dataset = random_split(
                self.full_dataset,
                lengths=[train_size, val_size],
                generator=generator
            )

            print(f"Full dataset size: {Counter(self.full_dataset.labels)}")
            print(f"Train dataset size: {Counter(self.train_dataset.dataset.labels[self.train_dataset.indices])}")
            print(f"Val dataset size: {Counter(self.val_dataset.dataset.labels[self.val_dataset.indices])}")
            print(f"Test dataset size: {Counter(self.test_dataset.labels)}")

            self.train_dataloader = DataLoader(
                    self.train_dataset, 
                    batch_size=self.BATCH_SIZE, 
                    shuffle=True)

            self.val_dataloader = DataLoader(
                    self.val_dataset, 
                    batch_size=self.BATCH_SIZE, 
                    shuffle=False)
            
        elif self.dataset == ARC:
            # self.full_dataset = MultipleChoiceSeparatedDataset(
            #     csv_file=ARC_TRAIN,
            #     get_embedding=self.get_embeddings_model,
            #     emb_dim=self.emb_dim,
            #     name=f'roarc_train_{self.model_name.lower()}',
            # )

            # self.test_dataset = MultipleChoiceSeparatedDataset(
            #     csv_file=ARC_TEST,
            #     get_embedding=self.get_embeddings_model,
            #     emb_dim=self.emb_dim,
            #     name=f'roarc_test_{self.model_name.lower()}',
            # )

            # self.full_dataset = MultipleChoicePointwiseCached(
            #     csv_file=ARC_TRAIN,
            #     get_embedding=self.get_embeddings_model,
            #     emb_dim=self.emb_dim,
            #     name=f'roarc_train_llmic',
            # )

            self.full_dataset = MultipleChoiceCombinedDataset(
                csv_file=ARC_TRAIN,
                get_embedding=self.get_embeddings_model,
                emb_dim=self.emb_dim,
                name=f'roarc_train_{self.model_name.lower()}_{self.strategy}',
                save_interval=1
            )

            print(f"Loaded {len(self.full_dataset)} samples.")

            train_ratio = 0.8
            train_size = int(train_ratio * len(self.full_dataset))
            val_size = len(self.full_dataset) - train_size

            self.train_dataset, self.val_dataset = random_split(
                self.full_dataset,
                lengths=[train_size, val_size],
                generator=generator
            )

            print(f"Full dataset size: {Counter(self.full_dataset.labels)}")
            print(f"Train dataset size: {Counter(self.train_dataset.dataset.labels[self.train_dataset.indices])}")
            print(f"Val dataset size: {Counter(self.val_dataset.dataset.labels[self.val_dataset.indices])}")

            print(f"Train samples: {len(self.train_dataset)}")
            print(f"Val samples: {len(self.val_dataset)}")

            self.train_dataloader = DataLoader(
                    self.train_dataset, 
                    batch_size=self.BATCH_SIZE, 
                    shuffle=True)

            self.val_dataloader = DataLoader(
                    self.val_dataset, 
                    batch_size=self.BATCH_SIZE, 
                    shuffle=False)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        
    def fit(self):
        """
        Train the model.
        """
        if self.train_dataloader is None or self.val_dataloader is None:
            self.load_datasets()

        if self.dataset == LAROSEDA:
            self.mlp_model = DualInputMLP(input_dim=self.emb_dim, hidden_dim=512).to(self.device)

            self.history, self.best_model = train_and_evaluate(
                model=self.mlp_model,
                train_loader=self.train_dataloader,
                val_loader=self.val_dataloader,
                criterion=nn.BCEWithLogitsLoss(),
                optimizer=optim.Adam(self.mlp_model.parameters(), lr=1e-3),
                num_epochs=10,
                device=self.device,
                name=f'{self.model_name}_{self.strategy}_laroseda',
                save=True,
                do_all_metrics=True,
            )

            test_model(self.mlp_model, f'{self.model_name}_laroseda', self.test_dataset, device=self.device)
        elif self.dataset == ARC:
            # self.mlp_model = MultiOptionMLP(
            #         input_dim=self.emb_dim,
            #         hidden_dim=1024,
            #     ).to(self.device)

            # self.history, self.best_model = train_and_evaluate(
            #     model=self.mlp_model,
            #     train_loader=self.train_dataloader,
            #     val_loader=self.val_dataloader,
            #     criterion=nn.CrossEntropyLoss(),
            #     optimizer=optim.SGD(
            #         self.mlp_model.parameters(),
            #         lr=0.0001,
            #         momentum=0.9,
            #     ),
            #     num_epochs=40,
            #     device=self.device,
            #     name=f'{self.model_name}_{self.strategy}_arc',
            #     save=True,
            #     do_all_metrics=True,
            #     is_binary=True,
            # )

            self.mlp_model = ArcMLP(
                input_dim=self.emb_dim,
                hidden_dim=self.emb_dim * 2,
            ).to(self.device)

            self.history, self.best_model = train_and_evaluate(
                model=self.mlp_model,
                train_loader=self.train_dataloader,
                val_loader=self.val_dataloader,
                criterion=PairwiseRankingLoss(),
                # optimizer = optim.AdamW(self.mlp_model.parameters(), lr=1e-4, weight_decay=1e-4),
                # optimizer=optim.Adam(self.mlp_model.parameters(), lr=1e-4),
                # optimizer=optim.SGD(self.mlp_model.parameters(), lr=1e-5, momentum=0.4),
                optimizer=optim.SGD(
                    self.mlp_model.parameters(),
                    lr=0.0001,
                    momentum=0.9,
                ),
                # use_scheduler=True,
                num_epochs=25,
                device=self.device,
                name=f'{self.model_name}_{self.strategy}_arc_new_new',
                save=True,
                is_binary=False,
                do_all_metrics=True,
                # one_input=True
            )

            # test_model(self.mlp_model, self.model_name, self.test_dataset, device=self.device)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

llmic_model = LLMicModel(strategy="mean", dataset=ARC)
llmic_model.fit()
