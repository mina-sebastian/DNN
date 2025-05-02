import torch

LAROSEDA = "laroseda"
ARC = "arc"

LAROSEDA_TRAIN = "data/laroseda/laroseda_train.csv"
LAROSEDA_TEST = "data/laroseda/laroseda_test.csv"

ARC_TRAIN = "data/roarc/roarc_train.csv"
ARC_TEST = "data/roarc/roarc_test.csv"


class BaseModel:
    """
    Base class for all models.
    """

    def __init__(self, model_name: str, strategy: str, dataset: str, emb_dim: int,
                 save_interval: int = 200, train_ratio: float = 0.8,
                 BATCH_SIZE: int = 32):
        self.model_name = model_name
        self.strategy = strategy
        self.dataset = dataset
        self.emb_dim = emb_dim
        self.save_interval = save_interval
        self.train_ratio = train_ratio
        self.BATCH_SIZE = BATCH_SIZE

        self.model = None
        self.tokenizer = None

        self.full_dataset = None
        self.test_dataset = None 
        self.train_dataloader = None
        self.val_dataloader = None

        self.best_model = None
        self.history = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def load_datasets(self):
        """
        Load the datasets for training and testing.
        """
        raise NotImplementedError("Subclasses should implement the load_datasets method.")

    def get_embeddings_model(self, text: str):
        """
        Get the embedding for the given text.
        """
        raise NotImplementedError("Subclasses should implement the get_embeddings_model method.")
    
    def fit(self):
        """
        Train the model.
        """
        raise NotImplementedError("Subclasses should implement the train method.")