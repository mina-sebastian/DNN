from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from utils.base_model import ARC_TRAIN, LAROSEDA_TRAIN
from utils.datasets_class import MultipleChoiceCombinedDataset, MultipleChoicePointwiseCached, MultipleChoiceSeparatedDataset, TitleContentDataset

##################### ROARC EMBEDDINGS #####################
# dataset_to_visualise = MultipleChoicePointwiseCached(
#                 csv_file=ARC_TRAIN,
#                 get_embedding=None,
#                 emb_dim=2560,
#                 name=f'roarc_train_llmic',
#             )

# print(f"Dataset size: {len(dataset_to_visualise)}")

# #tsne visualisation
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import numpy as np


# tsne = TSNE(n_components=2, random_state=42, n_iter=1000, perplexity=30, verbose=1)
# embeddings = []
# labels = []
# for i in range(len(dataset_to_visualise)):
#     data = dataset_to_visualise[i]
#     embeddings.append(data[0])
#     labels.append(data[1])
# embeddings = np.array(embeddings)
# labels = np.array(labels)
# embeddings = tsne.fit_transform(embeddings)

# plt.figure(figsize=(10, 10))
# plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='viridis', alpha=0.5)
# plt.colorbar()
# plt.title('t-SNE visualization of embeddings')
# plt.xlabel('t-SNE component 1')
# plt.ylabel('t-SNE component 2')
# plt.savefig('tsne_visualisation.png')




# dataset_laroseda = TitleContentDataset(
#     csv_file=LAROSEDA_TRAIN,
#     get_embedding=None,
#     type="train",
#     name="llmic",
#     emb_dim=2560,
# )

# print(f"Dataset size: {len(dataset_laroseda)}")
# print(f'Sample: {dataset_laroseda[0]}')

# #tsne visualisation
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import numpy as np


# tsne = TSNE(n_components=2, random_state=42, n_iter=1000, perplexity=30, verbose=1)
# embeddings = []
# labels = []
# for i in range(len(dataset_laroseda)):
#     data = dataset_laroseda[i]
#     embeddings.append(data[0])
#     labels.append(data[2])
# embeddings = np.array(embeddings)
# labels = np.array(labels)
# embeddings = tsne.fit_transform(embeddings)

# plt.figure(figsize=(10, 10))
# plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='viridis', alpha=0.5)
# plt.colorbar()
# plt.title('t-SNE visualization of embeddings')
# plt.xlabel('t-SNE component 1')
# plt.ylabel('t-SNE component 2')
# plt.savefig('tsne_visualisation_laroseda_title.png')



dataset_multiple_sep = MultipleChoiceSeparatedDataset(
                csv_file=ARC_TRAIN,
                get_embedding=None,
                emb_dim=2560,
                name=f'roarc_train_llmic',
            )

print(f"Dataset size: {len(dataset_multiple_sep)}")

tsne = TSNE(n_components=2, random_state=42, n_iter=1000, perplexity=1, verbose=1)
embeddings = []
labels = []
# for i in range(len(dataset_multiple_sep)):
#     data = dataset_multiple_sep[i]
#     embeddings.append(data[0])
#     labels.append(0)
#     correct = data[1 + data[2]]
#     labels.append(1)

# sample_idx = 8
# print(f'Sample: {dataset_multiple_sep[sample_idx]}')
# q_emb = dataset_multiple_sep[sample_idx][0]
# option_embs = dataset_multiple_sep[sample_idx][1]


# for i in range(len(option_embs)):
#     embeddings.append(option_embs[i])
#     labels.append(i)
# embeddings.append(q_emb)
# labels.append(4)
# embeddings = np.array(embeddings)
# labels = np.array(labels)
# embeddings = tsne.fit_transform(embeddings)

# plt.figure(figsize=(10, 10))
# from matplotlib.colors import ListedColormap

# distinct_colors = ListedColormap([
#     '#e6194b', '#3cb44b', '#ffe119', '#4363d8',
#     '#f58231', '#911eb4', '#46f0f0', '#f032e6',
#     '#bcf60c', '#fabebe', '#008080', '#e6beff',
#     '#9a6324', '#fffac8', '#800000', '#aaffc3',
#     '#808000', '#ffd8b1', '#000075', '#808080'
# ])

# plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap=distinct_colors, s=100, edgecolors='k')

# plt.colorbar()
# plt.title('t-SNE visualization of embeddings')
# plt.xlabel('t-SNE component 1')
# plt.ylabel('t-SNE component 2')
# plt.savefig('tsne_visualisation_laroseda_title.png')




##################### ROARC EMBEDDINGS #####################
dataset_to_visualise = MultipleChoiceCombinedDataset(
                csv_file=ARC_TRAIN,
                get_embedding=None,
                emb_dim=2560,
                name=f'roarc_train_llmic_mean',
                save_interval=1
            )

sample_idx = 0

print(f"Dataset size: {len(dataset_to_visualise)}")
print(f'Sample: {dataset_to_visualise[sample_idx]}')

#tsne visualisation
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


tsne = TSNE(n_components=2, random_state=42, n_iter=1000, perplexity=30, verbose=1)
embeddings = []
labels = []

for q in range(len(dataset_to_visualise)):
    embs, label = dataset_to_visualise[q]
    for i in range(len(embs)):
        embeddings.append(embs[i])
        if i == label:
            labels.append(1)
        else:
            labels.append(0)

embeddings = np.array(embeddings)
labels = np.array(labels)
embeddings = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 10))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.colorbar()
plt.title('t-SNE visualization of embeddings')
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
plt.savefig('tsne_visualisation_new.png')