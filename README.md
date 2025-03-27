conda create -n dnn_env python=3.12.3 -y
conda activate dnn_env

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets accelerate
pip install protobuf sentencepiece

Pentru LLama3 e nevoie sa mergem pe pagina principala a modeluli -> log-in -> request access
Dupa in terminalul proiectului:  `huggingface-cli login`

https://huggingface.co/datasets/OpenLLM-Ro/ro_arc_challenge/viewer/default/train?row=1&views%5B%5D=train

TODO
- clasa abstracta cu functii care sa abstractizeze modelul
- download la celelalte modele
- download la roARC
- de facut MLP
- preprocesare embeddings pe fiecare model