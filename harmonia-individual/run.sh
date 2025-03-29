sudo apt-get update
sudo apt-get install npm
sudo npm install -g localtunnel
export CUBLAS_WORKSPACE_CONFIG=:4096:8
git config --global credential.helper store
curl ifconfig.me
#TODO: remove secret
git clone https://EaswarGn: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@github.com/EaswarGn/chess_ai_model.git
cd ~/chess_ai_model
ls
pip install -r requirements.txt
pip install -U "huggingface_hub[cli]"
cd ~
huggingface-cli login --token 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
huggingface-cli download codingmonster1234/Chess_Star1234-data --repo-type dataset --local-dir .
cd ~
wget -O ablation_1.pt https://huggingface.co/datasets/codingmonster1234/full_trained_model/resolve/main/checkpoints/models/1900_step_118000.pt
wget https://chesstransformers.blob.core.windows.net/checkpoints/CT-EFT-20/averaged_CT-EFT-20.pt
wget -O pondering_time_step_15000.pt https://huggingface.co/datasets/codingmonster1234/pondering_time_model/resolve/main/checkpoints/models/1900_step_14000.pt
wget -O full_trained_model.pt https://huggingface.co/datasets/codingmonster1234/full_trained_model/resolve/main/checkpoints/models/1900_step_347000.pt
cd ~/chess_ai_model/harmonia-individual
python train.py individual_model
#python train_ddp.py ablation_1
#python validate_model.py ablation_1