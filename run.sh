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
huggingface-cli download codingmonster1234/test-repo --repo-type dataset --local-dir .
chmod +x ~/chess_ai_model/data_saving/c++/unzip_files.sh
cd ~/1900_zipped_training_chunks
~/chess_ai_model/data_saving/c++/unzip_files.sh
cd ~/ranged_chunks_zipped
~/chess_ai_model/data_saving/c++/unzip_files.sh
cd ~
wget -O ablation_1.pt https://huggingface.co/datasets/codingmonster1234/full_trained_model/resolve/main/checkpoints/models/1900_step_118000.pt
wget https://chesstransformers.blob.core.windows.net/checkpoints/CT-EFT-20/averaged_CT-EFT-20.pt
wget -O pondering_time_step_15000.pt https://huggingface.co/datasets/codingmonster1234/pondering_time_model/resolve/main/checkpoints/models/1900_step_14000.pt
cd ~/chess_ai_model/training/ddp_training
#python train_ddp.py pondering_time_model
python train_ddp.py ablation_4
#python validate_model.py ablation_1