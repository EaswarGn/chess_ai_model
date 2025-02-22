sudo apt-get install npm
npm install -g localtunnel
git config --global credential.helper store
curl ifconfig.me
git clone https://EaswarGn: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@github.com/EaswarGn/chess_ai_model.git
cd ~/chess_ai_model
ls
pip install -r requirements.txt
pip install -U "huggingface_hub[cli]"
huggingface-cli login
huggingface-cli download codingmonster1234/test-repo --repo-type dataset --local-dir .
chmod +x ~/chess_ai_model/data_saving/c++/unzip_files.sh
cd ~/1900_zipped_training_chunks
~/chess_ai_model/data_saving/c++/unzip_files.sh
cd ~/ranged_chunks_zipped
~/chess_ai_model/data_saving/c++/unzip_files.sh
cd ~/chess_ai_model/training
wget https://huggingface.co/datasets/codingmonster1234/ablation_8_with_checkpoint/resolve/main/checkpoints/models/1900_step_60000.pt
python train.py ablation_1