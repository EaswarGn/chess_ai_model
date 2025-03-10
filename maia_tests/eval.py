from maia2 import model, dataset, inference
from tqdm import tqdm

maia2_model = model.from_pretrained(type="rapid", device="cpu")
data = dataset.load_example_test_dataset()
prepared = inference.prepare()
correct = 0
total = len(data)

for fen, move, elo_self, elo_oppo in tqdm(data.values, desc="Processing FENs"):
    move_probs, win_prob = inference.inference_each(maia2_model, prepared, fen, elo_self, elo_oppo)
    
    if max(move_probs, key=move_probs.get) == move:
        correct += 1
print(correct/total)