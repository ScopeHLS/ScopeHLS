import os
import torch
from transformers import AutoTokenizer, AutoModel
from config.config import CONFIG

model_name = 'Salesforce/codet5p-770m'

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(CONFIG.codet5p_tokenizer_path)

model = AutoModel.from_pretrained(model_name)
model.save_pretrained(CONFIG.codet5p_path)

from model.model import pragma_cross_code as Net

model = Net()

model_path = os.path.join(CONFIG.model_base_dir, CONFIG.model_stage3_path)
torch.save(model.state_dict(), model_path)
