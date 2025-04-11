

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"


import pandas as pd
from datasets import Dataset
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from sentence_transformers import SentenceTransformer


# In[4]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[5]:


Train_Data_Frame = pd.read_csv('newtrain.csv')
# Train_Data_Frame= Train_Data_Frame.sample(n=1000, random_state=45)

Train_Data_Frame.head()


# In[6]:



# Train_Dataset_Paragraphs_Column_Name = 'transcript'
# Train_Dataset_Summaries_Column_Name = 'QUESTION HINDI'


# In[7]:


Train_Data_Frame = Dataset.from_pandas(Train_Data_Frame)
Train_Data_Frame = Train_Data_Frame.train_test_split(test_size=0.2)


# In[8]:


Train_Data_Frame["train"][0]


# In[9]:


# Train_Data_Frame["test"][0]
import os



# In[10]:


import sys
checkpoint="/DATA/deepjyoti_2311ai65/FVIDEOMAEBARTHINDI1MODEL"
sys.path.insert(0, checkpoint)
from transformers import  AutoTokenizer,MBart50TokenizerFast, MBartForConditionalGeneration
# from modeling_mbart import MBartForConditionalGeneration
model = MBartForConditionalGeneration.from_pretrained(checkpoint).to(device)
tokenizer = MBart50TokenizerFast.from_pretrained(checkpoint)

# from peft import get_peft_config, get_peft_model, LoraConfig, TaskType,PeftConfig


# peft_config = LoraConfig(
#     task_type=TaskType.SEQ_2_SEQ_LM, 
#     inference_mode=False, 
#     r=4, 
#     lora_alpha=8, 
#     lora_dropout=0.1,
#     target_modules=['fc2', 'fc1','k_proj','v_proj','q_proj']
# )

# # Wrap the model with PEFT
# peft_model = get_peft_model(model, peft_config)

# # from transformers import  AutoTokenizer,MBartForConditionalGeneration
# # from modeling_mbart import MBartForConditionalGeneration
# # tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX", tgt_lang="hi_IN")
# # mbart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25").to(device)
# from transformers import  AutoTokenizer,MBart50TokenizerFast,MBartForConditionalGeneration
# # from modeling_mbart import MBartForConditionalGeneration
# # tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX", tgt_lang="hi_IN")
# # mbart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25").to(device)
# mbart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50").to(device)
# tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="hi_IN",use_fast=False)
Semantic_Model = SentenceTransformer('paraphrase-MiniLM-L6-v2')




# mbart_model

# In[11]:


# peft_model


# In[12]:


# Prefix="Generate a Codemix question from the given Transcript. Transcript: {transcript}."
Prefix="Generate a Codemix question from the given video and transcript(60 percent Hindi and 40 percent English). Transcript: {transcript}."
# Paragraph_Max_Length = 1024
# Summary_Max_Length = 100
Embedding_Dimensions = (1, 1568, 1024)
batch_size = 4


# In[13]:


# train_split = Train_Data_Frame['train']  # Assuming 'train' is one of the splits

# # Convert the 'train' split to a pandas DataFrame
# train_df = pd.DataFrame(train_split)


# In[14]:


# # print(train_df.column_na)  # If it's a Dataset object

# # OR if it's a pandas DataFrame
# print(train_df.columns)


# In[15]:


# print(Train_Data_Frame.columns)


# In[16]:


import h5py
import pandas as pd

# Load HDF5 file and check if the vid_name values from the CSV exist as keys in the HDF5 file
def filter_records_by_h5(Records, hdf5_path='video_embed_f.h5'):
    with h5py.File(hdf5_path, 'r') as hf:
        valid_vid_names = []
        for vid_name in Records['vid_name']:
            if vid_name in hf.keys():  # Check if the value in the CSV exists as a key in the HDF5 file
                valid_vid_names.append(vid_name)
            else:
                print(f"Warning: Video name '{vid_name}' not found in HDF5 file.")
    
    filtered_Records = Records[Records['vid_name'].isin(valid_vid_names)].reset_index(drop=True)
    return filtered_Records

train_dataset = Train_Data_Frame['train'].to_pandas()   # Assuming there is a 'train' split
filtered_Records = filter_records_by_h5(train_dataset)


# In[17]:


test_dataset=Train_Data_Frame['test'].to_pandas()
filtered2=filter_records_by_h5(test_dataset)


# In[18]:


filtered_Records.info()


# In[19]:


# train_dataset=Dataset.from_pandas(Train_Data_Frame)


# In[20]:


def load_video_embedding(video_name, hdf5_path='video_embed_f.h5'):
    video_name = str(video_name)
    with h5py.File(hdf5_path, 'r') as hf:
        if video_name in hf:
            embeddings = hf[video_name][:]
            return torch.tensor(embeddings)
        else:
            print(f"Warning: Video name '{video_name}' not found in HDF5 file.")
            return torch.zeros(Embedding_Dimensions)


# In[21]:




def Tokenization(Records):
    Paragraphs = []
    video_embeds = []
    # Prefix="Generate a Hindi question from the Transcript and don't generate question that are not related to the transcript.Transcript:{transcript}"
    # The batch Records is a dictionary, so we iterate over its items
    for i in range(len(Records['vid_name'])):
        video_name = Records['vid_name'][i]  # Extract video name from the current batch entry
        # transcript = Records['transcript'][i] 
        transcript=""
        # # Extract transcript from the current batch entry

        try:
            video_embed = load_video_embedding(video_name)
            video_embeds.append(video_embed)
        except KeyError as e:
            print(e)
            video_embeds.append(torch.zeros((1, 1568, 1024)))  # Append zero embedding for missing video
        final_prefix = Prefix.format(transcript=transcript)
        Paragraphs.append(final_prefix + transcript)
        # Paragraphs.append(Prefix)

    Final_Tokenized_Inputs = tokenizer(
        Paragraphs, 
        max_length=512, 
        truncation=True, 
        padding="max_length", 
        return_tensors="pt"
    )
    
    # question_labels = Records['QUESTION HINDI']
    # question_labels = [str(label) if label is not None else "" for label in question_labels]

    # with tokenizer.as_target_tokenizer():
    Labels = tokenizer(
        Records['QUESTION CODEMIX'],
        max_length=128, 
        truncation=True, 
        padding="max_length", 
        return_tensors="pt"
    )

    Final_Tokenized_Inputs["labels"] = Labels["input_ids"]
    Final_Tokenized_Inputs["video_embed"] = video_embeds

    return Final_Tokenized_Inputs


# Tokenized_Records = Tokenization(Train_Data_Frame)
Tokenized_Records = Train_Data_Frame.map(Tokenization, batched=True)

# Verify the tokenized records
print(Tokenized_Records)


# In[22]:


# Tokenized_Records['train'][0]


# In[23]:


# Tokenized_Records['train']


# In[24]:



def collate_fn(batch):
    # print("Batch:", batch)
    # input_ids = torch.tensor([item['input_ids'] for item in batch])
    # attention_mask = torch.tensor([item['attention_mask'] for item in batch])
    # # video_embed = torch.stack([torch.tensor(item['video_embed']) for item in batch]).to(device)
    # labels = torch.tensor([item['labels'] for item in batch])
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    attention_mask = [torch.tensor(item['attention_mask']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]
    video_embeds = [item['video_embed'] for item in batch]
    
    # print(f"Original input_ids lengths: {[len(i) for i in input_ids]}")
    # print(f"Original labels lengths: {[len(l) for l in labels]}")
    # inputs_text = torch.nn.utils.rnn.pad_sequence([torch.tensor(lst) for lst in input_ids], batch_first=True, padding_value=0)
    # labels_text = torch.nn.utils.rnn.pad_sequence([torch.tensor(lst) for lst in labels], batch_first=True, padding_value=0)
    # attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(lst) for lst in attention_mask], batch_first=True, padding_value=0)
    
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    
    # print(f"Padded input_ids shape: {padded_input_ids.shape}")
    # print(f"Padded labels shape: {padded_labels.shape}")
    # input_ids = torch.tensor(input_ids)
    # attention_mask = torch.tensor(attention_mask)
    # video_embed = torch.stack([torch.tensor(v) for v in video_embed])  
    # labels = torch.tensor(labels)

    return {
        # "input_ids": input_ids,
        # "attention_mask": attention_mask,
        # "video_embedd": video_embeds,  # Your video embeddings
        # "labels": labels
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "video_embedd": video_embeds,  # Your video embeddings
        "labels": padded_labels
    }







# In[27]:



from rouge_score import rouge_scorer
import numpy as np

def Compute_Metrics_With_Rouge(Eval_Pred):
    Predictions, Labels = Eval_Pred
    Decoded_Preds = tokenizer.batch_decode(Predictions, skip_special_tokens=True)
    Labels = np.where(Labels != -100, Labels, tokenizer.pad_token_id)
    Decoded_Labels = tokenizer.batch_decode(Labels, skip_special_tokens=True)

    similarity_scores = []
    rouge_l_scores = []

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for pred, label in zip(Decoded_Preds, Decoded_Labels):
        pred = pred[len(Prefix):]  # Assuming Prefix is defined elsewhere
        embeddings = Semantic_Model.encode([pred, label], convert_to_tensor=True)

        # Compute cosine distance using NumPy
        embedding1 = embeddings[0].cpu().numpy()
        embedding2 = embeddings[1].cpu().numpy()
        
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        cosine_similarity = dot_product / (norm1 * norm2)  # Cosine similarity
        cosine_distance = 1 - cosine_similarity  # Cosine distance
        similarity_scores.append(1 - cosine_distance)  # Mimicking original behavior

        # Compute ROUGE scores
        rouge_scores = scorer.score(label, pred)
        rouge_l_scores.append(rouge_scores['rougeL'].fmeasure)

    Prediction_Lens = [np.count_nonzero(Pred != tokenizer.pad_token_id) for Pred in Predictions]

    Result = {
        "gen_len": np.mean(Prediction_Lens),
        "semantic_similarity": np.mean(similarity_scores),
        "rougeL": np.mean(rouge_l_scores),
    }
    print("Decoded Predictions:", Decoded_Preds)
    print("Decoded Labels:", Decoded_Labels)

    return {k: round(v, 4) for k, v in Result.items()}




# In[29]:

import wandb
wandb.init(project="MBARTLARGE50_codemix_F_NOPEFT")


# In[30]:


from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback

training_args = Seq2SeqTrainingArguments(
    output_dir="MBARTLARGE50_codemix_F_NOPEFT",
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=8,
    gradient_accumulation_steps=2,
    predict_with_generate=True,
    fp16=False,  
    logging_dir="./logs",  #
    logging_steps=50,     
    report_to="wandb",
    learning_rate=1e-5,  #less than 1e
    remove_unused_columns=False,
    warmup_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    max_grad_norm=1.0
)


trainer = Seq2SeqTrainer(
    model=model.to(device),  
    args=training_args,
    train_dataset=Tokenized_Records["train"],
    eval_dataset=Tokenized_Records["test"],
    tokenizer=tokenizer,
    data_collator=collate_fn,
    compute_metrics=Compute_Metrics_With_Rouge,
)

# Start training
trainer.train()
trainer.save_model("MBARTLARGE50_codemix_F_NOPEFT")


# Evaluate the model and store the results in a dictionary
test_results = trainer.evaluate()
print(f"Final evaluation results: {test_results}")

# Convert the results dictionary to a pandas DataFrame
results_df = pd.DataFrame([test_results])

# Save the DataFrame to a CSV file
results_df.to_csv("final_evaluation_results_MBARTLARGE50_codemix_F_NOPEFT.csv", index=False)

print("Evaluation results saved to final_evaluation_results.csv")

