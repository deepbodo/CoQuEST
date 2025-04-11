

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


# In[2]:


import pandas as pd
from datasets import Dataset
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from sentence_transformers import SentenceTransformer


# In[3]:



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[4]:



Train_Data_Frame = pd.read_csv('train.csv')
# Train_Data_Frame= Train_Data_Frame.sample(n=50, random_state=45)

Train_Data_Frame.head()


# In[5]:





# In[6]:



Train_Data_Frame = Dataset.from_pandas(Train_Data_Frame)
Train_Data_Frame = Train_Data_Frame.train_test_split(test_size=0.2)


# In[7]:



Train_Data_Frame["train"][0]


# In[8]:



from transformers import  AutoTokenizer,MBartForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50")
mbart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50").to(device)
Semantic_Model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


# In[9]:


mbart_model


# In[10]:


Prefix="Generate a Codemix question from the given video and transcript(60 percent Hindi and 40 percent English). Transcript: {transcript}."
Paragraph_Max_Length = 1024
Summary_Max_Length = 100
Embedding_Dimensions = (1, 1568, 1024)
batch_size = 2


# In[11]:


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


# In[12]:



test_dataset=Train_Data_Frame['test'].to_pandas()
filtered2=filter_records_by_h5(test_dataset)


# In[13]:




filtered_Records.info()


# In[14]:


def load_video_embedding(video_name, hdf5_path='video_embed_f.h5'): 
    video_name = str(video_name)
    with h5py.File(hdf5_path, 'r') as hf:
        if video_name in hf:
            embeddings = hf[video_name][:]
            return torch.tensor(embeddings)
        else:
            print(f"Warning: Video name '{video_name}' not found in HDF5 file.")
            return torch.zeros(Embedding_Dimensions)


# In[ ]:


def Tokenization(Records):
    Paragraphs = []
    video_embeds = []
    # Prefix="Generate a Hindi question from the Transcript and don't generate question that are not related to the transcript.Transcript:{transcript}"
    # The batch Records is a dictionary, so we iterate over its items
    for i in range(len(Records['vid_name'])):
        video_name = Records['vid_name'][i]  # Extract video name from the current batch entry
        transcript = Records['Transcript'][i]  # Extract transcript from the current batch entry

        try:
            video_embed = load_video_embedding(video_name)
            video_embeds.append(video_embed)
        except KeyError as e:
            print(e)
            video_embeds.append(torch.zeros((1, 1568, 1024)))  # Append zero embedding for missing video
        final_prefix = Prefix.format(transcript=transcript)
        Paragraphs.append(final_prefix + transcript)


    Final_Tokenized_Inputs = tokenizer(
        Paragraphs, 
        max_length=1024, 
        truncation=True, 
        padding="max_length", 
        return_tensors="pt"
    )
    
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
Tokenized_Records = Train_Data_Frame.map(Tokenization, batched=True)


# In[16]:


print(Tokenized_Records)


# In[17]:


def collate_fn(batch):

    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    attention_mask = [torch.tensor(item['attention_mask']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]
    video_embeds = [item['video_embed'] for item in batch]
    

    padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    


    return {
 
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "video_embedd": video_embeds,  # Your video embeddings
        "labels": padded_labels
    }


# In[18]:


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


# In[19]:


import wandb
wandb.init(project="FVIDEOMAEBARTDirectCodemix")


# In[20]:



training_args = Seq2SeqTrainingArguments(
    output_dir="FVIDEOMAEBARTDirectCodemix",
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
    model=mbart_model.to(device),  # Ensure model is moved to the correct device
    args=training_args,
    train_dataset=Tokenized_Records["train"],
    eval_dataset=Tokenized_Records["test"],
    tokenizer=tokenizer,
    data_collator=collate_fn,
    compute_metrics=Compute_Metrics_With_Rouge,
)

# Start training
trainer.train()
trainer.save_model("FVIDEOMAEBARTDirectCodemix")


# In[ ]:


test_results = trainer.evaluate()
print(f"Final evaluation results: {test_results}")

# Convert the results dictionary to a pandas DataFrame
results_df = pd.DataFrame([test_results])


# In[ ]:


results_df.to_csv("final_evaluation_results_directcodemix_nopeft.csv", index=False)

print("Evaluation results saved to final_evaluation_results.csv")



