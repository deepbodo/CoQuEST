import os
os.environ['CUDA_VISIBLE_DEVICES']="2"
import pandas as pd
import h5py
import torch
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

# Paths
checkpoint_path = "MBARTLARGE50_codemix_F_NOPEFT"
hdf5_path = 'video_embeddings_test.h5'
input_csv_path = 'test_data.csv'  # Replace with the actual test set CSV path
output_csv_path = 'testoutput_with_generated_questions_CoQuEST.csv'  # Replace with desired output CSV path

device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_dimensions = 1024

# Load model and tokenizer
model = MBartForConditionalGeneration.from_pretrained(checkpoint_path).to(device)
tokenizer = MBart50TokenizerFast.from_pretrained(checkpoint_path)

# Function to load video embedding from HDF5 file
def load_video_embedding(video_name, hdf5_path):
    with h5py.File(hdf5_path, 'r') as hf:
        if video_name in hf:
            embeddings = hf[video_name][:]
            return torch.tensor(embeddings)
        else:
            print(f"Warning: Video name '{video_name}' not found in HDF5 file.")
            return None

# Function to generate questions
def generate_questions_with_prompt(model, tokenizer, transcript, video_embedding,
                                   prompt="Generate a Codemix question from the given video and transcript (60% Hindi and 40% English). Transcript: {transcript}."):
    input_text = prompt.format(transcript=transcript)
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Move inputs and embedding to the model's device
    inputs = inputs.to(model.device)
    video_embeds = video_embedding.to(model.device)

    # Generate question
    with torch.no_grad():
        output_ids = model.generate(
            inputs.input_ids,
            video_embedd=video_embeds,  # Pass the video embedding here
            max_length=128,
            num_beams=6,
            temperature=0.5,
            top_k=10,  
            top_p=0.5,
            early_stopping=True
        )

    # Decode the generated question
    generated_question = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return generated_question

# Load the test set CSV
data = pd.read_csv(input_csv_path)

# Process each row in the test set
results = []
for _, row in data.iterrows():
    vid_name = str(row['vid_name'])
    transcript = row['Transcript']

    # Load video embedding
    video_embedd = load_video_embedding(vid_name, hdf5_path)

    if video_embedd is not None:
        # Generate question
        try:
            generated_question = generate_questions_with_prompt(model, tokenizer, transcript, video_embedd)
        except Exception as e:
            print(f"Error generating question for video '{vid_name}': {e}")
            generated_question = None
    else:
        generated_question = None

    # Add the result to the list
    row['Generated_Question_HINGBERT'] = generated_question
    results.append(row)

# Create a DataFrame with the results
results_df = pd.DataFrame(results)

# Save the results to a new CSV file
results_df.to_csv(output_csv_path, index=False)

print(f"Processing complete. Results saved to {output_csv_path}.")
