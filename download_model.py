from huggingface_hub import snapshot_download

model_id = "vineetsharma/qlora-adapter-Llama-2-7b-hf-TweetSumm"
local_dir = "./tweet-summary"

# Download the model
snapshot_download(repo_id=model_id, local_dir=local_dir)
