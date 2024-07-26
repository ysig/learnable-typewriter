from huggingface_hub import snapshot_download

for model in ['copiale', 'google']:
    snapshot_download(repo_id=f"learnable-typewriter/{model}", local_dir=f"runs/{model}", repo_type='model')

for dataset in ['copiale', 'google', 'fontenay']:
    snapshot_download(repo_id=f"learnable-typewriter/{dataset}", local_dir=f"datasets/{dataset}", repo_type='dataset')
