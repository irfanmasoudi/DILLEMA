from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id='TheBloke/Llama-2-13B-chat-GGML',
    filename='llama-2-13b-chat.ggmlv3.q5_1.bin',
    local_dir='models/',
    local_dir_use_symlinks=False
)