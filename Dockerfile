FROM nvcr.io/nvidia/tensorflow:22.11-tf2-py3

RUN pip install --upgrade pip
RUN pip install pylint qutip tensorboard tensorboard-plugin-profile
# sometimes git is unhappy because Docker root doesn't own the repo
RUN git config --global --add safe.directory /workspaces/bingo 
# RUN pip install workspaces/bingo/

ENV TF_XLA_FLAGS='--tf_xla_auto_jit=2'
ENV TF_GPU_ALLOCATOR='cuda_malloc_async'