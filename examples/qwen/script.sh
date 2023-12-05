# multi-gpu, use mpirun to run
mpirun -n 2 --allow-run-as-root python run_trt.py --engine_dir trt_engines/float16/
tp1_pp2


python build.py --world_size 2 --pp_size 2 --model_dir ./qwen_7b_chat --dtype float16 --use_gpt_attention_plugin float16 --use_gemm_plugin float16 --remove_input_padding


sha256sum -c ccdv___cnn_dailymail.tar.xz.sha256sum.txt

mkdir -p ~/.cache/huggingface

tar -xvf ccdv___cnn_dailymail.tar.xz -C ~/.cache/huggingface