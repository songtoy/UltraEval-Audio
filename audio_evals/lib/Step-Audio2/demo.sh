CUDA_VISIBLE_DEVICES=7 python web_demo.py --model-path ../../pretrained_models/stepfun-ai/Step-Audio-2-mini
CUDA_VISIBLE_DEVICES=7 python web_demo_vllm.py --model-path ../../pretrained_models/stepfun-ai/Step-Audio-2-mini



vllm serve /root/zhoust/pretrained_models/stepfun-ai/Step-Audio-2-mini \
  --served-model-name step-audio-2-mini \
  --port 8000 \
  --max-model-len 16384 \
  --max-num-seqs 32 \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser step_audio_2 \
  --tokenizer-mode step_audio_2 \
  --chat_template_content_format string \
  --audio-parser step_audio_2_tts_ta4 \
  --trust-remote-code
