import gradio as gr
import json
import os

# =================配置区域=================
# 请在这里配置你的模型名称和对应的 jsonl 文件路径
# 格式: "模型显示名称": "文件路径"
MODEL_FILES = {
    # Hard-Bench 
    #"Kimi-Audio": "/root/zhoust/UltraEval-Audio/res/kimiaudio-speech/impros-hard-bench/2026-01-15_18-02-27.jsonl",
    #"Ours(ins,1e-5)": "/root/zhoust/UltraEval-Audio/res/kimia_sft_0112_persona_5_epoch_eval_lr_1e-5_cosine_eos_both-speech/impros-hard-bench/2026-01-15_18-09-25.jsonl", 
    #"Ours(ins,1e-6)": "/root/zhoust/UltraEval-Audio/res/kimia_sft_0112_persona_5_epoch_eval_lr_1e-6_cosine_eos_both-speech/impros-hard-bench/2026-01-15_19-10-29.jsonl",
    #"Ours(ins,5e-6)": "/root/zhoust/UltraEval-Audio/res/kimia_sft_0112_persona_5_epoch_eval_lr_5e-6_cosine_eos_both-speech/impros-hard-bench/2026-01-15_19-45-09.jsonl",
    #"Ours(base,1e-5)": "/root/zhoust/UltraEval-Audio/res/kimia_base_sft_0112_persona_5_epoch_eval_lr_1e-5_cosine_eos_both-speech/impros-hard-bench/2026-01-18_00-42-57.jsonl",
    #"Ours(sys_context, nothinking)": "/root/zhoust/UltraEval-Audio/res/kimia_sft_0112_persona_5_epoch_eval_lr_1e-6_cosine_eos_both_with_sys_context-speech/impros-hard-bench/2026-01-30_18-44-52.jsonl",
    #"Ours(sys_context, thinking)": "/root/zhoust/UltraEval-Audio/res/kimia_sft_0112_persona_5_epoch_eval_lr_1e-6_cosine_eos_both_with_sys_context_cot-speech/impros-hard-bench/2026-02-19_13-57-08.jsonl",
    #"Ours(sys_context, thinking, qwen)": "/root/zhoust/UltraEval-Audio/res/kimia_sft_0112_persona_5_epoch_eval_lr_1e-6_cosine_eos_both_with_sys_context_cot_qwen-speech/impros-hard-bench/2026-03-05_11-38-15.jsonl",

    # Test-Bench V1
    #"Kimi-Audio": "/root/zhoust/UltraEval-Audio/res/kimiaudio-speech/impros-test-bench/2026-01-15_17-51-28.jsonl",
    #"Ours(ins,1e-5)": "/root/zhoust/UltraEval-Audio/res/kimia_sft_0112_persona_5_epoch_eval_lr_1e-5_cosine_eos_both-speech/impros-test-bench/2026-01-15_18-17-03.jsonl", 
    #"Ours(ins,1e-6)": "/root/zhoust/UltraEval-Audio/res/kimia_sft_0112_persona_5_epoch_eval_lr_1e-6_cosine_eos_both-speech/impros-test-bench/2026-01-15_19-17-47.jsonl",
    #"Ours(ins,5e-6)": "/root/zhoust/UltraEval-Audio/res/kimia_sft_0112_persona_5_epoch_eval_lr_5e-6_cosine_eos_both-speech/impros-test-bench/2026-01-15_19-52-49.jsonl",
    #"Ours(base,1e-5)": "/root/zhoust/UltraEval-Audio/res/kimia_base_sft_0112_persona_5_epoch_eval_lr_1e-5_cosine_eos_both-speech/impros-test-bench/2026-01-18_00-50-41.jsonl",

    # Test-Bench V2
    #"Kimi-Audio": "/root/zhoust/UltraEval-Audio/res/kimiaudio-speech/impros-test-bench-V2/2026-01-23_13-44-38.jsonl",
    #"Ours(ins,1e-6)": "/root/zhoust/UltraEval-Audio/res/kimia_sft_0112_persona_5_epoch_eval_lr_1e-6_cosine_eos_both-speech/impros-test-bench-V2/2026-01-23_16-33-05.jsonl",
    #"Ours(sys_context, nothinking)": "/root/zhoust/UltraEval-Audio/res/kimia_sft_0112_persona_5_epoch_eval_lr_1e-6_cosine_eos_both_with_sys_context-speech/impros-test-bench-V2/2026-01-30_18-44-52.jsonl",
    #"Ours(sys_context, thinking)": "/root/zhoust/UltraEval-Audio/res/kimia_sft_0112_persona_5_epoch_eval_lr_1e-6_cosine_eos_both_with_sys_context_cot-speech/impros-test-bench-V2/2026-03-02_21-16-07.jsonl",
    #"Ours(sys_context, thinking, qwen)": "/root/zhoust/UltraEval-Audio/res/kimia_sft_0112_persona_5_epoch_eval_lr_1e-6_cosine_eos_both_with_sys_context_cot_qwen-speech/impros-test-bench-V2/2026-03-05_11-51-25.jsonl",

    # Alpaca-Eval
    #"Kimi-Audio": "/root/zhoust/UltraEval-Audio/res/kimiaudio-speech/speech-chatbot-alpaca-eval/2026-01-05_00-07-06.jsonl",
    #"Ours(ins,1e-6)": "/root/zhoust/UltraEval-Audio/res/kimia_sft_0112_persona_5_epoch_eval_lr_1e-6_cosine_eos_both-speech/speech-chatbot-alpaca-eval/2026-01-15_15-31-16.jsonl",



    # ImPros-Bench
    "ImPros-Bench-Std" : {
        "Qwen2.5-omni": "/root/zhoust/UltraEval-Audio/res/qwen2.5-omni-speech/impros-task-bench-s2s-multidim/2026-03-24_01-33-13.jsonl",
        "Kimi-Audio": "/root/zhoust/UltraEval-Audio/res/kimiaudio-speech/impros-task-bench-s2s-multidim/2026-03-24_01-33-13.jsonl",
        "Kimi-Audio-ImProser-D": "/root/zhoust/Impros/bench/UltraEval-Audio/res/kimia_sft_0112_persona_5_epoch_eval_lr_1e-6_cosine_eos_both_with_sys_context_qwen-speech/impros-task-bench-s2s-multidim/2026-03-24_01-33-13.jsonl",
        "Kimi-Audio-ImProser-C": "/root/zhoust/UltraEval-Audio/res/kimia_sft_0112_persona_5_epoch_eval_lr_1e-6_cosine_eos_both_with_sys_context_cot_qwen-speech/impros-task-bench-s2s-multidim/2026-03-24_01-33-13.jsonl",
        "Step-Audio2": "/root/zhoust/UltraEval-Audio/res/stepaudio2-mini-speech/impros-task-bench-s2s-multidim/2026-03-25_12-14-37.jsonl",
        "Step-Audio2-ImProser-D": "/root/zhoust/UltraEval-Audio/res/stepaudio2_sft_5_epoch_lr_1e-6_bs_64_drop_qwen-speech/impros-task-bench-s2s-multidim/2026-03-25_18-20-14.jsonl",
        "Step-Audio2-ImProser-C": "/root/zhoust/UltraEval-Audio/res/stepaudio2_sft_5_epoch_lr_1e-6_bs_64_drop_cot_qwen_bugfix-speech/impros-task-bench-s2s-multidim/2026-03-25_18-20-14.jsonl",
    },
    "ImPros-Bench-Hard" : {
        "Qwen2.5-omni": "/root/zhoust/UltraEval-Audio/res/qwen2.5-omni-speech/impros-hard-bench-s2s-multidim/2026-03-24_03-48-31.jsonl",
        "Kimi-Audio": "/root/zhoust/UltraEval-Audio/res/kimiaudio-speech/impros-hard-bench-s2s-multidim/2026-03-25_04-07-42.jsonl",
        "Kimi-Audio-ImProser-D": "/root/zhoust/Impros/bench/UltraEval-Audio/res/kimia_sft_0112_persona_5_epoch_eval_lr_1e-6_cosine_eos_both_with_sys_context_qwen-speech/impros-hard-bench-s2s-multidim/2026-03-25_04-07-42.jsonl",
        "Kimi-Audio-ImProser-C": "/root/zhoust/Impros/bench/UltraEval-Audio/res/kimia_sft_0112_persona_5_epoch_eval_lr_1e-6_cosine_eos_both_with_sys_context_cot_qwen-speech/impros-hard-bench-s2s-multidim/2026-03-24_05-30-41.jsonl",
        "Step-Audio2": "/root/zhoust/UltraEval-Audio/res/stepaudio2-mini-speech/impros-hard-bench-s2s-multidim/2026-03-25_04-07-42.jsonl",
        "Step-Audio2-ImProser-D": "/root/zhoust/UltraEval-Audio/res/stepaudio2_sft_5_epoch_lr_1e-6_bs_64_drop_qwen-speech/impros-hard-bench-s2s-multidim/2026-03-25_18-20-14.jsonl",
        "Step-Audio2-ImProser-C": "/root/zhoust/UltraEval-Audio/res/stepaudio2_sft_5_epoch_lr_1e-6_bs_64_drop_cot_qwen_bugfix-speech/impros-hard-bench-s2s-multidim/2026-03-25_18-20-14.jsonl",
    }
}
# =========================================

def parse_jsonl(file_path):
    """
    解析单个 jsonl 文件，返回以 id 为 key 的字典。
    结构: {
        0: {
            "prompt_audio": "path...", 
            "response_text": "...", 
            "response_audio": "...", 
            "score": 2
        },
        ...
    }
    """
    data_map = {}
    
    if not os.path.exists(file_path):
        print(f"Warning: File not found {file_path}")
        return {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                row_id = item.get("id")
                row_type = item.get("type")
                row_data = item.get("data", {})

                if row_id not in data_map:
                    data_map[row_id] = {}

                # 1. 解析 Prompt (User Audio)
                if row_type == "prompt":
                    # 根据你的结构: data -> content[0] -> contents[0] -> value
                    try:
                        audio_path = row_data["content"][0]["contents"][0]["value"]
                        data_map[row_id]["prompt_audio"] = audio_path
                    except (KeyError, IndexError):
                        pass

                # 2. 解析 Inference (Model Response)
                elif row_type == "inference":
                    # 根据你的结构: data -> content 是一个 stringified json
                    try:
                        content_str = row_data.get("content", "{}")
                        content_json = json.loads(content_str)
                        data_map[row_id]["response_text"] = content_json.get("text", "")
                        data_map[row_id]["response_audio"] = content_json.get("audio", None)
                    except json.JSONDecodeError:
                        pass

                # 3. 解析 Eval (Score)
                elif row_type == "eval":
                    data_map[row_id]["score"] = row_data.get("geval", "N/A")

            except json.JSONDecodeError:
                continue
    
    return data_map

def load_all_data():
    """
    加载所有模型的数据并进行对齐
    """
    # 结构: [ { "id": 0, "prompt": "...", "models": {"ModelA": {...}, "ModelB": {...}} }, ... ]
    all_data_map = {} # key: id
    
    model_names = list(MODEL_FILES.keys())

    for model_name, file_path in MODEL_FILES.items():
        model_data = parse_jsonl(file_path)
        
        for record_id, record_content in model_data.items():
            if record_id not in all_data_map:
                all_data_map[record_id] = {
                    "id": record_id,
                    "prompt_audio": None,
                    "models": {}
                }
            
            # 如果是该ID第一次被处理，或者当前还没有 Prompt 音频，则存入
            # (假设所有模型的 Prompt 是一样的，取第一个读到的即可)
            if "prompt_audio" in record_content and not all_data_map[record_id]["prompt_audio"]:
                all_data_map[record_id]["prompt_audio"] = record_content["prompt_audio"]

            all_data_map[record_id]["models"][model_name] = {
                "text": record_content.get("response_text", ""),
                "audio": record_content.get("response_audio", None),
                "score": record_content.get("score", "N/A")
            }

    # 转换为按 ID 排序的列表
    sorted_ids = sorted(all_data_map.keys())
    final_data_list = [all_data_map[uid] for uid in sorted_ids]
    return final_data_list, model_names

# 加载数据
data_list, model_names = load_all_data()
total_samples = len(data_list)

def get_sample_data(index):
    if index < 0 or index >= len(data_list):
        return [None] * (1 + len(model_names) * 3) # Return empties
    
    sample = data_list[index]
    
    # 1. Prompt Audio
    results = [sample["prompt_audio"]]
    
    # 2. Loop through models to get Text, Audio, Score
    for name in model_names:
        model_info = sample["models"].get(name, {})
        results.append(model_info.get("text", ""))
        results.append(model_info.get("audio", None))
        results.append(model_info.get("score", 0))
        
    return results

# =================Gradio 构建=================
with gr.Blocks(title="Audio Model Eval Visualization") as demo:
    gr.Markdown("## LLM-as-a-Judge Audio Eval Visualization")
    
    # 状态存储
    current_index = gr.State(0)

    # 导航栏
    with gr.Row():
        prev_btn = gr.Button("Previous", scale=1)
        index_slider = gr.Slider(minimum=0, maximum=max(0, total_samples-1), step=1, label="Sample ID", value=0, scale=8)
        next_btn = gr.Button("Next", scale=1)

    # 主内容区
    with gr.Row():
        # 左侧：Prompt (Common)
        with gr.Column(scale=1):
            gr.Markdown("### User Input (Prompt)")
            prompt_audio_player = gr.Audio(label="User Audio", type="filepath", interactive=False)
            # 如果有 Prompt 文本也可以加在这里

    gr.HTML("<hr style='margin-top: 10px; margin-bottom: 10px; border: 1px solid #ddd;' />")

    # 3. 模型结果对比区 (一行多列)
    model_components = [] # 存储所有需要更新的模型组件

    with gr.Row():
        for name in model_names:
            with gr.Column(variant="panel"): # 使用 panel 样式让每列有个边框，区分更明显
                gr.Markdown(f"### 🤖 {name}")
                
                # 按照 get_sample_data 的顺序创建组件: Text -> Audio -> Score
                m_text = gr.Textbox(label="Response Text", interactive=False, lines=4)
                m_audio = gr.Audio(label="Response Audio", type="filepath", interactive=False)
                m_score = gr.Number(label="G-Eval Score", interactive=False)
                
                model_components.extend([m_text, m_audio, m_score])

    # 事件处理
    all_outputs = [prompt_audio_player] + model_components

    def update_view(idx):
        return get_sample_data(idx)

    def on_prev(idx):
        new_idx = max(0, idx - 1)
        return [new_idx] + update_view(new_idx)

    def on_next(idx):
        new_idx = min(total_samples - 1, idx + 1)
        return [new_idx] + update_view(new_idx)
    
    def on_slider(idx):
        return update_view(idx)

    # 绑定事件
    # Slider 变动
    index_slider.change(fn=on_slider, inputs=[index_slider], outputs=all_outputs)
    
    # 按钮变动 (需要同时更新 Slider 和 内容)
    prev_btn.click(fn=on_prev, inputs=[current_index], outputs=[index_slider] + all_outputs) \
            .then(fn=lambda x: x, inputs=[index_slider], outputs=[current_index])
            
    next_btn.click(fn=on_next, inputs=[current_index], outputs=[index_slider] + all_outputs) \
            .then(fn=lambda x: x, inputs=[index_slider], outputs=[current_index])

    # 初始化加载第一个
    demo.load(fn=lambda: update_view(0), outputs=all_outputs)

if __name__ == "__main__":
    if total_samples == 0:
        print("Error: No data loaded. Please check your JSONL paths.")
    else:
        print(f"Loaded {total_samples} samples with models: {model_names}")
        demo.launch(server_name="0.0.0.0", server_port=7861, allowed_paths=["/mnt/zhoust/Impros", "/root/zhoust/Impros"] )