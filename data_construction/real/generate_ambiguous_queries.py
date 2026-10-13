import json
import os
import hashlib
import re
import asyncio
from typing import Dict, Any, Set
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI, APIError, APITimeoutError

# --- 核心配置 ---
INPUT_FILE = os.environ.get('AMBIREFER3D_REAL_INPUT_JSON', 'data/real/descriptions.json')
OUTPUT_FILE = os.environ.get('AMBIREFER3D_REAL_AMBIGUOUS_JSONL', 'data/real/ambiguous_queries.jsonl') 

# 请确保端口正确
MODEL_NAME = os.environ.get('AMBIREFER3D_TEXT_MODEL', 'qwen-llm') 
VLLM_API_BASE = os.environ.get('AMBIREFER3D_TEXT_API_BASE', 'http://localhost:8200/v1') 
VLLM_API_KEY = os.environ.get('AMBIREFER3D_API_KEY', 'EMPTY') 

# --- 策略配置 ---
CONCURRENCY_LIMIT = 20     # 稍微调高一点并发，VLLM应该能扛得住
TIMEOUT_SECONDS = 600      
MAX_TOKENS = 1024         

SYSTEM_PROMPT = """
You are a data generator constructing an "Ambiguous User Query" dataset.
Your task is to take a detailed object description and generate 3 distinct "degraded" (ambiguous) versions of it by removing specific information.

Input: A detailed Ground Truth Description.
Output: A JSON object containing the extracted components and 3 variants of ambiguous queries.

VARIANTS TO GENERATE:
1. "no_spatial": Keep the object class and visual attributes (color, material, shape), but REMOVE all location info (e.g., "in the corner", "next to table").
2. "no_visual": Keep the object class and location info, but REMOVE all visual attributes (color, material, shape).
3. "minimal": REMOVE BOTH visual attributes and spatial info. Keep only the object class and a determiner (e.g., "the cabinet", "that chair").

Output Format (JSON ONLY):
{
  "parsed_components": {
    "class": "string",
    "visual_attributes": ["string", "string"],
    "spatial_relations": ["string", "string"]
  },
  "ambiguous_variants": [
    {
      "type": "no_spatial",
      "query": "string (the actual query text)",
      "removed_info": ["spatial"]
    },
    {
      "type": "no_visual",
      "query": "string (the actual query text)",
      "removed_info": ["visual_attributes"]
    },
    {
      "type": "minimal",
      "query": "string (the actual query text)",
      "removed_info": ["spatial", "visual_attributes"]
    }
  ]
}
"""

client = AsyncOpenAI(api_key=VLLM_API_KEY, base_url=VLLM_API_BASE)

def generate_fingerprint(item: Dict[str, Any]) -> str:
    """
    【修复核心】生成绝对唯一的指纹。
    强制加入 description 的哈希。这样即使 ann_id 一样，
    只要描述文本不同，就会被视为不同的任务。
    """
    scene = str(item.get("scene_id", "unknown"))
    ann = str(item.get("ann_id", "none"))
    desc = item.get("description", "").strip()
    
    # 关键：加上描述的 Hash，彻底解决指纹冲突问题
    desc_hash = hashlib.md5(desc.encode('utf-8')).hexdigest()[:8]
    
    return f"{scene}|{ann}|{desc_hash}"

def load_processed_ids(output_file: str) -> Set[str]:
    """读取已保存文件，按新逻辑重新计算指纹"""
    processed = set()
    if not os.path.exists(output_file): return processed
    print(f"正在扫描已完成数据: {output_file} ...")
    valid_count = 0
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
                # 使用原始数据的字段重建指纹
                temp_item = {
                    "scene_id": data.get("scene_id"),
                    "ann_id": data.get("ann_id"),
                    "object_id": data.get("object_id"),
                    "description": data.get("original_description", "")
                }
                # 重新计算指纹，确保与 generate_fingerprint 逻辑一致
                processed.add(generate_fingerprint(temp_item))
                valid_count += 1
            except json.JSONDecodeError: continue
            except Exception: continue
            
    print(f" -> 扫描完毕: 文件行数 {valid_count} -> 唯一任务指纹 {len(processed)}")
    return processed

def clean_json_content(content: str) -> str:
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    if "```" in content:
        match = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
        if match: content = match.group(1).strip()
    start_idx = content.find('{')
    end_idx = content.rfind('}')
    if start_idx != -1 and end_idx != -1:
        content = content[start_idx : end_idx + 1]
    return content

async def process_item_with_retry(item: Dict[str, Any], semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    gt_desc = item.get("description", "")
    if not gt_desc: return None
    
    async with semaphore:
        attempt = 0
        while True:
            try:
                user_msg = f'Ground Truth: "{gt_desc}"\nGenerate the JSON response.'
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg}
                ]
                
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=0.1,
                        max_tokens=MAX_TOKENS,
                    ),
                    timeout=TIMEOUT_SECONDS
                )
                
                raw_content = response.choices[0].message.content
                content = clean_json_content(raw_content)
                generated_data = json.loads(content)
                
                return {
                    "scene_id": item.get("scene_id"),
                    "object_id": item.get("object_id"),
                    "ann_id": item.get("ann_id"),
                    "original_description": gt_desc,
                    "generated_ambiguity": generated_data
                }

            except (asyncio.TimeoutError, APITimeoutError):
                attempt += 1
                await asyncio.sleep(2)
            except (APIError, Exception):
                attempt += 1
                await asyncio.sleep(5)

async def main():
    print(f"正在加载输入文件: {INPUT_FILE} ...")
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("找不到输入文件")
        return

    items = data if isinstance(data, list) else data.get("details", [])
    total_input = len(items)
    
    # 1. 获取已完成的ID集合 (使用新指纹逻辑)
    processed_ids = load_processed_ids(OUTPUT_FILE)
    
    # 2. 查漏补缺
    todo_items = []
    
    print("正在进行精确差量对比...")
    for item in items:
        fid = generate_fingerprint(item)
        if fid not in processed_ids:
            todo_items.append(item)

    skipped_count = total_input - len(todo_items)

    print("-" * 50)
    print(f" 输入总数: {total_input}")
    print(f" 文件已存: {len(processed_ids)} (唯一)") # 注意看这里的日志，是新代码的特征
    print(f" 跳过重复: {skipped_count}")
    print(f" **本次需跑: {len(todo_items)}** (确保不漏数据)")
    print("-" * 50)

    if not todo_items:
        print("✅ 检测到所有数据都已完美存在，无需运行！")
        return

    print(f"\n🚀 开始执行 (并发: {CONCURRENCY_LIMIT})...")
    
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = [process_item_with_retry(item, semaphore) for item in todo_items]
    
    with open(OUTPUT_FILE, "a", encoding="utf-8", buffering=1) as f_out:
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            res = await coro
            if res:
                f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
                f_out.flush()
                
    print(f"\n✅ 全部完成！")

if __name__ == "__main__":
    asyncio.run(main())