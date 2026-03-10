
import os
import sys
import json
import cv2
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.perception.vlm_mapper import VLMMapper
from data.debug.visualize_vlm_map import create_mask_from_vlm
from src.utils.logger import get_logger

log = get_logger("label_ludicity")

def main():
    img_path = "data/debug/ludicity_raw.png"
    if not os.path.exists(img_path):
        log.error(f"找不到截图: {img_path}")
        return

    # 1. 呼叫 VLM 进行解析
    try:
        # 使用 gemini-1.5-flash 比较快且省钱，如果效果不好再换 pro
        mapper = VLMMapper(model_name="gemini-1.5-flash")
        result = mapper.analyze_map(img_path)
        
        # 保存 JSON 结果
        json_path = "data/debug/ludicity_labels.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        log.info(f"VLM 解析结果已保存至: {json_path}")
        
        # 2. 可视化
        vis_img = create_mask_from_vlm(img_path, result)
        if vis_img is not None:
            vis_path = "data/debug/ludicity_visualized.png"
            cv2.imwrite(vis_path, vis_img)
            log.info(f"可视化结果已保存至: {vis_path}")
            print(f"SUCCESS:{vis_path}")
        
    except Exception as e:
        log.error(f"标注失败: {e}")

if __name__ == "__main__":
    main()
