"""
大语言模型建图引擎 - V3.0 "感" 核心

功能：
1. 初到一个新地图时，截取全屏高清画面 (1600x900)
2. 调用 Gemini 1.5 Pro / Flash 等多模态大语言模型 (VLM)
3. 给定 Prompt，要求大模型输出包含【主要地形平台】、【绳子/梯子】、【所有可见怪物】的结构化 JSON。
4. 在本地将其降维解析成 2D 坐标系，作为底层 A* 寻路的傻瓜挂 Nodes（节点）。
"""

import os
import json
import base64
from typing import Dict, Any

try:
    import google.generativeai as genai
    from PIL import Image
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

from src.utils.logger import get_logger

log = get_logger("vlm_mapper")


class VLMMapper:
    """
    负责连接 VLM (Gemini) 将游戏截图转换为结构化 NavMesh 数据。
    """
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-pro"):
        """
        初始化 Gemini 引擎。
        
        Args:
            api_key: Gemini API Key。如果不传，会自动尝试从环境变量 GEMINI_API_KEY 获取。
            model_name: 使用的模型版本。建议用 2.5-pro 进行复杂场景和空间坐标解析。
        """
        if not HAS_GENAI:
            raise ImportError(
                "缺少 google-generativeai 依赖库。请执行: \n"
                "pip install google-generativeai pillow"
            )

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if key:
            genai.configure(api_key=key)
            log.info("VLM 建图引擎使用 API Key 授权")
        else:
            try:
                # 尝试使用 OAuth (Application Default Credentials)
                import google.auth
                credentials, project_id = google.auth.default()
                genai.configure(credentials=credentials)
                log.info(f"VLM 建图引擎使用 OAuth 授权 (Project: {project_id})")
            except Exception as e:
                raise ValueError(
                    "⚠️ 未检测到 GEMINI_API_KEY，且尝试加载 OAuth 凭证失败！\n"
                    "请执行 `gcloud auth application-default login` 或参考:\n"
                    "https://docs.gemini.com/authentication/oauth\n"
                    f"详细错误: {e}"
                )
        
        # 强制要求 JSON 输出格式
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"response_mime_type": "application/json"}
        )
        log.info(f"VLM建图引擎已初始化，挂载模型: {model_name}")
        
        self.prompt = """
        你是一个专门为《冒险岛 (MapleStory)》开发2D智能体的顶尖计算机视觉专家。
        这是一张该游戏 1600x900 分辨率的游戏实机截图。

        请帮我提取出建立“寻路网格(NavMesh)”和“怪物图鉴”所需要的精确坐标信息。
        请以 JSON 格式返回，严格遵循以下结构：

        {
            "platforms": [
                {"description": "描述(如：最底层的平地)", "x1": 左边界像素, "y1": 上下边界像素(纵向居中), "x2": 右边界像素, "y2": 与y1一致},
                // 尽可能把所有明显可以站立的横向台阶/地面提取出来。
            ],
            "ropes_or_ladders": [
                {"description": "描述(如：中间的棕色绳子)", "x": 垂直绳子的中心横坐标, "y1": 顶端纵坐标, "y2": 底端纵坐标}
            ],
            "monsters": [
                {"type": "怪物名称/类别(如绿水灵、红蜗牛)", "x_center": 中心的横坐标, "y_center": 中心的纵坐标, "width": 像素宽度, "height": 像素高度}
            ]
        }

        注意：
        1. 坐标原点(0,0)在图片左上角。
        2. x 的最大值约 1600，y 的最大值约 900。
        3. 请极尽全力做到精确，这直接关系到 AI 的生死！
        4. 你的输出必须是合法的纯 JSON 字符串。
        """

    def analyze_map(self, image_path: str) -> Dict[str, Any]:
        """
        发送截图，获取 JSON 格式的解析结果。
        
        Args:
            image_path: 游戏截图的本地绝对/相对路径。
            
        Returns:
            Dict: 包含 platforms, ropes, monsters 的字典。
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"找不到截图文件: {image_path}")

        log.info(f"正在上传画面 [{image_path}] 呼叫大模型进行高维解析...")
        try:
            img = Image.open(image_path)
            response = self.model.generate_content([self.prompt, img])
            
            # 解析返回的 JSON
            json_text = response.text
            result = json.loads(json_text)
            
            log.info("✅ VLM 降维建图完成！")
            log.info(f"检测到 {len(result.get('platforms', []))} 个平台, "
                     f"{len(result.get('ropes_or_ladders', []))} 根绳子, "
                     f"{len(result.get('monsters', []))} 只怪物。")
                     
            return result
            
        except json.JSONDecodeError as e:
            log.error(f"解析 JSON 错误，大模型返回格式异常: {response.text}")
            raise e
        except Exception as e:
            log.error(f"大模型请求失败: {e}")
            raise e


if __name__ == "__main__":
    # 调试与测试入口
    import sys
    
    # 假设你有一张名为 full_map.png 的游戏截图放在这里
    test_img = "data/debug/full_map.png" 
    
    if not os.path.exists(test_img):
        print(f"请放置一张截图于 {test_img} 供测试。")
        sys.exit(1)

    try:
        # 实例化时如果不传 key，就会自动走上面的 OAuth 流程
        mapper = VLMMapper(model_name="gemini-2.5-pro")
        result = mapper.analyze_map(test_img)
        print("\n==== 解析结果 JSON ====\n")
        print(json.dumps(result, indent=4, ensure_ascii=False))
    except Exception as e:
        print(f"测试失败: {e}")
