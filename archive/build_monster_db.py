"""
Victoria Island Monster Sprite Database Builder
================================================
从 maplestory.io 的公开 API 批量下载所有 Victoria Island 怪物的精灵图
用于后续 YOLO 训练的图鉴数据库。

API:
- 怪物列表: https://maplestory.io/api/GMS/64/mob
- 精灵GIF:  https://maplestory.io/api/GMS/64/mob/{id}/render/stand
- 精灵PNG:  https://maplestory.io/api/GMS/64/mob/{id}/icon
"""
import os
import json
import time
import requests

DB_DIR = "data/monster_db"
os.makedirs(DB_DIR, exist_ok=True)

# Victoria Island 怪物ID范围 (GMS v64)
# 参考 https://www.artalemaplestory.com/en/monsters
VICTORIA_MOB_IDS = {
    # === Maple Island ===
    100100: "Snail",
    100101: "Blue Snail",
    # === Victoria Low Level ===
    1110100: "Green Mushroom",
    1110101: "Dark Stump",
    1120100: "Octopus",
    1130100: "Pig",
    1210100: "Stump",
    1210102: "Axe Stump",
    # === General list (we also fetch ALL from API if level <= 60) ===
}

def main():
    print("=== MapleStory Victoria Island Monster DB Builder ===")
    print("Step 1: Fetching full monster list from maplestory.io API...")
    
    resp = requests.get("https://maplestory.io/api/GMS/64/mob", timeout=30)
    if resp.status_code != 200:
        print(f"API Error: {resp.status_code}")
        return
        
    all_mobs = resp.json()
    print(f"API returned {len(all_mobs)} monsters total.")
    
    # 筛选 Victoria Island 范围的怪物 (Level 1-60, non-boss)
    victoria_mobs = [m for m in all_mobs if m.get('level', 0) <= 60 and not m.get('isBoss', False)]
    print(f"Filtered to {len(victoria_mobs)} Victoria-level monsters (Lv 1~60, non-boss).")
    
    print(f"\nStep 2: Downloading sprite images...")
    
    db_index = {}
    success = 0
    fail = 0
    
    for i, mob in enumerate(victoria_mobs):
        mob_id = mob['id']
        mob_name = mob.get('name', f'Unknown_{mob_id}')
        mob_level = mob.get('level', 0)
        
        safe_name = mob_name.replace(" ", "_").replace("/", "_").replace("'", "")
        
        print(f"  [{i+1}/{len(victoria_mobs)}] Lv.{mob_level} {mob_name} (ID:{mob_id})...", end=" ")
        
        # 下载 stand 动画GIF 和 icon PNG
        gif_path = os.path.join(DB_DIR, f"{safe_name}_{mob_id}.gif")
        png_path = os.path.join(DB_DIR, f"{safe_name}_{mob_id}.png")
        
        downloaded = False
        
        # 尝试下载 GIF (stand animation)
        try:
            url_gif = f"https://maplestory.io/api/GMS/64/mob/{mob_id}/render/stand"
            r = requests.get(url_gif, timeout=10)
            if r.status_code == 200 and len(r.content) > 100:
                with open(gif_path, "wb") as f:
                    f.write(r.content)
                downloaded = True
        except:
            pass
            
        # 尝试下载 icon PNG
        try:
            url_png = f"https://maplestory.io/api/GMS/64/mob/{mob_id}/icon"
            r = requests.get(url_png, timeout=10)
            if r.status_code == 200 and len(r.content) > 100:
                with open(png_path, "wb") as f:
                    f.write(r.content)
                downloaded = True
        except:
            pass
        
        if downloaded:
            db_index[str(mob_id)] = {
                "name": mob_name,
                "level": mob_level,
                "id": mob_id,
                "gif": gif_path if os.path.exists(gif_path) else None,
                "png": png_path if os.path.exists(png_path) else None,
            }
            success += 1
            print("OK")
        else:
            fail += 1
            print("FAIL")
            
        time.sleep(0.15)  # polite rate limit
    
    # Save index
    index_path = os.path.join(DB_DIR, "monster_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(db_index, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== DONE ===")
    print(f"Success: {success}, Failed: {fail}")
    print(f"Index: {index_path}")
    print(f"Images: {os.path.abspath(DB_DIR)}")

if __name__ == "__main__":
    main()
