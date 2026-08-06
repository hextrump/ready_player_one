"""
从 meowdb.com 下载 minimap 库 + 抓怪物参考数据, 存到 data/map_db/

用法:
  python scripts/build_map_db.py --download-minimaps   # 下载所有 minimap
  python scripts/build_map_db.py --scrape-monsters     # 抓怪物列表
  python scripts/build_map_db.py --all                 # 全跑

数据格式:
  data/map_db/
  ├── minimaps/010001010.png ...         # minimap 图像 (334x116)
  ├── index.json                          # {map_id: {name, url, has_minimap, monsters_count}}
  └── monsters.json                       # {map_id: [{name, level, count}, ...]}
"""
import argparse
import json
import re
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAP_DB_DIR = PROJECT_ROOT / "data" / "map_db"
MINIMAP_DIR = MAP_DB_DIR / "minimaps"
INDEX_FILE = MAP_DB_DIR / "index.json"
MONSTERS_FILE = MAP_DB_DIR / "monsters.json"

BASE_URL = "https://meowdb.com/msclassic/maps"
MINIMAP_URL = "https://meowdb.com/msclassic/maps/minimaps/{map_id}.png"
UA = "Mozilla/5.0 (compatible; MapDBBot/1.0)"


def http_get(url, timeout=15):
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req, timeout=timeout) as r:
        return r.read()


def http_get_text(url, timeout=15):
    return http_get(url, timeout).decode("utf-8", errors="ignore")


def discover_map_ids():
    """从 /maps/all  页面提取所有地图 ID"""
    print(f"  GET {BASE_URL}/all")
    html = http_get_text(f"{BASE_URL}/all")

    # 链接形如 /msclassic/maps/010001010  (9 位数字 ID)
    pattern = re.compile(r'/msclassic/maps/(\d{9})(?![\d/])')
    ids = sorted(set(pattern.findall(html)))
    print(f"  提取到 {len(ids)} 个地图 ID")
    return ids


def download_minimap(map_id: str, retries=2):
    """下载单张 minimap, 成功返回 True"""
    path = MINIMAP_DIR / f"{map_id}.png"
    if path.exists():
        return True
    url = MINIMAP_URL.format(map_id=map_id)
    try:
        data = http_get(url)
        path.write_bytes(data)
        return True
    except HTTPError as e:
        if e.code == 404:
            return False
        if retries > 0:
            time.sleep(0.5)
            return download_minimap(map_id, retries - 1)
        return False
    except Exception:
        return False


def download_all_minimaps(ids):
    MINIMAP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"下载 {len(ids)} 张 minimap 到 {MINIMAP_DIR}/")
    success = []
    miss = []
    for j, mid in enumerate(ids, 1):
        ok = download_minimap(mid)
        if ok:
            success.append(mid)
        else:
            miss.append(mid)
        if j % 50 == 0:
            print(f"  [{j}/{len(ids)}] 已下载 {len(success)}, 失败 {len(miss)}")
    print(f"  完成: 成功 {len(success)}, 失败 {len(miss)}")
    if miss[:5]:
        print(f"  失败样例: {miss[:5]}")
    return success


def scrape_map_metadata(map_id):
    """抓单张地图详情页, 提取地图名 + 怪物列表"""
    url = f"{BASE_URL}/{map_id}"
    try:
        html = http_get_text(url)
    except Exception as e:
        return None

    # 地图名: <h1>... 标题</h1>
    name_match = re.search(r'<h1[^>]*>(.*?)</h1>', html, re.DOTALL)
    name = name_match.group(1).strip() if name_match else map_id
    name = re.sub(r'<[^>]+>', '', name).strip()

    # 怪物列表: 在 <table> 里, 列名包含 "Monster" / "Name"
    monsters = []
    # 找怪物段落: 通常以 "Monsters" 标题开头, 跟一个表格
    table_match = re.search(r'Monsters.*?<table.*?</table>', html, re.DOTALL | re.IGNORECASE)
    if table_match:
        table_html = table_match.group(0)
        # 提取行
        rows = re.findall(r'<tr.*?</tr>', table_html, re.DOTALL)
        for row in rows[1:]:  # skip header
            cells = re.findall(r'<td.*?</td>', row, re.DOTALL)
            cells_text = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
            if cells_text and cells_text[0]:
                monsters.append({
                    "name": cells_text[0],
                    "level": cells_text[1] if len(cells_text) > 1 else "",
                    "count": cells_text[2] if len(cells_text) > 2 else "",
                })

    return {"id": map_id, "name": name, "monsters": monsters}


def scrape_all_monsters(ids):
    print(f"抓 {len(ids)} 张地图的怪物数据")
    out = {}
    for j, mid in enumerate(ids, 1):
        meta = scrape_map_metadata(mid)
        if meta and meta["monsters"]:
            out[mid] = meta
        if j % 30 == 0:
            print(f"  [{j}/{len(ids)}] 已抓 {len(out)} 张含怪物数据")
        time.sleep(0.1)  # 礼貌延迟
    print(f"  完成: {len(out)} 张地图有怪物数据")
    return out


def build_index(ids, success_ids):
    """生成 index.json"""
    return {
        "total_discovered": len(ids),
        "minimap_downloaded": len(success_ids),
        "map_ids": ids,
    }


def main():
    p = argparse.ArgumentParser(description="建 meowdb 地图数据库")
    p.add_argument("--download-minimaps", action="store_true")
    p.add_argument("--scrape-monsters", action="store_true")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    if not (args.download_minimaps or args.scrape_monsters or args.all):
        p.print_help()
        return

    MAP_DB_DIR.mkdir(parents=True, exist_ok=True)

    ids = discover_map_ids()

    if args.download_minimaps or args.all:
        success_ids = download_all_minimaps(ids)
        idx = build_index(ids, success_ids)
        INDEX_FILE.write_text(json.dumps(idx, indent=2), encoding="utf-8")
        print(f"  索引写入 {INDEX_FILE}")

    if args.scrape_monsters or args.all:
        monsters_data = scrape_all_monsters(ids)
        MONSTERS_FILE.write_text(
            json.dumps(monsters_data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"  怪物数据写入 {MONSTERS_FILE} ({len(monsters_data)} 张地图)")


if __name__ == "__main__":
    main()