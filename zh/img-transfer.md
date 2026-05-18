## 图片上传到 COS 并替换 Markdown 引用

### 概述

项目中图片分布在多个位置：
- `assets/` — 主共享图片目录（147 个文件）
- `en/examples/images/` 和 `zh/examples/images/` — 示例共置图片
- `zh/course/.../images/` — 课程共置图片
- `en/` 和 `zh/` 下各 example 子目录中的图片子文件夹（如 `en/examples/cats_dogs/`、`zh/examples/mlx_lm_finetune/` 等）

两类引用模式：
- 绝对路径：`![](/assets/badge1.svg)` — 引用 `assets/`
- 相对路径：`![](./images/glm4-instruct/instruct.png)` — 与 markdown 共置

不上传：`public/`（网站 UI 资源：favicon、导航图标）

### COS Key 命名策略

```
assets/badge1.svg                        → assets/badge1.svg
zh/course/.../instruct.png               → assets/zh/course/.../instruct.png
en/examples/cats_dogs/01.png             → assets/en/examples/cats_dogs/01.png
zh/examples/mlx_lm_finetune/apple.png    → assets/zh/examples/mlx_lm_finetune/apple.png
```

COS 完整 URL：`https://<BUCKET>.cos.<REGION>.myqcloud.com/assets/...`

### Step 1：安装与配置 coscmd

```bash
pip install coscmd

coscmd config -a <SecretId> -s <SecretKey> -b <Bucket> -r <Region>
# 例如: coscmd config -a AKIDxxx -s xxxxxx -b swanlab-docs-1250000000 -r ap-beijing
```

### Step 2：上传图片（Python SDK 脚本，仅上传图片文件）

> coscmd upload -r 会上传所有文件，用 Python SDK 做过滤更精确。

依赖：`pip install cos-python-sdk-v5`

创建 `scripts/upload-to-cos.py`：
```python
"""批量上传项目中的图片到腾讯云 COS（仅上传图片文件，排除 public/）"""
import os, glob
from qcloud_cos import CosConfig, CosS3Client

SECRET_ID = os.environ['COS_SECRET_ID']
SECRET_KEY = os.environ['COS_SECRET_KEY']
BUCKET = os.environ['COS_BUCKET']
REGION = os.environ['COS_REGION']

# 脚本位于 scripts/ 目录下
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

config = CosConfig(Region=REGION, SecretId=SECRET_ID, SecretKey=SECRET_KEY)
client = CosS3Client(config)

CONTENT_TYPES = {
    '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
    '.gif': 'image/gif', '.svg': 'image/svg+xml', '.webp': 'image/webp',
    '.mp4': 'video/mp4',
}

SKIP_DIRS = {'public', '.git', 'node_modules', '.vitepress'}

uploaded = 0
for ext in CONTENT_TYPES:
    for local_path in glob.glob(os.path.join(PROJECT_ROOT, '**', f'*{ext}'), recursive=True):
        rel_to_root = os.path.relpath(local_path, PROJECT_ROOT)
        # 跳过 public/、.git/ 等非内容目录
        top_dir = rel_to_root.split(os.sep)[0]
        if top_dir in SKIP_DIRS:
            continue

        cos_key = f'assets/{rel_to_root}'
        content_type = CONTENT_TYPES[ext]
        client.upload_file(Bucket=BUCKET, Key=cos_key, LocalFilePath=local_path, ContentType=content_type)
        uploaded += 1
        print(f'✓ {rel_to_root} → {cos_key}')

print(f'\n上传完成，共 {uploaded} 个文件')
```

运行：
```bash
COS_SECRET_ID=xxx COS_SECRET_KEY=xxx COS_BUCKET=xxx COS_REGION=xxx python scripts/upload-to-cos.py
```

### Step 3：替换 Markdown 中的图片引用

上传完成后，将 `en/` 和 `zh/` 下 markdown 文件中的图片路径替换为 COS URL。

创建 `scripts/replace-img-urls.py`：
```python
"""将 en/ 和 zh/ 下 markdown 文件中的图片引用替换为 COS URL"""
import os, re

COS_BASE = os.environ['COS_BASE_URL']  # 例如: https://swanlab-docs-1250000000.cos.ap-beijing.myqcloud.com

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

# 需要扫描的目录
SCAN_DIRS = [
    os.path.join(PROJECT_ROOT, 'zh'),
    os.path.join(PROJECT_ROOT, 'en'),
]

def resolve_to_cos_url(img_path, md_file_dir):
    """将图片路径转换为 COS URL"""
    if img_path.startswith('/assets/'):
        # 绝对路径: /assets/xxx.png → COS_BASE/assets/xxx.png
        return f'{COS_BASE}{img_path}'
    elif img_path.startswith('./') or img_path.startswith('images/') or not img_path.startswith('http'):
        # 相对路径: 解析为相对于项目根目录的路径
        abs_path = os.path.normpath(os.path.join(md_file_dir, img_path))
        if os.path.isfile(abs_path):
            rel_to_root = os.path.relpath(abs_path, PROJECT_ROOT)
            return f'{COS_BASE}/assets/{rel_to_root}'
    return None  # 外部 URL 或无法解析，跳过

def process_md_file(filepath):
    """处理单个 markdown 文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    md_dir = os.path.dirname(filepath)
    changed = False

    def replace_match(m):
        nonlocal changed
        alt = m.group(1)
        img_path = m.group(2)
        # 跳过已经是 COS URL 或外部链接的
        if img_path.startswith('http'):
            return m.group(0)
        new_url = resolve_to_cos_url(img_path, md_dir)
        if new_url:
            changed = True
            return f'![{alt}]({new_url})'
        return m.group(0)

    # 匹配 ![alt](path) 格式
    new_content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_match, content)

    # 匹配 <img src="path"> 格式
    def replace_img_tag(m):
        nonlocal changed
        prefix = m.group(1)
        img_path = m.group(2)
        suffix = m.group(3)
        if img_path.startswith('http'):
            return m.group(0)
        new_url = resolve_to_cos_url(img_path, md_dir)
        if new_url:
            changed = True
            return f'{prefix}{new_url}{suffix}'
        return m.group(0)

    new_content = re.sub(r'(<img\s+[^>]*src=["\'])([^"\']+)(["\'][^>]*>)', replace_img_tag, new_content)

    if changed:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f'✓ {os.path.relpath(filepath, PROJECT_ROOT)}')

count = 0
for scan_dir in SCAN_DIRS:
    for root, dirs, files in os.walk(scan_dir):
        for f in files:
            if f.endswith(('.md', '.mdx')):
                filepath = os.path.join(root, f)
                process_md_file(filepath)
                count += 1

print(f'\n扫描完成，共处理 {count} 个文件')
```

运行：
```bash
COS_BASE_URL=https://swanlab-docs-1250000000.cos.ap-beijing.myqcloud.com python scripts/replace-img-urls.py
```

### Step 4：验证

- 检查替换后的 markdown 文件，确认图片 URL 正确
- 随机打开几个文档页面，确认图片加载正常
- 确认 `public/` 目录下的 UI 资源未被替换
