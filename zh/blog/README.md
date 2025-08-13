# SwanLab 博客

这个目录包含了 SwanLab 的博客文章和相关资源。

## 目录结构

```
zh/blog/
├── index.md          # 博客首页
├── blog-data.ts      # 博客数据文件
├── README.md         # 本说明文档
└── assets/           # 博客相关资源
    └── images/       # 博客图片
```

## 添加新的博客文章

### 1. 更新博客数据

在 `blog-data.ts` 文件中添加新的博客文章信息：

```typescript
{
  title: '文章标题',
  description: '文章描述',
  date: '2024-12-20',
  author: '作者姓名',
  category: 'release', // 可选：release, product, how-to, developer, event, company, community
  tags: ['标签1', '标签2'],
  image: '/assets/blog/your-image.jpg', // 可选
  link: '/blog/your-article-link'
}
```

### 2. 添加博客图片

将博客图片放在 `assets/blog/` 目录下，建议尺寸为 800x400 像素。

### 3. 创建博客文章页面

在 `zh/blog/` 目录下创建对应的 Markdown 文件，例如 `your-article-link.md`。

## 分类说明

- **release**: 版本发布
- **product**: 产品功能
- **how-to**: 使用教程
- **developer**: 开发者相关
- **event**: 活动公告
- **company**: 公司动态
- **community**: 社区相关

## 样式指南

- 标题：简洁明了，突出核心内容
- 描述：100-150 字，概括文章要点
- 标签：2-4 个，便于分类和搜索
- 图片：高质量，与内容相关
- 日期：使用 YYYY-MM-DD 格式

## 注意事项

1. 确保所有链接都是有效的
2. 图片文件大小控制在合理范围内
3. 保持分类和标签的一致性
4. 定期更新博客内容 