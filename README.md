# RAG-test: OpenAI + Milvus + RAG 系统

这是一个使用OpenAI模型和Milvus向量数据库实现的RAG（检索增强生成）系统。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置

1. 根据 `example_config.json` 编辑 `config.json` 文件，设置API密钥、模型和其他参数。
2. 将PDF文档放在可访问的位置，并在配置文件中更新路径。

## 设置本地Milvus服务器

### 方法1: 使用Docker

```bash
# 下载docker-compose.yml
wget https://github.com/milvus-io/milvus/releases/download/v2.3.1/milvus-standalone-docker-compose.yml -O docker-compose.yml

# 启动Milvus服务
docker-compose up -d
```

### 方法2: 使用Milvus Lite模式

如果不想运行独立的Milvus服务器，可以使用Milvus Lite模式，将配置文件中的`uri`改为文件路径：

```json
"milvus": {
    "uri": "./hf_milvus_demo.db",
    ...
}
```

## 初始化系统

在首次使用前，您需要初始化Milvus数据库并加载PDF数据。有几种方法可以实现这一点：

### 方法1: 使用初始化脚本

```bash
# 基本初始化（使用配置文件中的设置）
python init_milvus.py

# 指定PDF文件初始化
python init_milvus.py --pdf path/to/your/document.pdf

# 自动选择第一个找到的PDF文件
python init_milvus.py --auto

# 在特定目录中查找PDF文件
python init_milvus.py --dir path/to/documents/folder

# 强制重建已存在的集合
python init_milvus.py --force-rebuild
```

### 方法2: 通过main.py初始化

```bash
# 使用配置文件中指定的PDF文件加载数据（不会重建已存在的集合）
python main.py --use-rag

# 强制重建集合并加载数据
python main.py --use-rag --force-rebuild

# 指定PDF文件加载
python main.py --use-rag --load-pdf path/to/your/document.pdf
```

## 运行

### 基本用法

```bash
# 使用配置文件中的设置
python main.py

# 手动启用RAG并加载PDF（不重建已存在的集合）
python main.py --use-rag --load-pdf path/to/your/doc.pdf

# 强制重建集合并加载PDF
python main.py --use-rag --load-pdf path/to/your/doc.pdf --force-rebuild

# 连接到特定的Milvus服务器
python main.py --milvus-uri "http://localhost:19530"
```

### 常见问题

- 确保Milvus服务器正在运行（如果使用独立服务器）
- 检查配置文件中的路径是否正确
- 确保有足够的磁盘空间用于存储向量

## 相关链接

- [Milvus 官方文档](https://milvus.io/docs)
