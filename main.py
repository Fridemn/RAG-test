from llm_client import LLMClient
import argparse
import os
import json

def load_config(config_path='config.json'):
    """加载配置文件"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"警告: 无法加载配置文件 {config_path}: {str(e)}")
        return {}

def main():
    # 加载配置
    config = load_config()
    rag_config = config.get('rag', {})
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="RAG增强的聊天系统")
    parser.add_argument("--config", type=str, default='config.json', help="配置文件路径")
    parser.add_argument("--use-rag", action="store_true", help="启用RAG系统")
    parser.add_argument("--no-rag", action="store_true", help="禁用RAG系统")
    parser.add_argument("--load-pdf", type=str, help="加载PDF到知识库")
    parser.add_argument("--model", type=str, help="使用的OpenAI模型")
    parser.add_argument("--api-key", type=str, help="OpenAI API密钥")
    parser.add_argument("--milvus-uri", type=str, help="Milvus服务器URI")
    parser.add_argument("--force-rebuild", action="store_true", help="强制重建集合，即使已存在")
    args = parser.parse_args()
    
    # 如果指定了不同的配置文件，重新加载
    if args.config != 'config.json':
        config = load_config(args.config)
        rag_config = config.get('rag', {})
    
    # 如果指定了Milvus URI，更新配置
    if args.milvus_uri:
        if 'milvus' not in rag_config:
            rag_config['milvus'] = {}
        rag_config['milvus']['uri'] = args.milvus_uri
    
    # 确定是否使用RAG
    use_rag = None
    if args.use_rag:
        use_rag = True
    elif args.no_rag:
        use_rag = False
    
    try:
        # 初始化LLM客户端，优先使用命令行参数
        client = LLMClient(
            config_path=args.config,
            use_rag=use_rag, 
            model=args.model, 
            api_key=args.api_key
        )
        
        # 如果需要加载PDF（从命令行或配置文件）
        pdf_path = args.load_pdf or rag_config.get('documents', {}).get('pdf_path')
        if pdf_path and client.use_rag:
            client.rag_system.load_data(pdf_path, force_rebuild=args.force_rebuild)
            print(f"已处理PDF知识库: {pdf_path}")
        
        print("\n" + "=" * 50)
        print("RAG增强的对话系统")
        print("=" * 50)
        print("输入 'quit' 或 'exit' 退出对话")
        print("-" * 50)
        
        if client.use_rag:
            milvus_uri = rag_config.get('milvus', {}).get('uri', 'default')
            collection_name = rag_config.get('milvus', {}).get('collection_name', 'rag_collection')
            print("RAG系统已启用 - 回答将基于知识库")
            print(f"Milvus服务器: {milvus_uri}")
            print(f"集合: {collection_name}")
            
            # 检查集合是否存在且有数据
            if client.rag_system.milvus_client.has_collection(collection_name):
                stats = client.rag_system.milvus_client.get_collection_stats(collection_name)
                row_count = stats.get('row_count', 0)
                print(f"集合中的数据量: {row_count} 条")
        else:
            print("RAG系统未启用 - 使用普通LLM回答")
        print(f"使用模型: {client.model}")
        print(f"API基础URL: {client.base_url or '默认'}")
        print("-" * 50)
    
        while True:
            prompt = input("\n用户: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
                
            if not prompt:
                continue
                
            try:
                response = client.call_llm(prompt)
                print("\nAI: " + response)
                print("-" * 50)
            except Exception as e:
                print(f"\n错误: {str(e)}")
                print("-" * 50)
    except Exception as e:
        print(f"初始化错误: {str(e)}")
        print("\nMilvus连接问题排查:")
        print("1. 确保Milvus服务已经启动")
        print("2. 检查配置文件中的URI是否正确")
        print("3. 确保网络连接正常")
        print("4. 如有必要，检查Milvus日志以获取更多信息")

if __name__ == "__main__":
    main()
