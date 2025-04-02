import os
import json
import argparse
from pymilvus import MilvusClient
from rag_system import RAGSystem
import sys
import time
from pathlib import Path

def load_config(config_path='config.json'):
    """加载配置文件"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"警告: 无法加载配置文件 {config_path}: {str(e)}")
        return {}

def check_milvus_connection(uri, user='', password='', max_retries=5, retry_interval=2):
    """检查是否可以连接到Milvus服务器"""
    print(f"正在尝试连接到Milvus服务器: {uri}")
    
    milvus_params = {'uri': uri}
    if user:
        milvus_params['user'] = user
        milvus_params['password'] = password
    
    for i in range(max_retries):
        try:
            client = MilvusClient(**milvus_params)
            # 执行一个简单的操作来验证连接
            client.list_collections()
            print("Milvus连接成功!")
            return client
        except Exception as e:
            print(f"尝试 {i+1}/{max_retries} 连接失败: {str(e)}")
            if i < max_retries - 1:
                print(f"等待 {retry_interval} 秒后重试...")
                time.sleep(retry_interval)
    
    print("无法连接到Milvus服务器，请检查服务是否运行以及URI是否正确")
    return None

def find_pdf_files(directory='.', recursive=True):
    """在指定目录中查找所有PDF文件"""
    if recursive:
        return list(Path(directory).glob('**/*.pdf'))
    else:
        return list(Path(directory).glob('*.pdf'))

def list_pdf_options(pdf_files):
    """列出所有可用的PDF文件选项"""
    if not pdf_files:
        return None
    
    print("\n可用的PDF文件:")
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"{i}. {pdf_file}")
    
    while True:
        try:
            choice = input("\n请选择一个PDF文件 (输入数字或完整路径, 'q' 退出): ")
            if choice.lower() in ['q', 'quit', 'exit']:
                return None
            
            # 尝试将输入解析为索引
            try:
                index = int(choice) - 1
                if 0 <= index < len(pdf_files):
                    return str(pdf_files[index])
                else:
                    print("无效的选择，请重试")
            except ValueError:
                # 如果不是数字，检查是否是有效的文件路径
                if os.path.exists(choice) and choice.lower().endswith('.pdf'):
                    return choice
                else:
                    print("找不到PDF文件，请重试")
        except KeyboardInterrupt:
            print("\n操作已取消")
            return None

def initialize_system(config_path='config.json', pdf_path=None, auto_select=False, search_dir='.', force_rebuild=False):
    """初始化RAG系统"""
    # 加载配置文件
    config = load_config(config_path)
    rag_config = config.get('rag', {})
    milvus_config = rag_config.get('milvus', {})
    
    # 获取Milvus连接参数
    milvus_uri = milvus_config.get('uri', 'http://localhost:19530')
    milvus_user = milvus_config.get('user', '')
    milvus_password = milvus_config.get('password', '')
    collection_name = milvus_config.get('collection_name', 'rag_collection')
    
    # 测试Milvus连接
    milvus_client = check_milvus_connection(milvus_uri, milvus_user, milvus_password)
    if not milvus_client:
        return False
    
    # 检查集合是否存在
    collection_exists = milvus_client.has_collection(collection_name)
    if collection_exists and not force_rebuild:
        print(f"\n集合 '{collection_name}' 已存在")
        stats = milvus_client.get_collection_stats(collection_name)
        row_count = stats.get('row_count', 0)
        print(f"集合中的数据量: {row_count} 条")
        
        if row_count > 0:
            choice = input("是否要重新创建并加载数据? (y/n): ").lower()
            if choice != 'y':
                print("保留现有集合，初始化完成")
                return True
            else:
                force_rebuild = True
        else:
            print("集合存在但没有数据，将加载数据")
    
    # 确定要加载的PDF文件
    if not pdf_path:
        pdf_path = rag_config.get('documents', {}).get('pdf_path')
    
    # 如果仍然没有PDF路径，搜索并提供选择
    if not pdf_path or not os.path.exists(pdf_path):
        print("\n未找到有效的PDF文件路径")
        pdf_files = find_pdf_files(search_dir)
        
        if not pdf_files:
            print(f"在 {search_dir} 目录中未找到PDF文件")
            return False
        
        if auto_select and pdf_files:
            pdf_path = str(pdf_files[0])
            print(f"自动选择第一个找到的PDF文件: {pdf_path}")
        else:
            pdf_path = list_pdf_options(pdf_files)
        
        if not pdf_path:
            print("未选择PDF文件，初始化取消")
            return False
    
    # 初始化RAG系统
    try:
        print(f"使用PDF文件初始化RAG系统: {pdf_path}")
        rag_system = RAGSystem(config_path=config_path)
        rag_system.load_data(pdf_path, force_rebuild=force_rebuild)
        print("\n初始化完成！系统已准备就绪")
        return True
    except Exception as e:
        print(f"初始化RAG系统时出错: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Milvus RAG系统初始化工具")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")
    parser.add_argument("--pdf", type=str, help="要加载的PDF文件路径")
    parser.add_argument("--auto", action="store_true", help="自动选择第一个找到的PDF文件")
    parser.add_argument("--dir", type=str, default=".", help="搜索PDF文件的目录")
    parser.add_argument("--force-rebuild", action="store_true", help="强制重建集合，即使已存在")
    args = parser.parse_args()
    
    success = initialize_system(
        config_path=args.config,
        pdf_path=args.pdf,
        auto_select=args.auto,
        search_dir=args.dir,
        force_rebuild=args.force_rebuild
    )
    
    if not success:
        print("\n初始化失败，请检查以上错误信息")
        sys.exit(1)

if __name__ == "__main__":
    main()
