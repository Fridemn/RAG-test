import os
import json
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from openai import OpenAI

class RAGSystem:
    def __init__(self, config_path='config.json', api_key=None, base_url=None, organization=None):
        """
        初始化RAG系统，从配置文件加载所有设置
        
        Args:
            config_path: 配置文件路径
            api_key: 可选的OpenAI API密钥（覆盖配置文件中的值）
            base_url: 可选的OpenAI API基础URL（覆盖配置文件中的值）
            organization: 可选的OpenAI组织ID（覆盖配置文件中的值）
        """
        # 加载配置文件
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"无法加载配置文件: {str(e)}")
            self.config = {}

        # 设置API参数（命令行参数优先）
        self.api_key = api_key or self.config.get('api_key')
        self.base_url = base_url or self.config.get('base_url')
        self.organization = organization or self.config.get('organization')

        # 获取RAG配置
        rag_config = self.config.get('rag', {})
        milvus_config = rag_config.get('milvus', {})
        embedding_config = rag_config.get('embedding', {})
        retrieval_config = rag_config.get('retrieval', {})
        
        # 设置Milvus参数
        self.collection_name = milvus_config.get('collection_name', 'rag_collection')
        self.milvus_uri = milvus_config.get('uri', 'http://localhost:19530')
        self.milvus_user = milvus_config.get('user', '')
        self.milvus_password = milvus_config.get('password', '')
        self.metric_type = milvus_config.get('metric_type', 'IP')
        self.consistency_level = milvus_config.get('consistency_level', 'Strong')
        
        # 设置嵌入参数
        self.use_openai_embeddings = embedding_config.get('use_openai', False)
        self.openai_model = embedding_config.get('openai_model', 'text-embedding-ada-002')
        self.local_embedding_model = embedding_config.get('local_model', 'BAAI/bge-small-en-v1.5')
        
        # 设置检索参数
        self.top_k = retrieval_config.get('top_k', 3)
        
        # 初始化嵌入模型
        if self.use_openai_embeddings:
            print("使用OpenAI嵌入模型...")
            client_params = {'api_key': self.api_key}
            if self.base_url:
                client_params['base_url'] = self.base_url
            if self.organization:
                client_params['organization'] = self.organization
                
            self.openai_client = OpenAI(**client_params)
        else:
            print(f"加载本地嵌入模型: {self.local_embedding_model}...")
            self.embedding_model = SentenceTransformer(self.local_embedding_model)
        
        # 连接到Milvus
        print(f"连接到Milvus服务器: {self.milvus_uri}")
        milvus_params = {'uri': self.milvus_uri}
        if self.milvus_user:
            milvus_params['user'] = self.milvus_user
            milvus_params['password'] = self.milvus_password
            
        try:
            self.milvus_client = MilvusClient(**milvus_params)
            print("Milvus连接成功")
        except Exception as e:
            print(f"连接到Milvus时出错: {str(e)}")
            print("请确保Milvus服务器正在运行且可访问")
            raise
        
        # 检查集合是否存在
        try:
            if not self.milvus_client.has_collection(self.collection_name):
                print(f"集合 {self.collection_name} 不存在，请先加载数据")
        except Exception as e:
            print(f"检查集合时出错: {str(e)}")
    
    def emb_text(self, text):
        """生成文本的嵌入向量"""
        if self.use_openai_embeddings:
            # 使用OpenAI生成嵌入
            response = self.openai_client.embeddings.create(
                model=self.openai_model,
                input=text
            )
            return response.data[0].embedding
        else:
            # 使用本地模型生成嵌入
            return self.embedding_model.encode([text], normalize_embeddings=True).tolist()[0]
    
    def load_data(self, pdf_path=None, chunk_size=None, chunk_overlap=None, force_rebuild=False):
        """
        从PDF加载数据到Milvus
        
        Args:
            pdf_path: PDF文件路径（None表示使用配置文件中的路径）
            chunk_size: 分块大小（None表示使用配置文件中的值）
            chunk_overlap: 分块重叠大小（None表示使用配置文件中的值）
            force_rebuild: 是否强制重建集合，即使已存在
        """
        # 从配置文件加载文档设置
        doc_config = self.config.get('rag', {}).get('documents', {})
        pdf_path = pdf_path or doc_config.get('pdf_path')
        chunk_size = chunk_size or doc_config.get('chunk_size', 1000)
        chunk_overlap = chunk_overlap or doc_config.get('chunk_overlap', 200)
        
        if not pdf_path:
            print("错误: 未指定PDF文件路径")
            return
            
        if not os.path.exists(pdf_path):
            print(f"错误: PDF文件 '{pdf_path}' 不存在")
            return
        
        # 检查集合是否已存在
        collection_exists = self.milvus_client.has_collection(self.collection_name)
        
        # 如果集合已存在且不强制重建，直接返回
        if collection_exists and not force_rebuild:
            print(f"集合 {self.collection_name} 已存在，跳过创建和数据加载")
            # 获取集合统计信息
            stats = self.milvus_client.get_collection_stats(self.collection_name)
            row_count = stats.get('row_count', 0)
            print(f"集合中的数据量: {row_count} 条")
            return
        
        print(f"加载PDF文件: {pdf_path}")
        # 加载PDF文件
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        # 将文档分割成块
        print(f"将文档分割成块 (大小: {chunk_size}, 重叠: {chunk_overlap})...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(docs)
        text_lines = [chunk.page_content for chunk in chunks]
        
        # 获取嵌入维度
        test_embedding = self.emb_text("测试文本")
        embedding_dim = len(test_embedding)
        
        try:
            # 如果集合存在且强制重建，先删除
            if collection_exists and force_rebuild:
                print(f"删除现有集合: {self.collection_name}")
                self.milvus_client.drop_collection(self.collection_name)
                collection_exists = False
            
            # 如果集合不存在，创建新集合
            if not collection_exists:
                print(f"创建新集合: {self.collection_name}")
                self.milvus_client.create_collection(
                    collection_name=self.collection_name,
                    dimension=embedding_dim,
                    metric_type=self.metric_type,
                    consistency_level=self.consistency_level,
                )
            
                # 准备数据并插入
                print("生成嵌入并插入数据...")
                data = []
                for i, line in enumerate(tqdm(text_lines, desc="创建嵌入")):
                    data.append({
                        "id": i, 
                        "vector": self.emb_text(line), 
                        "text": line
                    })
                
                # 插入数据到Milvus
                insert_res = self.milvus_client.insert(collection_name=self.collection_name, data=data)
                print(f"成功插入 {insert_res['insert_count']} 条数据")
            
        except Exception as e:
            print(f"操作Milvus时出错: {str(e)}")
            print("请检查Milvus服务器状态和配置")
            raise
    
    def retrieve(self, question, top_k=None):
        """
        检索与问题相关的文档
        
        Args:
            question: 问题文本
            top_k: 返回前k个结果（None表示使用配置文件中的值）
            
        Returns:
            检索到的文档文本
        """
        # 使用配置文件中的值（如果未指定）
        top_k = top_k or self.top_k
        
        # 将问题转换为嵌入向量
        question_embedding = self.emb_text(question)
        
        # 在Milvus中搜索相似向量
        search_res = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[question_embedding],
            limit=top_k,
            search_params={"metric_type": self.metric_type, "params": {}},
            output_fields=["text"],
        )
        
        # 提取检索到的文本
        retrieved_lines = [res["entity"]["text"] for res in search_res[0]]
        
        # 将检索到的文本合并为一个字符串
        return "\n".join(retrieved_lines)
