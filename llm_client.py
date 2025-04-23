import os
import json
from openai import OpenAI
from rag_system import RAGSystem
from tools.tool_manager import ToolManager

class LLMClient:
    def __init__(self, config_path='config.json', use_rag=None, model=None, api_key=None):
        """
        初始化LLM客户端
        
        Args:
            config_path: 配置文件路径
            use_rag: 是否使用RAG系统（None表示使用配置文件中的设置）
            model: 要使用的OpenAI模型ID（None表示使用配置文件中的模型）
            api_key: OpenAI API密钥（None表示使用配置文件中的密钥）
        """
        # 加载配置文件
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"无法加载配置文件: {str(e)}")
            self.config = {}
        
        # 优先使用传入的参数，如果没有传入则使用配置文件中的值
        self.api_key = api_key or self.config.get('api_key')
        self.model = model or self.config.get('model', 'gpt-3.5-turbo')
        self.base_url = self.config.get('base_url')
        self.organization = self.config.get('organization')
        
        # 如果use_rag未指定，从配置文件获取
        if use_rag is None:
            self.use_rag = self.config.get('rag', {}).get('enabled', False)
        else:
            self.use_rag = use_rag
        
        # 初始化OpenAI客户端
        client_params = {'api_key': self.api_key}
        if self.base_url:
            client_params['base_url'] = self.base_url
        if self.organization:
            client_params['organization'] = self.organization
            
        self.client = OpenAI(**client_params)
        
        # 如果启用RAG，初始化RAG系统
        if self.use_rag:
            self.rag_system = RAGSystem(
                config_path=config_path,
                api_key=self.api_key, 
                base_url=self.base_url, 
                organization=self.organization
            )
            
        # 初始化工具管理器
        self.tool_manager = ToolManager()
        self.tool_manager.load_tools_from_directory()
    
    def call_llm(self, prompt, max_tokens=1000):
        """
        调用OpenAI语言模型
        
        Args:
            prompt: 提示文本
            max_tokens: 生成的最大token数
            
        Returns:
            生成的回复文本
        """
        # 检查是否是工具调用
        if prompt.startswith('/'):
            # 解析命令和参数
            parts = prompt.strip().split(' ', 1)
            command = parts[0]
            query = parts[1] if len(parts) > 1 else ""
            
            # 执行工具
            result, requires_llm = self.tool_manager.execute_tool(command, query=query, tool_manager=self.tool_manager)
            
            # 如果工具需要LLM处理，将结果传递给LLM
            if requires_llm:
                return self.call_llm(result)
            else:
                # 否则直接返回结果
                return result
                
        if self.use_rag:
            # 使用RAG系统检索相关内容
            context = self.rag_system.retrieve(prompt)
            
            # 构建包含上下文的提示
            rag_prompt = f"""
Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
<context>
{context}
</context>
<question>
{prompt}
</question>
"""
            # 使用OpenAI生成回答
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": rag_prompt}
                ],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        else:
            # 直接使用OpenAI生成回答
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
