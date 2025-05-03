import gradio as gr
import os
import argparse
import json
import socket
from llm_client import LLMClient
import glob

def load_config(config_path='config.json'):
    """加载配置文件"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"警告: 无法加载配置文件 {config_path}: {str(e)}")
        return {}

def find_pdf_files(directory='.'):
    """查找指定目录中的所有PDF文件"""
    return [os.path.basename(f) for f in glob.glob(os.path.join(directory, "*.pdf"))]

def find_available_port(start_port=7860, max_attempts=20):
    """查找可用端口"""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    return start_port

# 全局客户端实例
client = None

def init_client(api_key, base_url, model, use_rag, collection_name, milvus_uri):
    """初始化LLM客户端"""
    global client
    
    # 创建临时配置
    temp_config = {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "rag": {
            "enabled": use_rag,
            "milvus": {
                "collection_name": collection_name,
                "uri": milvus_uri
            }
        }
    }
    
    # 保存临时配置
    with open("temp_config.json", "w") as f:
        json.dump(temp_config, f, indent=4)
    
    # 初始化客户端
    try:
        client = LLMClient(
            config_path="temp_config.json",
            use_rag=use_rag,
            model=model,
            api_key=api_key
        )
        return "客户端初始化成功！" + ("RAG系统已启用。" if use_rag else "RAG系统未启用。")
    except Exception as e:
        return f"初始化客户端时出错: {str(e)}"

def load_data(pdf_file, force_rebuild):
    """加载数据到RAG系统"""
    global client
    
    if not client:
        return "请先初始化客户端"
        
    if not client.use_rag:
        return "RAG系统未启用，无法加载数据"
        
    try:
        client.rag_system.load_data(pdf_file, force_rebuild=force_rebuild)
        return f"成功加载PDF: {pdf_file}" + (" (强制重建)" if force_rebuild else "")
    except Exception as e:
        return f"加载数据时出错: {str(e)}"

def process_message(message, history):
    """处理用户消息"""
    global client
    
    if not client:
        return history + [(message, "请先初始化客户端后再提问")]
        
    try:
        response = client.call_llm(message)
        return history + [(message, response)]
    except Exception as e:
        return history + [(message, f"处理消息时出错: {str(e)}")]

def list_tools():
    """列出可用的工具"""
    global client
    
    if not client:
        return "请先初始化客户端"
        
    tools = client.tool_manager.get_available_tools()
    tool_list = "\n".join([f"- {name}: {desc}" for name, desc in tools])
    return f"可用工具列表:\n{tool_list}"

def execute_tool(tool_command, query):
    """执行工具命令"""
    global client
    
    if not client:
        return "请先初始化客户端"
        
    if not tool_command.startswith('/'):
        tool_command = '/' + tool_command
        
    try:
        result, requires_llm = client.tool_manager.execute_tool(tool_command, query=query, tool_manager=client.tool_manager)
        if requires_llm and client:
            result = client.call_llm(result)
        return result
    except Exception as e:
        return f"执行工具命令时出错: {str(e)}"

def reload_pdfs():
    """重新加载PDF文件列表"""
    updated_files = find_pdf_files()
    return gr.Dropdown.update(choices=updated_files, value=updated_files[0] if updated_files else "")

# 创建简化版界面，避免复杂的嵌套结构
def create_simple_ui():
    config = load_config()
    rag_config = config.get('rag', {})
    
    # 设置默认值
    default_api_key = config.get('api_key', '')
    default_model = config.get('model', 'gpt-3.5-turbo')
    default_base_url = config.get('base_url', '')
    default_use_rag = rag_config.get('enabled', False)
    default_collection_name = rag_config.get('milvus', {}).get('collection_name', 'rag_collection')
    default_uri = rag_config.get('milvus', {}).get('uri', 'http://localhost:19530')
    
    # 查找PDF文件
    pdf_files = find_pdf_files()
    default_pdf = rag_config.get('documents', {}).get('pdf_path', '') if rag_config.get('documents', {}).get('pdf_path', '') in pdf_files else (pdf_files[0] if pdf_files else '')
    
    # 设置选项卡
    with gr.Blocks(title="RAG系统图形界面") as demo:
        gr.Markdown("# 📚 RAG系统图形界面\n\n基于OpenAI和Milvus实现的检索增强生成系统")
        
        with gr.Tab("设置"):
            api_key = gr.Textbox(label="OpenAI API密钥", value=default_api_key, type="password")
            base_url = gr.Textbox(label="API基础URL", value=default_base_url)
            model = gr.Dropdown(
                label="模型", 
                choices=["gpt-4", "gpt-4-1106-preview", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
                value=default_model
            )
            use_rag = gr.Checkbox(label="启用RAG系统", value=default_use_rag)
            collection_name = gr.Textbox(label="集合名称", value=default_collection_name)
            milvus_uri = gr.Textbox(label="Milvus服务器URI", value=default_uri)
            
            init_btn = gr.Button("初始化客户端")
            init_output = gr.Textbox(label="初始化状态", interactive=False)
            
            init_btn.click(
                fn=init_client,
                inputs=[api_key, base_url, model, use_rag, collection_name, milvus_uri],
                outputs=init_output
            )
        
        with gr.Tab("数据加载"):
            pdf_file = gr.Dropdown(label="PDF文件", choices=pdf_files, value=default_pdf)
            reload_btn = gr.Button("刷新PDF列表")
            reload_btn.click(fn=reload_pdfs, outputs=pdf_file)
            
            force_rebuild = gr.Checkbox(label="强制重建集合", value=False)
            load_btn = gr.Button("加载数据")
            load_output = gr.Textbox(label="加载状态", interactive=False)
            
            load_btn.click(
                fn=load_data,
                inputs=[pdf_file, force_rebuild],
                outputs=load_output
            )
        
        with gr.Tab("聊天"):
            chatbot = gr.Chatbot(type="messages")  # 修改为推荐的格式
            msg = gr.Textbox(label="发送消息", placeholder="输入您的问题...")
            send_btn = gr.Button("发送")
            clear_btn = gr.Button("清除聊天记录")
            
            send_btn.click(
                fn=process_message,
                inputs=[msg, chatbot],
                outputs=chatbot
            ).then(
                lambda: "", None, msg  # 清空输入框
            )
            
            msg.submit(
                fn=process_message,
                inputs=[msg, chatbot],
                outputs=chatbot
            ).then(
                lambda: "", None, msg  # 清空输入框
            )
            
            clear_btn.click(lambda: [], None, chatbot)
        
        with gr.Tab("工具"):
            list_tools_btn = gr.Button("列出可用工具")
            tools_output = gr.Textbox(label="工具列表", interactive=False)  # 移除height参数
            
            list_tools_btn.click(
                fn=list_tools,
                outputs=tools_output
            )
            
            tool_command = gr.Textbox(label="工具命令 (例如: /date)", value="/help")
            tool_query = gr.Textbox(label="参数 (可选)")
            execute_btn = gr.Button("执行工具")
            tool_result = gr.Textbox(label="执行结果", interactive=False)  # 移除height参数
            
            execute_btn.click(
                fn=execute_tool,
                inputs=[tool_command, tool_query],
                outputs=tool_result
            )
        
        with gr.Tab("关于"):
            gr.Markdown("""
            ## RAG系统图形界面
            
            这是一个基于OpenAI和Milvus向量数据库实现的RAG（检索增强生成）系统的图形界面。
            
            ### 使用说明:
            
            1. **设置**: 配置API密钥和RAG系统参数
            2. **数据加载**: 选择PDF文件并加载到向量数据库
            3. **聊天**: 与基于RAG的AI系统交互
            4. **工具**: 使用和测试系统提供的工具
            
            ### 提示:
            - 确保Milvus服务器已启动（如果使用外部Milvus服务）
            - 配置正确的API密钥和基础URL
            - RAG系统需要先加载数据才能基于文档回答问题
            """)
    
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG系统图形界面")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")
    parser.add_argument("--port", type=int, default=None, help="服务器端口")
    parser.add_argument("--share", action="store_true", help="创建公开链接")
    args = parser.parse_args()
    
    # 加载指定配置文件
    if args.config != "config.json":
        config = load_config(args.config)
    
    # 查找可用端口
    if args.port is None:
        port = find_available_port(20000)  # 从20000端口开始查找
    else:
        port = args.port
    
    print(f"尝试在端口 {port} 上启动服务...")
    
    # 使用简化版界面
    demo = None  # 初始化demo变量
    try:
        # 创建界面
        demo = create_simple_ui()
        
        # 启动服务
        demo.launch(
            server_port=port, 
            share=True, 
            server_name="0.0.0.0"  # 监听所有网络接口
        )
    except Exception as e:
        print(f"启动服务失败: {str(e)}")
        print("尝试使用备用配置...")
        
        # 确保demo已创建
        if demo is None:
            try:
                demo = create_simple_ui()
            except Exception as e2:
                print(f"创建界面失败: {str(e2)}")
                exit(1)
                
        # 使用备用配置
        try:
            # 尝试最简化的配置
            demo.launch(server_port=0, share=True)
        except Exception as e3:
            print(f"备用配置也失败了: {str(e3)}")
            print("尝试最后的方案...")
            # 最后的尝试，使用最基本的配置
            demo.launch()