import os
import importlib
import inspect
from typing import Dict, List, Optional, Type
from .base_tool import BaseTool

class ToolManager:
    """
    工具函数管理器
    负责管理所有可用的工具函数
    """
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        
    def register_tool(self, tool: BaseTool) -> None:
        """
        注册一个工具函数
        
        Args:
            tool: 工具函数实例
        """
        self.tools[tool.name] = tool
        
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        根据名称获取工具函数
        
        Args:
            name: 工具函数名称
            
        Returns:
            工具函数实例或None（如果不存在）
        """
        return self.tools.get(name)
        
    def load_tools_from_directory(self, directory: str = "tools") -> None:
        """
        从指定目录加载所有工具函数
        
        Args:
            directory: 工具函数所在目录
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        tools_dir = os.path.join(os.path.dirname(current_dir), directory)
        
        # 获取所有Python文件
        for filename in os.listdir(tools_dir):
            if filename.endswith(".py") and not filename.startswith("__") and filename not in ["base_tool.py", "tool_manager.py"]:
                module_name = filename[:-3]  # 移除.py扩展名
                
                try:
                    # 动态导入模块
                    module_path = f"tools.{module_name}"
                    module = importlib.import_module(module_path)
                    
                    # 查找继承自BaseTool的类
                    for _, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, BaseTool) and 
                            obj.__module__ == module.__name__):
                            
                            # 实例化并注册工具
                            tool_instance = obj()
                            self.register_tool(tool_instance)
                except Exception as e:
                    print(f"加载工具 {module_name} 时出错: {str(e)}")
    
    def execute_tool(self, command: str, *args, **kwargs) -> tuple[str, bool]:
        """
        执行工具函数
        
        Args:
            command: 工具命令（例如'/date'）
            *args, **kwargs: 传递给工具函数的参数
            
        Returns:
            tuple: (执行结果, 是否需要LLM处理)
        """
        tool = self.get_tool(command)
        if tool:
            result = tool.execute(*args, **kwargs)
            return result, tool.requires_llm
        return f"未找到工具: {command}", False
    
    def get_available_tools(self) -> List[tuple[str, str]]:
        """
        获取所有可用工具的列表
        
        Returns:
            List[tuple]: 工具名称和描述的列表
        """
        return [(tool.name, tool.description) for tool in self.tools.values()]