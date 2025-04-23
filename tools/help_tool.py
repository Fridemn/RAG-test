from .base_tool import BaseTool
from .tool_manager import ToolManager

class HelpTool(BaseTool):
    """
    帮助工具：列出所有可用的工具命令
    """
    
    @property
    def name(self) -> str:
        return "/help"
        
    @property
    def description(self) -> str:
        return "列出所有可用的工具命令"
        
    @property
    def requires_llm(self) -> bool:
        return False
        
    def execute(self, tool_manager=None, *args, **kwargs) -> str:
        if not tool_manager or not isinstance(tool_manager, ToolManager):
            return "错误：无法获取工具列表"
            
        tools = tool_manager.get_available_tools()
        result = "可用的工具命令:\n" + "-" * 30 + "\n"
        
        for name, desc in tools:
            result += f"{name}: {desc}\n"
            
        return result