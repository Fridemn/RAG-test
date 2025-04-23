import datetime
from .base_tool import BaseTool

class DateTool(BaseTool):
    """
    日期工具：显示当前日期和时间
    """
    
    @property
    def name(self) -> str:
        return "/date"
        
    @property
    def description(self) -> str:
        return "显示当前日期和时间"
        
    @property
    def requires_llm(self) -> bool:
        return False
        
    def execute(self, *args, **kwargs) -> str:
        now = datetime.datetime.now()
        return f"当前日期时间: {now.strftime('%Y年%m月%d日 %H:%M:%S')}"