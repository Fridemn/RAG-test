from .base_tool import BaseTool

class AskTool(BaseTool):
    """
    提问工具：将问题直接提交给LLM
    """
    
    @property
    def name(self) -> str:
        return "/ask"
        
    @property
    def description(self) -> str:
        return "直接向LLM提出一个问题，格式：/ask 你的问题"
        
    @property
    def requires_llm(self) -> bool:
        return True
        
    def execute(self, query="", *args, **kwargs) -> str:
        if not query:
            return "请在/ask后提供您的问题，例如：/ask 今天天气怎么样？"
        return query