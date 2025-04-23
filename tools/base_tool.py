from abc import ABC, abstractmethod

class BaseTool(ABC):
    """
    工具函数的抽象基类
    所有工具函数都应该继承这个类并实现其方法
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称，用于调用，例如'/date'"""
        pass
        
    @property
    @abstractmethod
    def description(self) -> str:
        """工具的描述"""
        pass
        
    @property
    @abstractmethod
    def requires_llm(self) -> bool:
        """是否需要将结果传递给LLM处理
        
        如果为True，工具的输出将被发送给LLM处理
        如果为False，工具的输出将直接展示给用户
        """
        pass
        
    @abstractmethod
    def execute(self, *args, **kwargs) -> str:
        """
        执行工具功能
        
        Returns:
            str: 工具执行的结果
        """
        pass