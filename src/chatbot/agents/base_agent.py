import sys
from pathlib import Path
from abc import ABC, abstractmethod
from langchain_core.prompts import ChatPromptTemplate

current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parents[1]
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))


class BaseAgent(ABC):
    def __init__(self):
        self.prompt = self._init_prompt()
        self._init_tools()
        self._init_agent()

    @abstractmethod
    def _init_prompt(self) -> ChatPromptTemplate:
        pass

    @abstractmethod
    def _init_tools(self) -> None:
        pass

    @abstractmethod
    def _init_agent(self) -> None:
        pass
