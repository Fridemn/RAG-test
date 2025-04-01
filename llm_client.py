import json
from openai import OpenAI
from typing import Dict, List
import os

class LLMClient:
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.memory_path = "memory.json"
        self.memory = self._load_memory()

        self.client = OpenAI(
            api_key=self.config['api_key'],
            base_url=self.config.get('base_url'),
            organization=self.config.get('organization')
        )

    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            return json.load(f)

    def _load_memory(self) -> Dict:
        if not os.path.exists(self.memory_path):
            return {"conversations": []}
        with open(self.memory_path, 'r') as f:
            return json.load(f)

    def _save_memory(self):
        with open(self.memory_path, 'w') as f:
            json.dump(self.memory, f, indent=4)

    def call_llm(self, prompt: str) -> str:
        try:
            messages = []
            for conv in self.memory["conversations"]:
                messages.append({"role": "user", "content": conv["prompt"]})
                messages.append({"role": "assistant", "content": conv["response"]})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.config['model'],
                messages=messages
            )
            
            answer = response.choices[0].message.content
            
            self.memory["conversations"].append({
                "prompt": prompt,
                "response": answer
            })
            self._save_memory()
            
            return answer
        except Exception as e:
            raise Exception(f"API调用失败: {str(e)}")
