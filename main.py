from llm_client import LLMClient

def main():
    client = LLMClient()
    
    print("输入 'quit' 或 'exit' 退出对话")
    print("-" * 50)

    while True:
        prompt = input("\n用户: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
            
        if not prompt:
            continue
            
        try:
            response = client.call_llm(prompt)
            print("\nAI: " + response)
            print("-" * 50)
        except Exception as e:
            print(f"\n错误: {str(e)}")
            print("-" * 50)

if __name__ == "__main__":
    main()
