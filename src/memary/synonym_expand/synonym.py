from langchain_openai import OpenAI, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import List
import os
from memary.synonym_expand.output import Output
from dotenv import load_dotenv

def custom_synonym_expand_fn(keywords: str) -> List[str]:
    load_dotenv()
    # llm = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"), openai_api_key=os.getenv("OPENAI_KEY"), temperature=0, model_name=os.getenv("OPENAI_MODEL"))
    llm = ChatOpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),  # Ollama 的 OpenAI 兼容接口
        api_key="ollama",  # Ollama 不验证 API Key，可填任意值
        model=os.getenv("OPENAI_MODEL"),    # 本地模型名称（通过 `ollama list` 查看）
        temperature=0,
        max_tokens=os.getenv("OPENAI_MAX_TOKENS")
    )

    parser = JsonOutputParser(pydantic_object=Output)

    template = """
    You are an expert synonym exapnding system. Find synonyms or words commonly used in place to reference the same word for every word in the list:

    Some examples are:
    - a synonym for Palantir may be Palantir technologies or Palantir technologies inc.
    - a synonym for Austin may be Austin texas
    - a synonym for Taylor swift may be Taylor 
    - a synonym for Winter park may be Winter park resort
    - a synonym for 小明 may be 小明
    
    Format: {format_instructions}

    Text: {keywords}

    **注意**: 请严格按照输出格式给出结果，并且不要将原始文本翻译成其他语言
    """

    # **Note**: please use the format strictly, do NOT translate original language

    prompt = PromptTemplate(
        template=template,
        input_variables=["keywords"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser
    result = chain.invoke({"keywords": keywords})
    print(f"result={result}")
    l = []
    for category in result:
        for synonym in result[category]:
            l.append(synonym.capitalize())
    
    return l

# testing
# print(custom_synonym_expand_fn("[Nvidia]"))
