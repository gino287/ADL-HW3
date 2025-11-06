# #A版本
# from typing import List
# import re
    
# def get_inference_system_prompt() -> str:
#     return "You are an assistant."

# def get_inference_user_prompt(query: str, context_list: List[str]) -> str:
#     context = "\n\n".join(context_list)
#     prompt = f"""
# Answer the following question based on the given context.

# Context:
# {context}

# Question:
# {query}

# Answer:
# """
#     return prompt.strip()

# def parse_generated_answer(pred_ans: str) -> str:
#     return pred_ans.strip()


#B版本
from typing import List
import re

def get_inference_system_prompt() -> str:
    # 簡短、明確角色：檢索增強的知識提取專家
    return "You are a retrieval-augmented knowledge extraction specialist. Answer strictly based on the provided Context."

def get_inference_user_prompt(query: str, context_list: List[str]) -> str:
    context = "\n\n".join(context_list)
    prompt = f"""
Answer the following question using ONLY the information from the Context.
Respond in ONE short sentence.
After your answer, on a new line, write:
according to:[paste a supporting phrase copied verbatim from the Context]

Context:
{context}

Question:
{query}

Answer:
""".strip()
    return prompt

def parse_generated_answer(pred_ans: str) -> str:
    s = pred_ans

    # 1) 從最後一個 "Answer:" 後開始（避免復誦）
    idx = s.lower().rfind("answer:")
    if idx != -1:
        s = s[idx + len("answer:"):]

    # 2) 清理常見雜訊
    s = re.sub(r'^\s*assistant\s*$', '', s, flags=re.IGNORECASE | re.MULTILINE)
    s = re.sub(r'<\s*think\s*>.*?<\s*/\s*think\s*>', '', s,
               flags=re.IGNORECASE | re.DOTALL)

    s = s.strip()

    # 3) 取「最後一個」 according to:[...]
    acc = None
    for m in re.finditer(r'according to:\s*\[(.+?)\]',
                         s, flags=re.IGNORECASE | re.DOTALL):
        acc = m

    if acc:
        answer_text = s[:acc.start()].strip()
        # 若 according to 前沒內容，視為無法回答
        if len(answer_text) == 0:
            return "CANNOTANSWER"
        # 有內容就只回答案本體（不帶引用，避免稀釋 CosSim）
        return answer_text

    # 若沒有 according to，就直接回整段（照舊）
    return s.strip()

# #C版本
# import re

# def get_inference_system_prompt() -> str:
#     return (
#         "You are a retrieval-augmented question answering specialist. "
#         "Your goal is to extract accurate answers from the provided Context. "
#         "Do not include any reasoning traces, self-talk, or speculative content."
#     )

# def get_inference_user_prompt(query: str, context_list: list[str]) -> str:
#     context = "\n\n".join(context_list)
#     prompt = f"""
# Answer the following question using ONLY the information explicitly stated in the Context.
# Do NOT include your reasoning process or thoughts.
# Respond in ONE short, factual sentence.
# If the answer is not explicitly found in the Context, reply exactly:
# CANNOTANSWER
# and do NOT include any citation.

# Otherwise, after your answer, add a new line:
# according to:[copy 1–2 complete sentences from the Context body text that support your answer]

# Make sure:
# - Do NOT restate these instructions.
# - Do NOT fabricate or summarize new information beyond the Context.
# - Do NOT include reasoning or self-reflection.

# Context:
# {context}

# Question:
# {query}

# Answer:
# """.strip()
#     return prompt

# def parse_generated_answer(pred_ans: str) -> str:
#     s = pred_ans or ""
#     for line in s.splitlines():
#         if line.strip():
#             if re.fullmatch(r"CANNOTANSWER", line.strip(), flags=re.IGNORECASE):
#                 return "CANNOTANSWER"
#             break 

#     idx = s.lower().rfind("answer:")
#     if idx != -1:
#         s = s[idx + len("answer:"):]

#     s = re.sub(r'^\s*assistant\s*$', '', s, flags=re.IGNORECASE | re.MULTILINE)
#     s = re.sub(r'<\s*think\s*>.*?<\s*/\s*think\s*>', '', s, flags=re.IGNORECASE | re.DOTALL)
#     s = re.sub(r'<[^>]+>', '', s, flags=re.DOTALL)
#     s = s.strip()

#     acc_match = None
#     for m in re.finditer(r'according to:\s*\[(.+?)\]', s, flags=re.IGNORECASE | re.DOTALL):
#         acc_match = m

#     if acc_match is None:
#         if re.search(r'\bCANNOTANSWER\b', s, flags=re.IGNORECASE):
#             return "CANNOTANSWER"
#         return "CANNOTANSWER"

#     answer_text = s[:acc_match.start()].strip()
#     citation = acc_match.group(1).strip().strip('"\'')

#     if len(answer_text) < 3:
#         return "CANNOTANSWER"

#     citation_min_chars = 8 
#     bad_labels = {"description", "summary", "introduction", "overview", "caption", "title", "name", "label"}
#     if len(re.sub(r'\s+', '', citation)) < citation_min_chars or citation.lower() in bad_labels:
#         return "CANNOTANSWER"

#     if re.search(r'\bCANNOTANSWER\b', answer_text, flags=re.IGNORECASE):
#         return "CANNOTANSWER"

#     first_sentence = _take_first_sentence(answer_text)
#     return first_sentence.strip()

# def _take_first_sentence(text: str) -> str:
#     t = (text or "").strip()
#     t = t.splitlines()[0].strip()
#     m = re.search(r'[。．\.!?？]', t)
#     if m:
#         return t[:m.end()]
#     return t[:300]
