import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from openai import OpenAI

# 初始化OpenAI客户端
# 注意：需要设置OPENAI_API_KEY环境变量
try:
    client = OpenAI()
except Exception as e:
    print("警告：OpenAI客户端初始化失败，请确保已设置正确的API密钥。")
    client = None

# 评分提示词
QUESTION_QUALITY_PROMPT_EN_NO_COT = (
    "Please as a grading expert, judge whether the final answers given by the candidates below are consistent with "
    "the standard answers, that is, whether the candidates answered correctly. "
    "Here are some evaluation criteria: "
    "1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because "
    "the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the "
    "standard answer according to the form of the question. THE STANDARD ANSWER IS ALWAYS CORRECT AND THE QUESTION IS "
    "PERFECTLY VALID. NEVER QUESTION THEM. "
    "2. ONLY compare the FINAL ANSWER - COMPLETELY IGNORE any potential errors in the REASONING PROCESSES. "
    "3. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some "
    "answers may be a textual description, as long as the meaning expressed is the same. Before making a judgment, "
    "please understand the question and the standard answer first, and then judge whether the candidate's answer is "
    "correct.If the standard answer does not specify a unit, but the candidate's answer includes a unit that is "
    "correct for the value given, consider it correct. "
    "4. Some answers may consist of multiple items, such as multiple-choice questions, multiple-select questions, "
    "fill-in-the-blank questions, etc. Regardless of the question type, the final answer will be considered correct "
    "as long as it matches the standard answer, regardless of whether the reasoning process is correct. For "
    "multiple-select questions and multi-blank fill-in-the-blank questions, all corresponding options or blanks must "
    "be answered correctly and match the standard answer exactly to be deemed correct. "
    "5. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the "
    "candidate's answer is consistent with the standard answer. "
    "6. If the candidate's answer is invalid (e.g., incomplete (cut off mid-response), lots of unnormal repetitive "
    "content, or irrelevant to the question, saying it can't answer the question because some irresistible factors, "
    "like ethical issues, no enough information, etc.), select option C (INVALID).Please judge whether the following "
    "answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this "
    "new question as one of: "
    "A: CORRECT "
    "B: INCORRECT "
    "C: INVALID "
    'Just return the letters "A", "B", or "C", with no text around it. '
    "Here is your task. Simply reply with either CORRECT, INCORRECT, or INVALID. Don't apologize or correct yourself "
    "if there was a mistake; we are just trying to grade the answer. "
    "\n<Original Question Begin>: "
    "{question} "
    "\n<Original Question End> "
    "\n<Standard Answer Begin>: "
    "{gold_answer} "
    "\n<Standard Answer End> "
    "\n<Candidate's Answer Begin>: "
    "{llm_response} "
    "\n<Candidate's Answer End> "
    "\nJudging the correctness of the candidate's answer:"
)


def extract_final_answer(response_text: str) -> str:
    """
    从带有思考过程的回复中提取最终答案
    如果有<think>标签，则只取<think>之后的内容
    """
    # 查找<think>标签后的最终答案
    thinking_match = re.search(r"<think>(.*?)</think>(.*)", response_text, re.DOTALL)
    if thinking_match:
        # 如果找到<think>标签，只取</think>后面的部分作为最终答案
        return thinking_match.group(2).strip()

    # 如果没有找到标准标签，尝试查找其他可能的格式
    final_answer_match = re.search(r"</think>(.*)", response_text, re.DOTALL)
    if final_answer_match:
        return final_answer_match.group(1).strip()

    # 否则返回整个回答
    return response_text.strip()


def evaluate_answer_with_gpt(
    question: str, gold_answer: str, llm_response: str, max_retries: int = 3
) -> str:
    """
    使用GPT判断答案正确性，添加失败重试逻辑
    """
    if client is None:
        return "C"  # 如果没有可用的OpenAI客户端，则返回无效

    # 提取最终答案
    final_answer = extract_final_answer(llm_response)

    for attempt in range(max_retries):
        try:
            prompt = QUESTION_QUALITY_PROMPT_EN_NO_COT.format(
                question=question, gold_answer=gold_answer, llm_response=final_answer
            )

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )

            result = response.choices[0].message.content.strip()
            # 确保返回的是A、B或C之一
            if result in ["A", "B", "C"]:
                return result
            else:
                print(f"第{attempt + 1}次尝试：GPT返回无效结果: {result}")
                if attempt < max_retries - 1:
                    continue
                else:
                    return "C"  # 如果返回的不是预期结果，则认为无效
        except Exception as e:
            print(f"第{attempt + 1}次尝试：GPT评估出错: {e}")
            if attempt < max_retries - 1:
                continue
            else:
                return "C"  # 出错时返回无效

    return "C"  # 所有重试都失败时返回无效


def process_single_entry(entry_data: dict[str, Any]) -> dict[str, Any] | None:
    """
    处理单个JSON条目
    """
    try:
        line_num = entry_data["line_num"]
        data = entry_data["data"]

        # 提取必要字段
        question = data.get("example_meta", {}).get("problem", "")
        gold_answers = data.get("gold_answers", [])
        response = data.get("response", "")

        # 将gold_answers列表转换为字符串
        gold_answer_str = ", ".join(gold_answers) if gold_answers else ""

        # 使用GPT评估答案（现在包含重试逻辑）
        result = evaluate_answer_with_gpt(question, gold_answer_str, response)

        return {
            "line_number": line_num,
            "question": question,
            "gold_answer": gold_answer_str,
            "model_response": response,
            "evaluation": result,
        }
    except Exception as e:
        print(f"处理第{entry_data['line_num']}行时出错: {e}")
        return None


def process_jsonl_file(file_path: str, max_workers: int = 5) -> dict[str, Any]:
    """
    处理JSONL文件并评估所有答案（支持多线程）
    """
    entries = []

    # 读取所有条目
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                entries.append({"line_num": line_num, "data": data})
            except json.JSONDecodeError:
                print(f"第{line_num}行JSON解析错误")
                continue
            except Exception as e:
                print(f"读取第{line_num}行时出错: {e}")
                continue

    # 使用线程池处理条目
    results = []
    correct_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_entry = {
            executor.submit(process_single_entry, entry): entry for entry in entries
        }

        # 收集结果
        for future in as_completed(future_to_entry):
            result = future.result()
            if result is not None:
                results.append(result)
                if result["evaluation"] == "A":
                    correct_count += 1

    # 按行号排序结果
    results.sort(key=lambda x: x["line_number"])

    return {
        "results": results,
        "total": len(results),
        "correct_count": correct_count,
        "accuracy": correct_count / len(results) if results else 0,
    }


def main():
    """
    主函数
    """
    file_path = "qwen3_8b_memory.jsonl"

    print("开始处理JSONL文件...")
    result = process_jsonl_file(file_path, max_workers=10)

    print(f"\n处理完成！")
    print(f"总条目数: {result['total']}")
    print(f"正确答案数: {result['correct_count']}")
    print(f"准确率: {result['accuracy']:.2%}")

    # 保存详细结果到文件
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n详细结果已保存到 evaluation_results.json")


if __name__ == "__main__":
    main()
