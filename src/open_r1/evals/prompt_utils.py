SYSTEM_COMPETITIVE = (
   "You are Qwen, a coding expert. Given a competition-level coding problem, you need to write a python program to solve it."
   "You may start by outlining your thought process."
   "In the end, please provide the complete code in a code block, enclosed in ```python``` tags."
   "Just output code that solves the problem and prints the output in the required output format."
   "Always output valid, runnable code."
   "Note that the input and output are in the format provided in the problem statement."
   "There will always be multiple test cases."
)

# SYSTEM_COMPETITIVE = (
# "You will be given a competitive programming problem."
# #"Analyze the maximum input constraints and identify the optimal algorithmic approach and data structures needed to process the largest possible test cases within the time and memory limits, then explain why your chosen implementation strategy is the most efficient solution. 
# #"Please reason step by step about your solution approach, then "
# "provide a complete implementation in python that is thoroughly optimized for both speed and memory usage.\n\nYour solution must read input from standard input, write output to standard output .\nDo not include any debug prints or additional output.\n\nPut your final solution within a single code block:\n```python\n<your code here>\n```"
# )

# SYSTEM_COMPETITIVE = (
#     "You will be given a competitive programming problem"
#     "Generate python code that solved the programming problem in a code block, enclosed in ```python``` tags"?
# )


def build_user_prompt(row: dict) -> str:
    """Turn a Codeforces HF row into a user prompt string."""
    parts = []
    title = row.get("title")
    if title:
        parts.append(f"# {title}")
    desc = row.get("description") or row.get("statement") or ""
    if desc:
        parts.append("Problem: " + desc)
    inp = row.get("input_format")
    out = row.get("output_format")
    if inp:
        parts += ["\n## Input", inp]
    if out:
        parts += ["\n## Output", out]
    # Add first example if available
    examples = row.get("examples") or []
    if examples:
        ex = examples[0]
        parts += [
            "\n## Example",
            f"```input\n{(ex.get('input') or '').strip()}\n```",
            f"```output\n{(ex.get('output') or '').strip()}\n```",
        ]
    return "\n\n".join([p for p in parts if p])