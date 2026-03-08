import openai

# Add your OpenAI API Key
openai.api_key = "7e48b22b6a11457a84054850cbdb4533"

model = "gpt-4o-mini"

def ask_gpt(prompt):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response['choices'][0]['message']['content']


# Problem Statement
problem = "If a train travels 60 km in 1 hour, how far will it travel in 5 hours?"

print("\nPROBLEM:", problem)

# 1. ZERO SHOT PROMPTING

zero_shot_prompt = f"""
Solve the following problem:

{problem}
"""

print("\nZERO SHOT RESULT:")
print(ask_gpt(zero_shot_prompt))

# 2. FEW SHOT PROMPTING

few_shot_prompt = f"""
Example 1:
Q: A car travels 50 km in 1 hour. How far in 4 hours?
A: Distance = Speed × Time
50 × 4 = 200 km

Example 2:
Q: A bike travels 30 km in 1 hour. How far in 3 hours?
A: Distance = Speed × Time
30 × 3 = 90 km

Now solve:
Q: {problem}
"""

print("\nFEW SHOT RESULT:")
print(ask_gpt(few_shot_prompt))

# 3. CHAIN OF THOUGHT (COT)

cot_prompt = f"""
Solve step by step.

Problem:
{problem}

Explain the reasoning before giving the final answer.
"""

print("\nCHAIN OF THOUGHT RESULT:")
print(ask_gpt(cot_prompt))

# 4. TREE OF THOUGHT (TOT)

tot_prompt = f"""
Solve the problem using multiple reasoning paths and choose the best solution.

Problem:
{problem}

Step 1: Think of possible approaches.
Step 2: Evaluate them.
Step 3: Choose the best solution.
"""

print("\nTREE OF THOUGHT RESULT:")
print(ask_gpt(tot_prompt))


# 5. INTERVIEW PROMPTING

interview_prompt = f"""
You are interviewing a student.

Ask yourself questions and answer them step by step to solve the problem.

Problem:
{problem}
"""

print("\nINTERVIEW PROMPT RESULT:")
print(ask_gpt(interview_prompt))