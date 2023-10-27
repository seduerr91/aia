FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

\```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
\```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the following format(the prefix of "Thought: " and "{ai_prefix}: " are must be included):

\```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
\```"""

user_examples = [
    "Find the last mail in my gmail account.",
    "How many people live in Canada?",
    "What is 2 to the 30th power?",
    "If x+y=10 and x-y=4, what are x and y?",
    "How much did it rain in SF today?",
    "Let's play three truths and a lie",
    "Let's play a game of rock paper scissors",
    "Search Google to find out when the next FC Sounders Game will take place.",
]

MAX_TOKENS = 512

TOOLS = ['pal-math', 'serpapi',]  # 'google-search', 'news-api']
