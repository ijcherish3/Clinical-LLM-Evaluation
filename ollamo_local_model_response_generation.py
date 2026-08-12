import ollama
import pandas as pd


# Load dataset
df = pd.read_csv("medical_pipeline_eval_dataset.csv")


# Same HealthGPT instructions used for Gemini
system_prompt = """
You are HealthGPT, a careful, factual health assistant.

Rules:
- Ground every claim in the records and health data shown below.
- Do not invent values, dates, diagnoses, or medications.
- When you cite numbers, dates, units, or medication names, quote them verbatim from the records.
- If the records contain partial information, answer with what is available and note what is missing.
- If the records contain nothing relevant, say so plainly.
- Do not give a diagnosis or prescription.
- Be concise. Prefer short, structured answers.
"""


def ask_model(row):

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"""
Chart Context:

{row['retrieved_json']}

Question:

{row['question']}
"""
            }
        ]
    )

    return response["message"]["content"]


print("Running Llama 3.2 3B...")

df["llama3_2_answer"] = df.apply(
    ask_model,
    axis=1
)

df.to_csv(
     "llama3_2_results.csv",
    index=False
)

print("Finished!")
