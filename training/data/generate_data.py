import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

GENERATOR_MODEL = "gpt-4o" 
PROMPT_TEMPLATE_FILE = "generator_prompt_template.txt"
INPUT_QUESTIONS_FILE = "input_questions.txt"
OUTPUT_DATA_FILE = "synthetic_data.jsonl"
REQUEST_DELAY_SECONDS = 2

def load_prompt_template(filename: str) -> str:
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Prompt template file '{filename}' not found.")
        raise

def load_questions(filename: str) -> list[str]:
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Warning: Input questions file '{filename}' not found. No questions to process.")
        return []

def generate_sample(client: OpenAI, prompt_template: str, question: str) -> dict | None:
    full_prompt = prompt_template.format(user_question=question)
    
    print(f"Processing question: '{question}'...")
    try:
        response = client.chat.completions.create(
            model=GENERATOR_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert-level data generator for AI model fine-tuning."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.5,
            max_tokens=32786
        )
        
        content = response.choices[0].message.content
        try:
            if content.strip().startswith("```json"):
                content = content.strip().replace("```json", "").replace("```", "").strip()
            
            data = json.loads(content)
            
            if "Q" in data and "A" in data and "<insight>" in data["A"] and "<think>" in data["A"]:
                print(" -> Successfully generated and parsed a valid sample.")
                return data
            else:
                print(" -> Error: Generated content does not match the required format.")
                print(f"   Content: {content}")
                return None
        except json.JSONDecodeError:
            print(f" -> Error: Failed to parse JSON from LLM response.")
            print(f"   Content: {content}")
            return None

    except Exception as e:
        print(f" -> Error: An API call failed. {e}")
        return None

def main():
    client = OpenAI(api_key=API_KEY)
    
    try:
        prompt_template = load_prompt_template(PROMPT_TEMPLATE_FILE)
    except FileNotFoundError:
        return
        
    questions = load_questions(INPUT_QUESTIONS_FILE)
    if not questions:
        print("No questions to process. Exiting.")
        return

    print(f"Found {len(questions)} questions to process.")
    
    with open(OUTPUT_DATA_FILE, 'a', encoding='utf-8') as f:
        for question in questions:
            sample = generate_sample(client, prompt_template, question)
            
            if sample:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            time.sleep(REQUEST_DELAY_SECONDS)
            
    print(f"\nData generation complete. Samples saved to '{OUTPUT_DATA_FILE}'.")


if __name__ == "__main__":
    main()
