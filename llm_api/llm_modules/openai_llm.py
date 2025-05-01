#!/usr/bin/env python3
import json
from openai import OpenAI
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Prompt Templates
SUMMARY_PROMPT_TEMPLATE = """
You are an expert summarizer. Given the lecture transcript, provide a concise summary in 100-150 words, highlighting 4-6 key topics.

Format:
Summary: [brief summary]
Topics:
- Topic 1
- Topic 2
- ...

Transcript:
{transcript}
"""

SYSTEM_PROMPT_TEMPLATE = """
You are a professional interviewer assessing a user's understanding of a lecture based on the provided summary. Your task is to ask questions about the lecture content and evaluate responses.

Instructions:
- Ask one question at a time, covering the key topics: {topics}.
- Output only a JSON object with 'type' ('question' or 'conclusion') and 'content' (the question text or conclusion message).
- If the response is off-topic, don't go astray and please focus on the lecture content.
- After three off-topic responses, ask a targeted question from an unaddressed topic.
- Ask follow-up questions if the response is unique and interesting and also if the response is incomplete or needs clarification; otherwise, move to a new topic.
- Maintain a professional tone and track conversation history to ensure relevance.

Lecture Summary:
{summary}
"""

CUSTOM_PROMPT_TEMPLATE = """
You are a professional interviewer creating a customized interview.

Instructions:
- Ask one question at a time on the topic: "{topic}".
- Match the difficulty level: "{difficulty}".
- Tailor the questions to suit a candidate with experience level: "{experience}".
- Use a "{tone}" tone while asking.
- Output only a JSON object with 'type' ('question' or 'conclusion') and 'content' (question text or conclusion message).
- If a response is off-topic or insufficient, guide the user back or ask for clarification.

Start the interview now.
"""

client = None

def initialize_client():
    global client
    if client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        client = OpenAI(api_key=api_key)
    return client

def generate_summary(transcript):
    client = initialize_client()
    prompt = SUMMARY_PROMPT_TEMPLATE.format(transcript=transcript)

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert summarizer."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.3
    )

    text = resp.choices[0].message.content.strip()
    summary = ""
    topics = []

    for line in text.splitlines():
        if line.startswith("Summary:"):
            summary = line.split("Summary:", 1)[1].strip()
        elif line.startswith("- "):
            topics.append(line[2:].strip())

    return summary, topics

def get_next_response(messages):
    client = initialize_client()

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=200,
        temperature=0.7,
        response_format={"type": "json_object"}
    )

    return resp.choices[0].message.content.strip()

def run_interview(state, user_answer=None, store_question_fn=None, store_user_response_fn=None, store_conversation_fn=None):
    """
    Modified to handle API-driven interview process.
    Args:
        state: Dict containing 'transcript', 'mode', 'custom_info', 'messages', 'question_count', 'conversation_log', 'off_topic_count', 'provider'
        user_answer: User's answer to the current question (None for first question)
        store_question_fn, store_user_response_fn, store_conversation_fn: Functions to store data
    Returns:
        Dict with 'status', 'type', 'content', 'state' (updated state)
    """
    if not store_question_fn or not store_user_response_fn or not store_conversation_fn:
        return {"status": "error", "message": "Storage functions not provided"}

    try:
        # Initialize state if empty
        if not state.get("messages"):
            mode = state.get("mode", "lecture")
            transcript = state.get("transcript", "")
            custom_info = state.get("custom_info", {})
            conversation_log = ""

            if mode == "custom":
                topic = custom_info.get("topic", "General")
                difficulty = custom_info.get("difficulty", "Intermediate")
                experience = custom_info.get("experience", "Fresher")
                tone = custom_info.get("tone", "Professional")

                system_prompt = CUSTOM_PROMPT_TEMPLATE.format(
                    topic=topic,
                    difficulty=difficulty,
                    experience=experience,
                    tone=tone
                )
                conversation_log = f"Custom Interview\nTopic: {topic}\nDifficulty: {difficulty}\nExperience: {experience}\nTone: {tone}\n\n"
            else:
                summary, topics = generate_summary(transcript)
                system_prompt = SYSTEM_PROMPT_TEMPLATE.format(topics=', '.join(topics), summary=summary)
                conversation_log = f"Lecture Summary: {summary}\nTopics: {', '.join(topics)}\n\n"

            state.update({
                "messages": [{"role": "system", "content": system_prompt}],
                "question_count": 0,
                "conversation_log": conversation_log,
                "off_topic_count": 0,
                "provider": state.get("provider", "openai")
            })

        messages = state["messages"]
        question_count = state["question_count"]
        conversation_log = state["conversation_log"]
        off_topic_count = state.get("off_topic_count", 0)

        # Process user answer if provided
        if user_answer is not None:
            store_user_response_fn(user_answer)
            conversation_log += f"Answer: {user_answer}\n"
            messages.append({"role": "user", "content": user_answer})

        # Check if max questions reached
        if question_count <= 2:
            conclusion = "Interview completed. Thank you for participating!"
            conversation_log += f"Conclusion: {conclusion}\n"
            store_conversation_fn(conversation_log)
            state["conversation_log"] = conversation_log
            return {
                "status": "success",
                "type": "conclusion",
                "content": conclusion,
                "state": state
            }

        # Get next response from LLM
        response_json = get_next_response(messages)
        try:
            response = json.loads(response_json)
            response_type = response.get("type")
            content = response.get("content")
        except (json.JSONDecodeError, KeyError):
            conversation_log += "Error: Invalid model output.\n"
            store_conversation_fn(conversation_log)
            return {
                "status": "error",
                "message": "Invalid JSON output from model",
                "state": state
            }

        if response_type == "question":
            question = content
            store_question_fn(question)
            conversation_log += f"Question {question_count + 1}: {question}\n"
            messages.append({"role": "assistant", "content": json.dumps({"type": "question", "content": question})})
            state.update({
                "messages": messages,
                "question_count": question_count + 1,
                "conversation_log": conversation_log,
                "off_topic_count": off_topic_count
            })
            return {
                "status": "success",
                "type": "question",
                "content": question,
                "state": state
            }
        elif response_type == "conclusion":
            conversation_log += f"Conclusion: {content}\n"
            store_conversation_fn(conversation_log)
            state["conversation_log"] = conversation_log
            return {
                "status": "success",
                "type": "conclusion",
                "content": content,
                "state": state
            }

    except Exception as e:
        conversation_log += f"Error: {str(e)}\n"
        store_conversation_fn(conversation_log)
        state["conversation_log"] = conversation_log
        return {
            "status": "error",
            "message": str(e),
            "state": state
        }