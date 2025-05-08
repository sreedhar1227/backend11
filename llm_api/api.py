from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .serializers import InterviewStartSerializer, InterviewAnswerSerializer
from pymongo import MongoClient
import pyodbc
import uuid
import os
from openai import OpenAI
from groq import Groq
import google.generativeai as genai
import anthropic
import json
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# List available transcripts
@api_view(['GET'])
def list_transcripts(request):
    try:
        mongo_client = MongoClient('mongodb://192.168.48.200:27017')
        db = mongo_client['video_transcriptions']
        transcripts_collection = db['transcriptions']
        transcripts = list(transcripts_collection.find(
            {},
            {"lecture_id": 1, "videoid": 1, "transcript": 1, "transcription": 1, "_id": 0}
        ))
        result = []
        for t in transcripts:
            transcript_id = t.get('lecture_id', t.get('videoid', 'Unknown'))
            transcript_text = t.get('transcript', t.get('transcription', ''))
            description = (transcript_text[:100] + '...') if transcript_text else 'No transcript available'
            result.append({
                'transcript_id': str(transcript_id),
                'description': description
            })
        return Response({
            'transcripts': result,
            'count': len(result)
        }, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Error in list_transcripts: {str(e)}")
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# SQL Server connection
def get_db_connection():
    conn_str = (
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=192.168.48.200;'
        'DATABASE=InterviewSystem;'
        'UID=sa;'
        'PWD=Welcome@123'
    )
    return pyodbc.connect(conn_str)

# Initialize LLM clients
def initialize_llm_client(provider):
    if provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        return OpenAI(api_key=api_key)
    elif provider == 'groq':
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        return Groq(api_key=api_key)
    elif provider == 'gemini':
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.0-flash')
    elif provider == 'claude':
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        return anthropic.Anthropic(api_key=api_key)
    else:
        raise ValueError("Invalid provider")

# Generate LLM response
def generate_llm_response(client, provider, messages, max_tokens=1000):
    try:
        if provider == 'openai':
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7
            )
            return json.loads(response.choices[0].message.content)
        elif provider == 'groq':
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7
            )
            try:
                return json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                if response.choices[0].message.content.strip():
                    return {"type": "question", "content": response.choices[0].message.content.strip()}
                else:
                    raise ValueError("Empty response from Groq")
        elif provider == 'gemini':
            response = client.generate_content(
                [m['content'] for m in messages if m['role'] == 'user' or m['role'] == 'system'],
                generation_config={'max_output_tokens': max_tokens, 'temperature': 0.7}
            )
            try:
                return json.loads(response.text)
            except json.JSONDecodeError:
                if response.text.strip():
                    return {"type": "question", "content": response.text.strip()}
                else:
                    raise ValueError("Empty response from Gemini")
        elif provider == 'claude':
            system_prompt = next((m['content'] for m in messages if m['role'] == 'system'), "")
            filtered_messages = [m for m in messages if m['role'] != 'system']
            if not filtered_messages:
                filtered_messages.append({"role": "user", "content": "Start the interview"})
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=max_tokens,
                temperature=0.7,
                system=system_prompt,
                messages=filtered_messages
            )
            try:
                return json.loads(response.content[0].text)
            except json.JSONDecodeError:
                if response.content[0].text.strip():
                    return {"type": "question", "content": response.content[0].text.strip()}
                else:
                    raise ValueError("Empty response from Claude")
    except Exception as e:
        logger.error(f"LLM response generation failed: {str(e)}")
        raise

# Evaluate answer
def evaluate_answer(client, provider, question, answer, transcript, mode, custom_info):
    try:
        evaluation_prompt = (
            "You are an expert evaluator. Given a question, the user's answer, and context (transcript or custom info), "
            "evaluate the answer's accuracy, relevance, and completeness. Assign a score from 0 to 100, where 100 is a perfect answer. "
            "Return a JSON object with 'score' (integer) and 'feedback' (brief explanation of the score). "
            "Consider the context to ensure the answer aligns with the lecture content or custom topic requirements.\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Context: {transcript if mode == 'lecture' else json.dumps(custom_info)}\n\n"
            "Output format: {'score': <int>, 'feedback': '<string>'}"
        )

        messages = [
            {"role": "system", "content": "You are an expert evaluator."},
            {"role": "user", "content": evaluation_prompt}
        ]

        response = generate_llm_response(client, provider, messages, max_tokens=200)
        score = response.get('score', 50)
        feedback = response.get('feedback', 'No feedback provided.')
        return score, feedback
    except Exception as e:
        logger.error(f"Error evaluating answer: {str(e)}")
        return 50, f"Evaluation failed: {str(e)}"

# Compute performance rating
def get_performance_rating(percentage):
    if percentage >= 90:
        return "Excellent"
    elif percentage >= 75:
        return "Very Good"
    elif percentage >= 50:
        return "Good"
    else:
        return "Need to Improve"

@api_view(['POST'])
def start_interview(request):
    serializer = InterviewStartSerializer(data=request.data)
    if serializer.is_valid():
        mode = serializer.validated_data['mode']
        provider = serializer.validated_data['provider']
        transcript_ids = serializer.validated_data.get('transcript_ids', [])
        custom_info = serializer.validated_data.get('custom_info', {})

        try:
            client = initialize_llm_client(provider)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        try:
            mongo_client = MongoClient('mongodb://192.168.48.200:27017')
            db = mongo_client['video_transcriptions']
            transcripts_collection = db['transcriptions']
        except Exception as e:
            logger.error(f"MongoDB connection failed: {str(e)}")
            return Response({"error": "Failed to connect to MongoDB"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        session_id = str(uuid.uuid4())
        system_prompt = ""
        transcript = ""
        conversation_log = ""
        messages = []

        if mode == 'lecture':
            if not transcript_ids:
                return Response({"error": "Transcript IDs required for lecture mode"}, status=status.HTTP_400_BAD_REQUEST)
            try:
                transcript_ids = [int(id) for id in transcript_ids]
            except ValueError:
                return Response({"error": "Transcript IDs must be valid integers"}, status=status.HTTP_400_BAD_REQUEST)
            transcripts = list(transcripts_collection.find(
                {"$or": [
                    {"lecture_id": {"$in": transcript_ids}},
                    {"videoid": {"$in": [str(id) for id in transcript_ids]}}
                ]},
                {"transcript": 1, "transcription": 1}
            ))
            if not transcripts:
                return Response({"error": "No valid transcripts found"}, status=status.HTTP_404_NOT_FOUND)
            transcript = "\n\n".join(
                t.get("transcript", t.get("transcription", ""))
                for t in transcripts
                if "transcript" in t or "transcription" in t
            )
            system_prompt = (
                "You are a professional interviewer conducting an interview based on lecture transcripts.\n\n"
                "Instructions:\n"
                "- Ask exactly 10 questions, one at a time, based on the provided transcript.\n"
                "- Match the difficulty to an intermediate level suitable for a fresher.\n"
                "- Use a professional tone.\n"
                "- Output only a JSON object with 'type' ('question' or 'conclusion') and 'content' (question text or conclusion message).\n"
                "- If a response is off-topic or insufficient, guide the user back or ask for clarification.\n"
                "- After the 10th question, return a conclusion summarizing the interview.\n\n"
                f"Transcript:\n{transcript}\n\nStart the interview now."
            )
            conversation_log = f"Lecture Interview\nTranscript Summary: {transcript[:100]}...\n\n"
        elif mode == 'custom':
            if not custom_info:
                return Response({"error": "Custom info required for custom mode"}, status=status.HTTP_400_BAD_REQUEST)
            topic = custom_info.get('topic', 'General')
            difficulty = custom_info.get('difficulty', 'Intermediate')
            experience = custom_info.get('experience', 'Fresher')
            tone = custom_info.get('tone', 'Professional')
            transcript = (
                f"\n            The student has chosen to be evaluated on the topic: {topic}.\n"
                f"            The intended difficulty level is: {difficulty}.\n"
                f"            The student has experience level: {experience}.\n"
                f"            The tone of the interview should be: {tone}.\n        "
            )
            system_prompt = (
                "You are a professional interviewer creating a customized interview.\n\n"
                "Instructions:\n"
                f"- Ask exactly 10 questions, one at a time, on the topic: \"{topic}\".\n"
                f"- Match the difficulty level: \"{difficulty}\".\n"
                f"- Tailor questions to a candidate with experience level: \"{experience}\".\n"
                f"- Use a \"{tone}\" tone.\n"
                "- Output only a JSON object with 'type' ('question' or 'conclusion') and 'content' (question text or conclusion message).\n"
                "- If a response is off-topic or insufficient, guide the user back or ask for clarification.\n"
                "- After the 10th question, return a conclusion summarizing the interview.\n\n"
                "Start the interview now."
            )
            conversation_log = (
                f"Custom Interview\n"
                f"Topic: {topic}\n"
                f"Difficulty: {difficulty}\n"
                f"Experience: {experience}\n"
                f"Tone: {tone}\n\n"
            )

        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": "Start the interview"})
        
        try:
            response = generate_llm_response(client, provider, messages)
            if response['type'] != 'question':
                return Response({"error": "Invalid initial response from LLM"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            question = response['content']
            messages.append({"role": "assistant", "content": json.dumps(response)})
            conversation_log += f"Question 1: {question}\n"
            
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO Conversations (conversation_log) VALUES (?)",
                (conversation_log,)
            )
            cursor.execute(
                "INSERT INTO Questions (question_text) VALUES (?)",
                (question,)
            )
            conn.commit()
            cursor.close()
            conn.close()

            state = {
                "transcript": transcript,
                "mode": mode,
                "custom_info": custom_info,
                "messages": messages,
                "question_count": 1,
                "conversation_log": conversation_log,
                "off_topic_count": 0,
                "provider": provider,
                "scores": []
            }

            return Response({
                "session_id": session_id,
                "type": "question",
                "content": question,
                "state": state
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error in start_interview: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def submit_answer(request):
    serializer = InterviewAnswerSerializer(data=request.data)
    if serializer.is_valid():
        session_id = serializer.validated_data['session_id']
        answer = serializer.validated_data['answer']
        state = serializer.validated_data['state']

        if not state:
            return Response({"error": "State not provided"}, status=status.HTTP_400_BAD_REQUEST)

        provider = state.get('provider')
        messages = state.get('messages', [])
        question_count = state.get('question_count', 0)
        conversation_log = state.get('conversation_log', '')
        off_topic_count = state.get('off_topic_count', 0)
        transcript = state.get('transcript', '')
        mode = state.get('mode', '')
        custom_info = state.get('custom_info', {})
        scores = state.get('scores', [])

        try:
            client = initialize_llm_client(provider)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        # Get the current question from the last assistant message
        current_question = json.loads(messages[-1]['content'])['content'] if messages and messages[-1]['role'] == 'assistant' else ''

        # Evaluate the answer
        score, feedback = evaluate_answer(client, provider, current_question, answer, transcript, mode, custom_info)
        scores.append({"question": current_question, "answer": answer, "score": score, "feedback": feedback})

        messages.append({"role": "user", "content": answer})
        conversation_log += f"Answer: {answer}\n"

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO UserResponses (response_text, score) VALUES (?, ?)",
            (answer, score)
        )
        cursor.execute(
            "UPDATE Conversations SET conversation_log = ? WHERE id = (SELECT MAX(id) FROM Conversations)",
            (conversation_log,)
        )
        conn.commit()
        cursor.close()
        conn.close()

        system_prompt = messages[0]['content']
        messages[0] = {"role": "system", "content": system_prompt}
        
        try:
            if question_count >= 10:
                # Calculate total percentage
                total_percentage = sum(score['score'] for score in scores) / len(scores) if scores else 0
                rating = get_performance_rating(total_percentage)
                conclusion_content = (
                    f"Thank you for completing the interview. "
                    f"Your total score is {total_percentage:.1f}% ({rating}). "
                    f"Review your responses below:\n" +
                    "\n".join(
                        f"Question {i+1}: {score['question']} - Score: {score['score']}% - Feedback: {score['feedback']}"
                        for i, score in enumerate(scores)
                    )
                )
                conversation_log += f"Conclusion: {conclusion_content}\nTotal Percentage: {total_percentage:.1f}%\nRating: {rating}\n"
                
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE Conversations SET conversation_log = ?, total_percentage = ?, rating = ? WHERE id = (SELECT MAX(id) FROM Conversations)",
                    (conversation_log, total_percentage, rating)
                )
                conn.commit()
                cursor.close()
                conn.close()

                return Response({
                    "session_id": session_id,
                    "type": "conclusion",
                    "content": conclusion_content,
                    "total_percentage": total_percentage,
                    "rating": rating,
                    "scores": scores,
                    "completed": True
                }, status=status.HTTP_200_OK)

            response = generate_llm_response(client, provider, messages)
            messages.append({"role": "assistant", "content": json.dumps(response)})

            if response['type'] == 'question':
                question_count += 1
                conversation_log += f"Question {question_count}: {response['content']}\n"
                
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO Questions (question_text) VALUES (?)",
                    (response['content'],)
                )
                cursor.execute(
                    "UPDATE Conversations SET conversation_log = ? WHERE id = (SELECT MAX(id) FROM Conversations)",
                    (conversation_log,)
                )
                conn.commit()
                cursor.close()
                conn.close()

                state.update({
                    "messages": messages,
                    "question_count": question_count,
                    "conversation_log": conversation_log,
                    "off_topic_count": off_topic_count,
                    "scores": scores
                })

                return Response({
                    "session_id": session_id,
                    "type": "question",
                    "content": response['content'],
                    "state": state
                }, status=status.HTTP_200_OK)

            elif response['type'] == 'conclusion':
                total_percentage = sum(score['score'] for score in scores) / len(scores) if scores else 0
                rating = get_performance_rating(total_percentage)
                conclusion_content = (
                    response['content'] + f"\nTotal Score: {total_percentage:.1f}% ({rating})\n" +
                    "\n".join(
                        f"Question {i+1}: {score['question']} - Score: {score['score']}% - Feedback: {score['feedback']}"
                        for i, score in enumerate(scores)
                    )
                )
                conversation_log += f"Conclusion: {conclusion_content}\nTotal Percentage: {total_percentage:.1f}%\nRating: {rating}\n"
                
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE Conversations SET conversation_log = ?, total_percentage = ?, rating = ? WHERE id = (SELECT MAX(id) FROM Conversations)",
                    (conversation_log, total_percentage, rating)
                )
                conn.commit()
                cursor.close()
                conn.close()

                return Response({
                    "session_id": session_id,
                    "type": "conclusion",
                    "content": conclusion_content,
                    "total_percentage": total_percentage,
                    "rating": rating,
                    "scores": scores,
                    "completed": True
                }, status=status.HTTP_200_OK)

            else:
                return Response({"error": "Invalid response type from LLM"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        except Exception as e:
            logger.error(f"Error in submit_answer: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def end_interview(request):
    serializer = InterviewAnswerSerializer(data=request.data)
    if serializer.is_valid():
        session_id = serializer.validated_data['session_id']
        state = serializer.validated_data['state']

        if not state:
            return Response({"error": "State not provided"}, status=status.HTTP_400_BAD_REQUEST)

        scores = state.get('scores', [])
        conversation_log = state.get('conversation_log', '')

        if not scores:
            return Response({
                "session_id": session_id,
                "type": "conclusion",
                "content": "You ended the interview without answering any questions.",
                "total_percentage": 0,
                "rating": "N/A",
                "scores": [],
                "completed": False
            }, status=status.HTTP_200_OK)

        try:
            # Calculate total percentage based on answered questions
            total_percentage = sum(score['score'] for score in scores) / len(scores) if scores else 0
            rating = get_performance_rating(total_percentage)
            conclusion_content = (
                f"You have ended the interview early after answering {len(scores)} question(s). "
                f"Your total score so far is {total_percentage:.1f}% ({rating}). "
                f"Review your responses below:\n" +
                "\n".join(
                    f"Question {i+1}: {score['question']} - Score: {score['score']}% - Feedback: {score['feedback']}"
                    for i, score in enumerate(scores)
                )
            )
            conversation_log += f"Conclusion: {conclusion_content}\nTotal Percentage: {total_percentage:.1f}%\nRating: {rating}\n"

            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE Conversations SET conversation_log = ?, total_percentage = ?, rating = ? WHERE id = (SELECT MAX(id) FROM Conversations)",
                (conversation_log, total_percentage, rating)
            )
            conn.commit()
            cursor.close()
            conn.close()

            return Response({
                "session_id": session_id,
                "type": "conclusion",
                "content": conclusion_content,
                "total_percentage": total_percentage,
                "rating": rating,
                "scores": scores,
                "completed": False
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error in end_interview: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)