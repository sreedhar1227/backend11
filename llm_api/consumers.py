# llm_api/consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from rest_framework import status
from .api import initialize_llm_client, generate_llm_response, get_db_connection
from .serializers import InterviewStartSerializer, InterviewAnswerSerializer
import uuid
import logging
from pymongo import MongoClient

logger = logging.getLogger(__name__)

class InterviewConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        await self.accept()
        logger.debug(f"WebSocket connected for session: {self.session_id}")

    async def disconnect(self, close_code):
        logger.debug(f"WebSocket disconnected for session: {self.session_id}")

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            message_type = data.get('type')

            if message_type == 'start_interview':
                await self.handle_start_interview(data)
            elif message_type == 'submit_answer':
                await self.handle_submit_answer(data)
            else:
                await self.send(text_data=json.dumps({
                    'status': 'error',
                    'message': 'Invalid message type'
                }))

        except Exception as e:
            logger.error(f"Error in WebSocket receive: {str(e)}")
            await self.send(text_data=json.dumps({
                'status': 'error',
                'message': str(e)
            }))

    async def handle_start_interview(self, data):
        serializer = InterviewStartSerializer(data=data)
        if serializer.is_valid():
            mode = serializer.validated_data['mode']
            provider = serializer.validated_data['provider']
            transcript_ids = serializer.validated_data.get('transcript_ids', [])
            custom_info = serializer.validated_data.get('custom_info', {})

            try:
                client = initialize_llm_client(provider)
            except ValueError as e:
                await self.send(text_data=json.dumps({
                    'status': 'error',
                    'message': str(e)
                }))
                return

            # MongoDB connection
            try:
                mongo_client = MongoClient('mongodb://192.168.48.200:27017')
                db = mongo_client['video_transcriptions']
                transcripts_collection = db['transcriptions']
            except Exception as e:
                logger.error(f"MongoDB connection failed: {str(e)}")
                await self.send(text_data=json.dumps({
                    'status': 'error',
                    'message': 'Failed to connect to MongoDB'
                }))
                return

            session_id = str(uuid.uuid4())
            system_prompt = ""
            transcript = ""
            conversation_log = ""
            messages = []

            if mode == 'lecture':
                if not transcript_ids:
                    await self.send(text_data=json.dumps({
                        'status': 'error',
                        'message': 'Transcript IDs required for lecture mode'
                    }))
                    return

                try:
                    transcript_ids = [int(id) for id in transcript_ids]
                except ValueError:
                    await self.send(text_data=json.dumps({
                        'status': 'error',
                        'message': 'Transcript IDs must be valid integers'
                    }))
                    return

                transcripts = list(transcripts_collection.find(
                    {"$or": [
                        {"lecture_id": {"$in": transcript_ids}},
                        {"videoid": {"$in": [str(id) for id in transcript_ids]}}
                    ]},
                    {"transcript": 1, "transcription": 1}
                ))
                if not transcripts:
                    await self.send(text_data=json.dumps({
                        'status': 'error',
                        'message': 'No valid transcripts found'
                    }))
                    return

                transcript = "\n\n".join(
                    t.get("transcript", t.get("transcription", ""))
                    for t in transcripts
                    if "transcript" in t or "transcription" in t
                )
                system_prompt = (
                    "You are a professional interviewer conducting an interview based on lecture transcripts.\n\n"
                    "Instructions:\n"
                    "- Ask one question at a time based on the provided transcript.\n"
                    "- Match the difficulty to an intermediate level suitable for a fresher.\n"
                    "- Use a professional tone.\n"
                    "- Output only a JSON object with 'type' ('question' or 'conclusion') and 'content' (question text or conclusion message).\n"
                    "- If a response is off-topic or insufficient, guide the user back or ask for clarification.\n\n"
                    f"Transcript:\n{transcript}\n\nStart the interview now."
                )
                conversation_log = f"Lecture Interview\nTranscript Summary: {transcript[:100]}...\n\n"

            elif mode == 'custom':
                if not custom_info:
                    await self.send(text_data=json.dumps({
                        'status': 'error',
                        'message': 'Custom info required for custom mode'
                    }))
                    return

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
                    f"- Ask one question at a time on the topic: \"{topic}\".\n"
                    f"- Match the difficulty level: \"{difficulty}\".\n"
                    f"- Tailor questions to a candidate with experience level: \"{experience}\".\n"
                    f"- Use a \"{tone}\" tone.\n"
                    "- Output only a JSON object with 'type' ('question' or 'conclusion') and 'content' (question text or conclusion message).\n"
                    "- If a response is off-topic or insufficient, guide the user back or ask for clarification.\n\n"
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

            try:
                response = generate_llm_response(client, provider, messages)
                if response['type'] != 'question':
                    await self.send(text_data=json.dumps({
                        'status': 'error',
                        'message': 'Invalid initial response from LLM'
                    }))
                    return

                question = response['content']
                messages.append({"role": "assistant", "content": json.dumps(response)})
                conversation_log += f"Question 1: {question}\n"

                # Save to SQL Server
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
                    "provider": provider
                }

                await self.send(text_data=json.dumps({
                    'status': 'success',
                    'session_id': session_id,
                    'type': 'question',
                    'content': question,
                    'state': state
                }))

            except Exception as e:
                logger.error(f"Error in start_interview: {str(e)}")
                await self.send(text_data=json.dumps({
                    'status': 'error',
                    'message': str(e)
                }))
        else:
            await self.send(text_data=json.dumps({
                'status': 'error',
                'message': serializer.errors
            }))

    async def handle_submit_answer(self, data):
        serializer = InterviewAnswerSerializer(data=data)
        if serializer.is_valid():
            session_id = serializer.validated_data['session_id']
            answer = serializer.validated_data['answer']
            state = serializer.validated_data['state']

            if not state:
                await self.send(text_data=json.dumps({
                    'status': 'error',
                    'message': 'State not provided'
                }))
                return

            provider = state.get('provider')
            messages = state.get('messages', [])
            question_count = state.get('question_count', 0)
            conversation_log = state.get('conversation_log', '')
            off_topic_count = state.get('off_topic_count', 0)

            try:
                client = initialize_llm_client(provider)
            except ValueError as e:
                await self.send(text_data=json.dumps({
                    'status': 'error',
                    'message': str(e)
                }))
                return

            messages.append({"role": "user", "content": answer})
            conversation_log += f"Answer: {answer}\n"

            # Update SQL Server
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO UserResponses (response_text) VALUES (?)",
                (answer,)
            )
            cursor.execute(
                "UPDATE Conversations SET conversation_log = ? WHERE id = (SELECT MAX(id) FROM Conversations)",
                (conversation_log,)
            )
            conn.commit()
            cursor.close()
            conn.close()

            # Prepare next question or conclude
            system_prompt = messages[0]['content']
            messages[0] = {"role": "system", "content": system_prompt}

            try:
                response = generate_llm_response(client, provider, messages)
                messages.append({"role": "assistant", "content": json.dumps(response)})

                if response['type'] == 'question':
                    question_count += 1
                    conversation_log += f"Question {question_count}: {response['content']}\n"

                    # Update SQL Server
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
                        "off_topic_count": off_topic_count
                    })

                    await self.send(text_data=json.dumps({
                        'status': 'success',
                        'session_id': session_id,
                        'type': 'question',
                        'content': response['content'],
                        'state': state
                    }))

                elif response['type'] == 'conclusion':
                    conversation_log += f"Conclusion[token]: {response['content']}\n"

                    # Update SQL Server
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE Conversations SET conversation_log = ? WHERE id = (SELECT MAX(id) FROM Conversations)",
                        (conversation_log,)
                    )
                    conn.commit()
                    cursor.close()
                    conn.close()

                    await self.send(text_data=json.dumps({
                        'status': 'success',
                        'session_id': session_id,
                        'type': 'conclusion',
                        'content': response['content']
                    }))

                else:
                    await self.send(text_data=json.dumps({
                        'status': 'error',
                        'message': 'Invalid response type from LLM'
                    }))

            except Exception as e:
                logger.error(f"Error in submit_answer: {str(e)}")
                await self.send(text_data=json.dumps({
                    'status': 'error',
                    'message': str(e)
                }))
        else:
            await self.send(text_data=json.dumps({
                'status': 'error',
                'message': serializer.errors
            }))