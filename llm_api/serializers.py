from rest_framework import serializers

class CustomInfoSerializer(serializers.Serializer):
    topic = serializers.CharField(max_length=100, default="General")
    difficulty = serializers.CharField(max_length=50, default="Intermediate")
    experience = serializers.CharField(max_length=50, default="Fresher")
    tone = serializers.CharField(max_length=50, default="Professional")

class InterviewStartSerializer(serializers.Serializer):
    mode = serializers.ChoiceField(choices=['lecture', 'custom'])
    provider = serializers.ChoiceField(choices=['openai', 'gemini', 'claude', 'groq'])
    transcript_ids = serializers.ListField(child=serializers.CharField(), required=False)
    custom_info = CustomInfoSerializer(required=False)

class InterviewAnswerSerializer(serializers.Serializer):
    session_id = serializers.CharField(max_length=36)
    answer = serializers.CharField()
    state = serializers.DictField()