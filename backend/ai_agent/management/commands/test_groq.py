import os
import asyncio
import logging

from django.core.management.base import BaseCommand

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Test the configured Groq endpoint using the GroqLLM adapter.'

    def handle(self, *args, **options):
        # Print environment-derived config
        self.stdout.write("Groq configuration:\n")
        self.stdout.write(f"  GROQ_API_KEY set: {'GROQ_API_KEY' in os.environ}\n")
        self.stdout.write(f"  GROQ_API_URL: {os.environ.get('GROQ_API_URL')}\n")
        self.stdout.write(f"  GROQ_MODEL: {os.environ.get('GROQ_MODEL')}\n")
        self.stdout.write(f"  GROQ_MODEL_ENDPOINT: {os.environ.get('GROQ_MODEL_ENDPOINT')}\n")

        try:
            # Import here so Django settings are loaded first
            from ai_agent.live_providers.groq_llm import GroqLLM

            llm = GroqLLM()

            async def run_test():
                messages = [{'role': 'user', 'content': 'Say hello and identify yourself.'}]
                result = await llm.generate_response(messages, system_prompt='Test prompt')
                return result

            res = asyncio.run(run_test())
            self.stdout.write("\nGroq test result:\n")
            self.stdout.write(str(res) + "\n")
        except Exception as e:
            logger.exception("Groq test command failed")
            self.stderr.write(f"Groq test failed: {e}\n")
