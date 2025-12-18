import pytest
import jsonschema
from ai_personas.builder import PersonaBuilder, PERSONA_SCHEMA, DEFAULT_CONFIG

class TestPersonaBuilder:
    def test_schema_valid_default(self):
        # Default config should match schema
        jsonschema.validate(instance=DEFAULT_CONFIG, schema=PERSONA_SCHEMA)

    @pytest.mark.django_db
    def test_heuristic_fallback(self):
        builder = PersonaBuilder()
        desc = "A pirate who loves coding"
        config = builder._heuristic_fallback(desc)
        
        assert "pirate" in config['system_prompt'].lower()
        assert config['display_name'] == "Generated Persona"
        jsonschema.validate(instance=config, schema=PERSONA_SCHEMA)

    @pytest.mark.asyncio
    async def test_generate_config_mock(self):
        # Assumes DEFAULT AI_MODE=mock
        builder = PersonaBuilder()
        desc = "Teacher"
        config = await builder.a_generate_config(desc)
        
        assert "Generated Persona" == config['display_name']
        assert "Teacher" in desc
        jsonschema.validate(instance=config, schema=PERSONA_SCHEMA)

    def test_validate_and_fix(self):
        builder = PersonaBuilder()
        # Invalid config (missing system_prompt)
        bad_config = {"display_name": "Bad"}
        
        fixed = builder.validate_and_fix(bad_config)
        
        assert "system_prompt" in fixed
        assert fixed['display_name'] == "Bad" # Should preserve valid fields
