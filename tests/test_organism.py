import pytest
from core.organism import Organism

def test_organism_init():
    org = Organism()
    assert org.device.type in ["cuda", "cpu"]
    assert len(org.short_term_memory) == 0

def test_perceive_text():
    org = Organism()
    result = org.perceive("Hello, world!", modality="text")
    assert "output" in result
    assert "awareness" in result