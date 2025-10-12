# tests/test_warden.py
"""
Unit tests for the Warden component.
"""
import pytest
from sisf.components.warden import Warden
from sisf.utils.policy_store import PolicyStore
from sisf.schemas.policies import HeuristicPolicy, EmbeddingSimilarityPolicy, PolicyAction

@pytest.fixture
def empty_store():
    return PolicyStore()

@pytest.fixture
def test_warden(empty_store):
    return Warden(model_name="mock-model", policy_store=empty_store)

def test_warden_allows_benign_prompt(test_warden):
    result = test_warden.process("Hello, how are you?")
    assert result["status"] == "ALLOWED"
    assert "MOCK response" in result["response"]

def test_warden_blocks_heuristic_policy(test_warden, empty_store):
    policy = HeuristicPolicy(
        id="pol_test_regex",
        description="Block the word 'forbidden'",
        action=PolicyAction.BLOCK,
        regex_pattern="forbidden"
    )
    empty_store.add_policy(policy)
    
    result = test_warden.process("This prompt is forbidden.")
    assert result["status"] == "BLOCKED"
    assert result["policy_id"] == "pol_test_regex"
    
    result_safe = test_warden.process("This prompt is allowed.")
    assert result_safe["status"] == "ALLOWED"

def test_warden_blocks_embedding_policy(test_warden, empty_store):
    goodbye_embedding = test_warden.embedding_model.encode("goodbye")
    policy = EmbeddingSimilarityPolicy(
        id="pol_test_embed",
        description="Block things that mean 'goodbye'",
        action=PolicyAction.BLOCK,
        reference_embedding=goodbye_embedding,
        similarity_threshold=0.9
    )
    empty_store.add_policy(policy)
    
    result = test_warden.process("goodbye")
    assert result["status"] == "BLOCKED"
    assert result["policy_id"] == "pol_test_embed"
    
    result_safe = test_warden.process("hello")
    assert result_safe["status"] == "ALLOWED"

def test_warden_flags_prompt(test_warden, empty_store):
    policy = HeuristicPolicy(
        id="pol_test_flag",
        description="Flag the word 'watch'",
        action=PolicyAction.FLAG_FOR_REVIEW,
        regex_pattern="watch"
    )
    empty_store.add_policy(policy)
    
    result = test_warden.process("I want to watch this.")
    assert result["status"] == "ALLOWED"
    assert result["flagged_by_policy_id"] == "pol_test_flag"

def test_warden_block_overrides_flag(test_warden, empty_store):
    flag_policy = HeuristicPolicy(id="pol_flag", description="Flag 'prompt'", action=PolicyAction.FLAG_FOR_REVIEW, regex_pattern="prompt")
    block_policy = HeuristicPolicy(id="pol_block", description="Block 'forbidden'", action=PolicyAction.BLOCK, regex_pattern="forbidden")
    empty_store.add_policy(flag_policy)
    empty_store.add_policy(block_policy)

    result = test_warden.process("This is a forbidden prompt.")
    assert result["status"] == "BLOCKED"
    assert result["policy_id"] == "pol_block"