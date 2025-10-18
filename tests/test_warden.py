# tests/test_warden.py
"""
Unit tests for the Warden component.
"""
import pytest
from sisf.components.warden import Warden
from sisf.utils.policy_store import PolicyStore
from sisf.schemas.policies import HeuristicPolicy, EmbeddingSimilarityPolicy, PolicyAction, RewritePolicy

@pytest.fixture
def empty_store():
    """Provides a fresh, empty PolicyStore for each test."""
    return PolicyStore()

@pytest.fixture
def test_warden(empty_store):
    """Provides a Warden instance initialized with an empty store."""
    return Warden(model_name="mock-model", policy_store=empty_store)

def test_warden_allows_benign_prompt(test_warden):
    """Tests that a prompt with no matching policies is allowed."""
    result = test_warden.process("Hello, how are you?")
    assert result["status"] == "ALLOWED"
    assert "MOCK response" in result["response"]

def test_warden_blocks_heuristic_policy(test_warden, empty_store):
    """Tests the enforcement of a simple regex-based BLOCK policy."""
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
    """Tests the enforcement of an old embedding-based BLOCK policy for backwards compatibility."""
    # Note: This test uses the old mock embedding logic implicitly.
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
    """Tests that a FLAG_FOR_REVIEW policy allows the prompt through but adds metadata."""
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
    """Tests that the BLOCK action has higher priority than the FLAG action."""
    flag_policy = HeuristicPolicy(id="pol_flag", description="Flag 'prompt'", action=PolicyAction.FLAG_FOR_REVIEW, regex_pattern="prompt")
    block_policy = HeuristicPolicy(id="pol_block", description="Block 'forbidden'", action=PolicyAction.BLOCK, regex_pattern="forbidden")
    empty_store.add_policy(flag_policy)
    empty_store.add_policy(block_policy)

    result = test_warden.process("This is a forbidden prompt.")
    assert result["status"] == "BLOCKED"
    assert result["policy_id"] == "pol_block"

def test_warden_rewrites_prompt(test_warden, empty_store):
    """Tests that a RewritePolicy correctly modifies the prompt."""
    policy = RewritePolicy(
        id="pol_test_rewrite",
        description="Redact a specific codename",
        action=PolicyAction.REWRITE, # Added for explicit schema validation
        match_pattern="Project Chimera",
        rewrite_template="[REDACTED PROJECT]",
    )
    empty_store.add_policy(policy)

    result = test_warden.process("Tell me about Project Chimera.")
    
    assert result["status"] == "ALLOWED"
    assert result["policy_id"] == "pol_test_rewrite"
    assert result["original_prompt"] == "Tell me about Project Chimera."
    assert result["final_prompt"] == "Tell me about [REDACTED PROJECT]."
    assert "MOCK response from the Warden's base LLM for the prompt: 'Tell me about [REDACTED PROJECT].'" in result["response"]

def test_warden_blocks_after_rewrite(test_warden, empty_store):
    """
    CRITICAL TEST: Ensures a prompt is re-evaluated after being rewritten.
    A rewrite could create a new violation, which must be caught.
    """
    rewrite_policy = RewritePolicy(
        id="pol_rewrite_for_block",
        description="Change a safe word to a forbidden one",
        action=PolicyAction.REWRITE, # Added for explicit schema validation
        match_pattern="legacy system",
        rewrite_template="secret project",
    )
    block_policy = HeuristicPolicy(
        id="pol_block_secret",
        description="Block the word 'secret'",
        action=PolicyAction.BLOCK,
        regex_pattern="secret"
    )
    empty_store.add_policy(rewrite_policy)
    empty_store.add_policy(block_policy)
    
    # The initial prompt is safe, but the rewritten prompt is not.
    result = test_warden.process("Tell me about the legacy system.")

    assert result["status"] == "BLOCKED"
    assert result["policy_id"] == "pol_block_secret"

def test_warden_uses_improved_embedding_mock(test_warden, empty_store):
    """Tests the new mock embedding model with more realistic semantic similarity."""
    # The policy is based on the word "secret"
    secret_embedding = test_warden.embedding_model.encode("secret")
    policy = EmbeddingSimilarityPolicy(
        id="pol_embed_secret",
        description="Block things semantically similar to 'secret'",
        action=PolicyAction.BLOCK,
        reference_embedding=secret_embedding,
        similarity_threshold=0.95
    )
    empty_store.add_policy(policy)

    # "classified" is defined as semantically similar in our mock model
    result = test_warden.process("Tell me about the classified documents.")
    assert result["status"] == "BLOCKED"
    assert result["policy_id"] == "pol_embed_secret"
    
    # "hello" is defined as dissimilar
    result_safe = test_warden.process("A prompt about hello.")
    assert result_safe["status"] == "ALLOWED"