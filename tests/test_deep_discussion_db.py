"""Tests for DeepDiscussionSession database model and Pydantic schemas."""
from datetime import datetime


def test_deep_discussion_session_db_defaults():
    """DeepDiscussionSessionDB creates with correct defaults."""
    from adam.api.models import DeepDiscussionSessionDB

    session = DeepDiscussionSessionDB(
        project_id="proj-123",
        question="What is the best way to structure a data pipeline?",
        pattern="peer_review",
    )

    assert session.project_id == "proj-123"
    assert session.question == "What is the best way to structure a data pipeline?"
    assert session.pattern == "peer_review"
    assert session.status == "configuring"
    assert session.total_cost == 0.0
    assert session.budget == 2.0
    assert session.conversation_id is None
    assert session.result is None
    assert session.scratchpad_data is None


def test_deep_discussion_session_db_id_generated():
    """DeepDiscussionSessionDB auto-generates a UUID id."""
    from adam.api.models import DeepDiscussionSessionDB

    session = DeepDiscussionSessionDB(
        project_id="proj-456",
        question="How should we handle incremental loads?",
        pattern="sequential",
    )

    assert session.id is not None
    assert len(session.id) == 36  # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx


def test_deep_discussion_session_create_schema_valid():
    """Pydantic create schema accepts valid input."""
    from adam.api.models import DeepDiscussionSessionCreate

    schema = DeepDiscussionSessionCreate(
        project_id="proj-123",
        question="What is the best deployment strategy?",
    )

    assert schema.project_id == "proj-123"
    assert schema.question == "What is the best deployment strategy?"
    assert schema.pattern == "peer_review"  # default
    assert schema.conversation_id is None  # default


def test_deep_discussion_session_create_schema_with_all_fields():
    """Pydantic create schema accepts all optional fields."""
    from adam.api.models import DeepDiscussionSessionCreate

    schema = DeepDiscussionSessionCreate(
        project_id="proj-789",
        question="Sequential pattern test question",
        pattern="sequential",
        conversation_id="conv-001",
    )

    assert schema.pattern == "sequential"
    assert schema.conversation_id == "conv-001"


def test_deep_discussion_session_response_schema_serializes():
    """Pydantic response schema serializes correctly from ORM object."""
    from adam.api.models import DeepDiscussionSessionDB, DeepDiscussionSessionResponse

    now = datetime(2026, 3, 25, 12, 0, 0)

    session = DeepDiscussionSessionDB(
        id="sess-abc-123",
        project_id="proj-123",
        conversation_id="conv-001",
        question="Is this architecture correct?",
        pattern="debate",
        model_assignments={"role_a": "claude-opus-4-5", "role_b": "gemini-pro"},
        budget=5.0,
        total_cost=1.25,
        status="completed",
        result="The architecture is sound.",
        scratchpad_data={"notes": "some intermediate notes"},
        created_at=now,
        completed_at=now,
    )

    response = DeepDiscussionSessionResponse.model_validate(session)

    assert response.id == "sess-abc-123"
    assert response.project_id == "proj-123"
    assert response.conversation_id == "conv-001"
    assert response.question == "Is this architecture correct?"
    assert response.pattern == "debate"
    assert response.model_assignments == {"role_a": "claude-opus-4-5", "role_b": "gemini-pro"}
    assert response.budget == 5.0
    assert response.total_cost == 1.25
    assert response.status == "completed"
    assert response.result == "The architecture is sound."
    assert response.scratchpad_data == {"notes": "some intermediate notes"}
    assert response.created_at == now
    assert response.completed_at == now


def test_deep_discussion_session_update_schema_allows_partial():
    """Update schema allows partial updates (all fields optional)."""
    from adam.api.models import DeepDiscussionSessionUpdate

    # Empty update is valid
    update_empty = DeepDiscussionSessionUpdate()
    assert update_empty.model_assignments is None
    assert update_empty.pattern is None
    assert update_empty.budget is None

    # Partial update with only budget
    update_budget = DeepDiscussionSessionUpdate(budget=10.0)
    assert update_budget.budget == 10.0
    assert update_budget.model_assignments is None
    assert update_budget.pattern is None

    # Partial update with only model_assignments
    update_models = DeepDiscussionSessionUpdate(
        model_assignments={"analyst": "claude-opus-4-5"}
    )
    assert update_models.model_assignments == {"analyst": "claude-opus-4-5"}
    assert update_models.pattern is None
    assert update_models.budget is None

    # Full update
    update_full = DeepDiscussionSessionUpdate(
        model_assignments={"analyst": "gemini-pro"},
        pattern="debate",
        budget=3.0,
    )
    assert update_full.pattern == "debate"
    assert update_full.budget == 3.0


def test_deep_discussion_session_db_tablename():
    """DeepDiscussionSessionDB uses correct table name."""
    from adam.api.models import DeepDiscussionSessionDB

    assert DeepDiscussionSessionDB.__tablename__ == "deep_discussion_sessions"


def test_deep_discussion_session_response_default_total_cost():
    """DeepDiscussionSessionResponse defaults total_cost to 0.0."""
    from adam.api.models import DeepDiscussionSessionResponse

    response = DeepDiscussionSessionResponse(
        id="sess-001",
        project_id="proj-001",
        question="Test question",
        pattern="peer_review",
        model_assignments={},
        budget=2.0,
        status="configuring",
        created_at=datetime(2026, 3, 25, 12, 0, 0),
    )

    assert response.total_cost == 0.0
    assert response.conversation_id is None
    assert response.result is None
    assert response.scratchpad_data is None
    assert response.completed_at is None
