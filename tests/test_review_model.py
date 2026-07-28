from types import SimpleNamespace

from tau2.evaluator import reviewer
from tau2.runner import batch


def test_review_simulation_uses_custom_model_for_full_turn_based(monkeypatch):
    captured = {}
    expected_review = object()
    expected_auth = object()
    review_model = "gpt-4.1"

    def fake_review(**kwargs):
        captured["review_model"] = kwargs["review_model"]
        captured["full_trajectory"] = kwargs["full_trajectory"]
        return expected_review

    def fake_classify(**kwargs):
        captured["auth_model"] = kwargs["model"]
        captured["auth_messages"] = kwargs["messages"]
        return expected_auth

    monkeypatch.setattr(
        reviewer.ConversationReviewer,
        "review",
        staticmethod(fake_review),
    )
    monkeypatch.setattr(
        reviewer.AuthenticationClassifier,
        "classify",
        staticmethod(fake_classify),
    )

    simulation = SimpleNamespace(messages=["message"], ticks=None)

    review, auth = reviewer.review_simulation(
        simulation=simulation,
        task=object(),
        mode=reviewer.ReviewMode.FULL,
        user_info=object(),
        policy="policy",
        review_model=review_model,
    )

    assert review is expected_review
    assert auth is expected_auth
    assert captured == {
        "review_model": review_model,
        "full_trajectory": ["message"],
        "auth_model": review_model,
        "auth_messages": ["message"],
    }


def test_review_simulation_uses_custom_model_for_full_duplex(monkeypatch):
    captured = {}
    expected_review = object()
    expected_auth = object()
    review_model = "openrouter/anthropic/claude-sonnet-4"

    def fake_review(**kwargs):
        captured["review_model"] = kwargs["review_model"]
        captured["full_trajectory"] = kwargs["full_trajectory"]
        captured["interruption_enabled"] = kwargs["interruption_enabled"]
        return expected_review

    def fake_classify(**kwargs):
        captured["auth_model"] = kwargs["model"]
        captured["auth_ticks"] = kwargs["ticks"]
        return expected_auth

    monkeypatch.setattr(
        reviewer.FullDuplexConversationReviewer,
        "review",
        staticmethod(fake_review),
    )
    monkeypatch.setattr(
        reviewer.FullDuplexAuthenticationClassifier,
        "classify",
        staticmethod(fake_classify),
    )

    simulation = SimpleNamespace(messages=[], ticks=["tick"])

    review, auth = reviewer.review_simulation(
        simulation=simulation,
        task=object(),
        mode=reviewer.ReviewMode.FULL,
        user_info=object(),
        policy="policy",
        interruption_enabled=True,
        review_model=review_model,
    )

    assert review is expected_review
    assert auth is expected_auth
    assert captured == {
        "review_model": review_model,
        "full_trajectory": ["tick"],
        "interruption_enabled": True,
        "auth_model": review_model,
        "auth_ticks": ["tick"],
    }


def test_review_simulation_uses_custom_model_for_user_only_turn_based(
    monkeypatch,
):
    captured = {}
    expected_review = object()
    review_model = "gemini/gemini-2.5-pro"

    def fake_review(**kwargs):
        captured["review_model"] = kwargs["review_model"]
        captured["full_trajectory"] = kwargs["full_trajectory"]
        return expected_review

    monkeypatch.setattr(
        reviewer.UserOnlyReviewer,
        "review",
        staticmethod(fake_review),
    )

    simulation = SimpleNamespace(messages=["message"], ticks=None)

    review, auth = reviewer.review_simulation(
        simulation=simulation,
        task=object(),
        mode=reviewer.ReviewMode.USER,
        user_info=object(),
        review_model=review_model,
    )

    assert review is expected_review
    assert auth is None
    assert captured == {
        "review_model": review_model,
        "full_trajectory": ["message"],
    }


def test_review_simulation_uses_custom_model_for_user_only_full_duplex(
    monkeypatch,
):
    captured = {}
    expected_review = object()
    review_model = "anthropic/claude-sonnet-4-5"

    def fake_review(**kwargs):
        captured["review_model"] = kwargs["review_model"]
        captured["full_trajectory"] = kwargs["full_trajectory"]
        captured["interruption_enabled"] = kwargs["interruption_enabled"]
        return expected_review

    monkeypatch.setattr(
        reviewer.FullDuplexUserOnlyReviewer,
        "review",
        staticmethod(fake_review),
    )

    simulation = SimpleNamespace(messages=[], ticks=["tick"])

    review, auth = reviewer.review_simulation(
        simulation=simulation,
        task=object(),
        mode=reviewer.ReviewMode.USER,
        user_info=object(),
        interruption_enabled=True,
        review_model=review_model,
    )

    assert review is expected_review
    assert auth is None
    assert captured == {
        "review_model": review_model,
        "full_trajectory": ["tick"],
        "interruption_enabled": True,
    }


def test_auto_review_uses_custom_review_model(monkeypatch):
    captured = {}
    expected_review = SimpleNamespace(has_errors=False)
    expected_auth = object()
    review_model = "gpt-4.1-mini"

    def fake_review_simulation(**kwargs):
        captured["review_model"] = kwargs["review_model"]
        captured["mode"] = kwargs["mode"]
        return expected_review, expected_auth

    monkeypatch.setattr(
        reviewer,
        "review_simulation",
        fake_review_simulation,
    )

    simulation = SimpleNamespace()
    task = SimpleNamespace(id="task-1")

    batch.run_auto_review(
        simulation=simulation,
        task=task,
        review_mode="full",
        review_model=review_model,
        user="user_simulator",
        llm_user="gpt-4.1",
        llm_args_user={},
        user_persona_config=None,
        user_voice_settings=None,
        policy="policy",
        is_audio_native=False,
    )

    assert simulation.review is expected_review
    assert simulation.auth_classification is expected_auth
    assert captured == {
        "review_model": review_model,
        "mode": reviewer.ReviewMode.FULL,
    }
