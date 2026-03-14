import pytest

from tau2.domains.tossbank.environment import get_environment as toss_get_env
from tau2.domains.kakaobank.environment import get_environment as kakao_get_env
from tau2.domains.kbank.environment import get_environment as kbank_get_env


@pytest.mark.parametrize(
    "domain,get_env",
    [
        ("tossbank", toss_get_env),
        ("kakaobank", kakao_get_env),
        ("kbank", kbank_get_env),
    ],
)
def test_environment_loads(domain, get_env):
    env = get_env()
    assert env.get_domain_name() == domain
    assert len(env.get_policy()) > 100
    tools = env.get_tools()
    assert len(tools) > 5


@pytest.mark.parametrize(
    "get_env",
    [toss_get_env, kakao_get_env, kbank_get_env],
)
def test_solo_mode_raises(get_env):
    with pytest.raises(ValueError, match="does not support solo mode"):
        get_env(solo_mode=True)
