import json

from experiments.hyperparam.responses_report import build_responses_report


def test_build_responses_report_embeds_results_and_simulations(tmp_path):
    exp_dir = tmp_path / "responses-exp"
    exp_dir.mkdir()
    (exp_dir / "manifest.json").write_text(
        json.dumps(
            {
                "exp_name": "responses-exp",
                "shape": "ofat",
                "model": "gpt-5.4-mini",
                "points": [
                    {
                        "reasoning_effort": "medium",
                        "verbosity": "medium",
                        "web_search_mode": "off",
                        "service_tier": "default",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (exp_dir / "results.csv").write_text(
        "name,domain,mode,llm,reasoning_effort,verbosity,web_search_mode,service_tier,avg_reward,avg_duration_seconds,avg_estimated_total_cost_usd\n"
        "baseline,retail,default,gpt-5.4-mini,medium,medium,off,default,1,5,0.01\n",
        encoding="utf-8",
    )
    (exp_dir / "simulations.csv").write_text(
        "name,domain,mode,llm,reasoning_effort,verbosity,web_search_mode,service_tier,simulation_id,task_id,trial,reward,duration_seconds,estimated_total_cost_usd,agent_total_tokens,user_total_tokens,agent_web_search_calls,agent_llm_calls,user_llm_calls\n"
        "baseline,retail,default,gpt-5.4-mini,medium,medium,off,default,sim-1,task-1,0,1,5,0.01,100,20,0,1,0\n",
        encoding="utf-8",
    )

    out = build_responses_report(exp_dir)

    html = out.read_text(encoding="utf-8")
    assert "Run Progress" in html
    assert "OFAT Coverage Matrix" in html
    assert "Performance vs latency" in html
    assert "Task Table" in html
    assert "Median latency" in html
    assert "Expanded task metrics" in html
    assert "Latency p95" in html
    assert "Token p95" in html
    assert "Success vs Latency And Steps" in html
    assert "Success rate by latency" in html
    assert "Success heatmap" in html
    assert "run-state-data" in html
    assert "simulations-data" in html
    assert "task-1" in html
