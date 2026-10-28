import pytest
from app.retrieval import Match
from app.ollama_client import make_prompt,generate_answer,LLMUnavailable


def source(): return [Match('a','paper.pdf','Coverage is at least 90%.',0.98,5)]


def test_prompt_carries_page_and_injection_warning():
    p=make_prompt('What coverage?',source())
    assert 'paper.pdf, page 5' in p[1]['content']
    assert 'untrusted data' in p[0]['content']


def test_generate_uses_local_api(monkeypatch):
    class Response:
        def raise_for_status(self): pass
        def json(self):return {"message":{"content":"At least 90% [1]."}}
    def fake_post(url,json,timeout):
        assert url=='http://127.0.0.1:11434/api/chat'
        assert json['stream'] is False
        return Response()
    monkeypatch.setattr('app.ollama_client.httpx.post',fake_post)
    assert generate_answer('What coverage?',source(),url='http://127.0.0.1:11434')=='At least 90% [1].'


def test_empty_results_do_not_call_ollama():
    assert 'No relevant' in generate_answer('question',[])


def test_ollama_failure_is_actionable(monkeypatch):
    import httpx
    def fail(*args,**kwargs): raise httpx.ConnectError('offline')
    monkeypatch.setattr('app.ollama_client.httpx.post',fail)
    with pytest.raises(LLMUnavailable,match='Install Ollama'): generate_answer('q',source())
