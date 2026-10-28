from fastapi.testclient import TestClient
from app.main import app
from app.retrieval import Match

client=TestClient(app)


class DummyRetriever:
    def search(self,query,top_k):
        return [Match('id','sample.pdf','Calibrated intervals can provide coverage.',0.9,1)]


def test_health_is_lightweight():
    r=client.get('/health')
    assert r.status_code==200
    assert r.json()['status']=='ok'


def test_invalid_request_returns_422_without_model_download():
    assert client.post('/ask',json={'query':'  ','top_k':3}).status_code==422
    assert client.post('/search',json={'query':'valid','top_k':0}).status_code==422


def test_search(monkeypatch):
    monkeypatch.setattr('app.main.get_retriever',lambda:DummyRetriever())
    r=client.post('/search',json={'query':'intervals','top_k':1})
    assert r.status_code==200
    assert r.json()['results'][0]['page']==1


def test_ask(monkeypatch):
    monkeypatch.setattr('app.main.get_retriever',lambda:DummyRetriever())
    monkeypatch.setattr('app.main.generate_answer',lambda q,ms:'They provide coverage [1].')
    r=client.post('/ask',json={'query':'What intervals?','top_k':2})
    assert r.status_code==200
    assert '[1]' in r.json()['answer']
    assert r.json()['results'][0]['source']=='sample.pdf'


def test_no_index_returns_503(monkeypatch):
    def missing(): raise ValueError('not indexed')
    monkeypatch.setattr('app.main.get_retriever',missing)
    r=client.post('/search',json={'query':'real question','top_k':3})
    assert r.status_code==503
