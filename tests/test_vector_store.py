import numpy as np
import pytest
from app.documents import Chunk
from app.vector_store import build_index, PersistentRetriever


class DummyEmbedder:
    model_id="fake-model"
    def encode_documents(self,texts): return np.array([[1,0],[0,1]],dtype='float32')
    def encode_query(self,query): return np.array([1,0],dtype='float32')


class FakeCollection:
    def __init__(self): self.rows={}
    def upsert(self,ids,documents,metadatas,embeddings):
        self.rows.update({id:(document,meta,embedding) for id,document,meta,embedding in zip(ids,documents,metadatas,embeddings)})
    def count(self): return len(self.rows)
    def get(self,limit,include):return {"metadatas":[v[1] for v in list(self.rows.values())[:limit]]}
    def query(self,query_embeddings,n_results,include):
        emb=np.asarray(query_embeddings[0]);ranked=sorted(self.rows.items(),key=lambda r:-np.dot(emb,r[1][2]))[:n_results]
        return {"ids":[[r[0] for r in ranked]],"documents":[[r[1][0] for r in ranked]],
        "metadatas":[[r[1][1] for r in ranked]],"distances":[[1-np.dot(emb,r[1][2]) for r in ranked]]}


class FakeClient:
    def __init__(self): self.collection=FakeCollection()
    def delete_collection(self,name): self.collection=FakeCollection()
    def get_or_create_collection(self,*args,**kwargs): return self.collection
    def get_collection(self,*args,**kwargs): return self.collection


def test_persistent_index_lifecycle(monkeypatch,tmp_path):
    client=FakeClient()
    monkeypatch.setattr('app.vector_store.client_for',lambda _:client)
    chunks=[Chunk('a','one.pdf','conformal predictions',2),Chunk('b','two.txt','Docker serving')]
    assert build_index(chunks,tmp_path,DummyEmbedder())==2
    retriever=PersistentRetriever(tmp_path,DummyEmbedder())
    matches=retriever.search('uncertainty',top_k=2)
    assert matches[0].source=='one.pdf'
    assert matches[0].page==2
    assert matches[0].score==pytest.approx(1)


def test_empty_index_rejected(monkeypatch,tmp_path):
    monkeypatch.setattr('app.vector_store.client_for',lambda _:FakeClient())
    with pytest.raises(ValueError,match='empty'): PersistentRetriever(tmp_path,DummyEmbedder())
