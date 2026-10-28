import pytest
from app.documents import load_chunks,load_corpus,split_words


def test_word_chunks_overlap():
    assert split_words("one two three four five six",size=4,overlap=2) == ["one two three four", "three four five six"]


def test_invalid_chunking():
    with pytest.raises(ValueError): split_words("hi",size=4,overlap=4)


def test_text_source(tmp_path):
    (tmp_path/"sample.txt").write_text("alpha beta gamma delta",encoding="utf-8")
    result=load_chunks(tmp_path,size=3,overlap=1)
    assert len(result)==2
    assert result[0].source=="sample.txt"
    assert result[0].page is None


def test_empty_corpus(tmp_path):
    with pytest.raises(ValueError): load_corpus([tmp_path])


def test_pdf_page_numbers(tmp_path):
    import pymupdf
    path=tmp_path/"paper.pdf"
    document=pymupdf.open()
    for text in ("Conformal prediction intervals", "Semantic search with vectors"):
        page=document.new_page()
        page.insert_text((72,72),text)
    document.save(str(path));document.close()
    chunks=load_chunks(tmp_path)
    assert [c.page for c in chunks]==[1,2]
    assert all(c.source=="paper.pdf" for c in chunks)
