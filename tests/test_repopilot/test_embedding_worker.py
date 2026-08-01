from openharness.repopilot.embedding_worker import _encode_missing_in_batches


def test_missing_embeddings_are_persisted_after_each_batch() -> None:
    encoded_batches = []
    persisted_batches = []

    class FakeModel:
        def encode(self, texts, **_kwargs):
            encoded_batches.append(list(texts))
            return [[float(text)] for text in texts]

    def persist(ids, values):
        persisted_batches.append((list(ids), list(values)))

    generated = _encode_missing_in_batches(
        missing=[0, 1, 2, 3, 4],
        texts=["0", "1", "2", "3", "4"],
        embedding_ids=["a", "b", "c", "d", "e"],
        model=FakeModel(),
        batch_size=2,
        persist=persist,
    )

    assert encoded_batches == [["0", "1"], ["2", "3"], ["4"]]
    assert [ids for ids, _values in persisted_batches] == [
        ["a", "b"],
        ["c", "d"],
        ["e"],
    ]
    assert generated == {
        0: [0.0],
        1: [1.0],
        2: [2.0],
        3: [3.0],
        4: [4.0],
    }
