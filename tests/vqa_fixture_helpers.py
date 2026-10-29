"""Controlled VLM outputs and candidate records for CPU VQA contracts."""


class FakeVLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, image, prompt):
        self.calls.append((image, prompt))
        if not self.responses:
            raise AssertionError("unexpected VLM call")
        return self.responses.pop(0)


def _candidate(candidate_id, source_file, **payload):
    return {
        "id": candidate_id,
        "rrf_score": 0.4,
        "payload": {"source_file": source_file, "timestamp": 0.0, **payload},
    }
