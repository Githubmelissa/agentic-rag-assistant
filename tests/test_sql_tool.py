from pathlib import Path

from assistant.llm import LLMResponse, MockLLM
from assistant.tools.sql_tool import answer_sql_question, extract_sql


class ScriptedLLM:
    """Returns pre-written replies in order, and records the prompts it received."""

    model = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, system: str, user: str) -> LLMResponse:
        self.prompts.append(user)
        return LLMResponse(self.replies.pop(0), self.model, 1, 1, 0.0)


def test_extract_sql_from_code_block() -> None:
    assert extract_sql("Here:\n```sql\nSELECT 1;\n```\nDone") == "SELECT 1"
    assert extract_sql("SELECT 2") == "SELECT 2"


def test_happy_path(sales_db: Path) -> None:
    llm = ScriptedLLM(["```sql\nSELECT COUNT(*) FROM sales_clean\n```", "There are 3 lines."])
    a = answer_sql_question("How many lines?", llm, sales_db)
    assert a.succeeded and a.result is not None
    assert a.result.rows == [(3,)]
    assert a.answer == "There are 3 lines."
    assert (a.retries, a.guard_rejections) == (0, 0)


def test_error_is_fed_back_and_retried(sales_db: Path) -> None:
    llm = ScriptedLLM(
        [
            "SELECT revenue FROM sales_clean",  # no such column -> SQLite error
            "DELETE FROM sales",  # rejected by the guard
            "SELECT ROUND(SUM(line_total), 2) FROM sales_clean",
            "Revenue was 53.55 GBP.",
        ]
    )
    a = answer_sql_question("Total revenue?", llm, sales_db, max_retries=2)
    assert a.succeeded
    assert a.result is not None and a.result.rows == [(53.55,)]  # 6*2.55 + 2*12.75 + 1*12.75
    assert (a.retries, a.guard_rejections) == (2, 1)
    assert "no such column" in llm.prompts[1]  # the error text reached the model
    assert "Only SELECT" in llm.prompts[2]


def test_no_retry_mode_gives_up_after_one_failure(sales_db: Path) -> None:
    llm = ScriptedLLM(["SELECT nope FROM sales_clean"])
    a = answer_sql_question("?", llm, sales_db, max_retries=0)
    assert not a.succeeded
    assert len(a.attempts) == 1
    assert "could not answer" in a.answer


def test_mock_llm_works_end_to_end(sales_db: Path) -> None:
    a = answer_sql_question("How many lines?", MockLLM(), sales_db)
    assert a.succeeded and a.result is not None and a.result.rows == [(3,)]


def test_generate_answer_false_skips_second_call(sales_db: Path) -> None:
    llm = ScriptedLLM(["SELECT COUNT(*) FROM sales_clean"])
    a = answer_sql_question("?", llm, sales_db, generate_answer=False)
    assert a.succeeded and a.answer == "" and len(a.llm_calls) == 1
