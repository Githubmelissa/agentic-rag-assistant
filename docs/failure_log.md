# Failure log

Real failures observed while building, with their cause. This feeds the README's failure
analysis (Stage 8). Only failures actually seen in a run are listed.

| # | Stage | Question / input | What happened | Cause | Status |
|---|---|---|---|---|---|
| 1 | 2 | "What was Kering's revenue in 2025?" (dense only) | Top 5 were general financial-statement pages; none states the revenue | Every chunk in the section starts with the same heading and many share words like "Kering" and "2025", so they look alike to the embedding model | Hybrid ranked p.13 (correct) first; to be measured in Stage 6 |
| 2 | 2 | Same question, `hybrid_rerank` | Rank 1 was p.45, about *2026* first-quarter revenue | The cross-encoder matched "Kering" + "revenue" but missed the year | Correct page at rank 2; measured in Stage 6 |
| 3 | 3 | "How many employees did Kering have at the end of 2025?" (qwen2.5:3b) | Answered "721 employees", without any citation. The report states 43,731 employees (PDF p.172) | Probably took the headcount of the parent company Kering SA from the financial statements; small model ignored the citation rule | Uncited answers now flagged with a warning |
| 4 | 3 | "What was Kering's revenue in 2025?" | Correct answer and page, but flagged as an invalid citation | Model wrote `Kering_urd_2025`; citation check was case-sensitive (our bug) | Fixed: matching ignores case |
