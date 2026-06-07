# Chunk Check FOUND No-Answer Decisions

Scope: burs, yaz_okulu, devamsizlik, askerlik_tecili.

## Summary

- Global retrieval thresholds were not lowered.
- Burs no-answer cases are treated as genuinely unanswerable with the current chunks, because matched chunks contain misleading URL/title/topic signals but do not contain burs application or eligibility text.
- Yaz okulu, devamsizlik and askerlik tecili failures were caused by weak intent/source scoring and missing topic-specific evidence rules.

## Decisions By Case

| Case | Decision | Fix |
| --- | --- | --- |
| burs_001 | Genuinely unanswerable with current chunks | Kept no-answer behavior, added stricter scholarship focus so generic "burs" or wrong URL/title matches cannot produce an answer. Marked the evaluation case as `expect_no_sources`. |
| burs_002 | Genuinely unanswerable with current chunks | Same as burs_001. The referenced source is not a burs eligibility document in `chunks.json`, so lowering the threshold would cause hallucination. |
| yaz_okulu_002 | Retrieval/scoring issue | Added yaz okulu max-load intent and boosted chunks containing `yaz okulu` + `24 AKTS`. Added direct answer: course count depends on AKTS, upper limit is 24 AKTS. |
| yaz_okulu_005 | Retrieval/scoring issue | Same max-load fix; moved yaz okulu max-load direct answer before general AKTS registration logic to avoid 30 AKTS semester-load confusion. |
| devamsizlik_003 | Evidence focus issue | Practical/attendance questions now use attendance focus instead of generic process focus. The official 70% theoretical / 80% practical rule is selected. |
| devamsizlik_004 | Evidence focus issue | Strengthened devamsizlik source hints and evidence scoring for `devam zorunlulugu`, `%70`, `%80`. |
| askerlik_tecili_001 | Metadata/scoring issue | Added military deferral source hints and specialized candidates for `askerlik tecil islemleri` + `ogrenci isleri`. |
| askerlik_tecili_003 | Intent detection issue | Expanded tecil detection so "Tecili öğrenci işleri mi yapıyor?" is recognized even without the word "askerlik". |

## Verification Snapshot

- burs_001: no-answer, no sources.
- burs_002: no-answer, no sources.
- askerlik_tecili_001: answer produced, acceptable source satisfied.
- askerlik_tecili_003: answer produced, acceptable source satisfied.
- yaz_okulu_002: answer produced, acceptable source satisfied.
- yaz_okulu_005: answer produced, acceptable source satisfied.
- devamsizlik_003: answer produced, acceptable source satisfied.
- devamsizlik_004: answer produced, acceptable source satisfied.
