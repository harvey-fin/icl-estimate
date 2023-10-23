tasks_qamrc = ["crossfit:adversarialqa", "crossfit:ropes", "crossfit:squad-with_context", "crossfit:tweet_qa"]
tasks_cbqa = ["crossfit:squad-no_context", "crossfit:numer_sense", "crossfit:kilt_trex", "crossfit:kilt_zsre", "crossfit:lama-trex", "crossfit:lama-squad", "crossfit:lama-google_re", "crossfit:lama-conceptnet", "crossfit:kilt_hotpotqa", "crossfit:kilt_nq", "crossfit:freebase_qa", "crossfit:web_questions", "crossfit:jeopardy"]
tasks_other = ["crossfit:spider", "crossfit:ade_corpus_v2-dosage", "crossfit:wikisql", "crossfit:gigaword"]

ontology = {"qamrc": tasks_qamrc, "cbqa": tasks_cbqa, "other": tasks_other, "seq2seq": tasks_qamrc + tasks_cbqa + tasks_other}


# mcqa: "crossfit:superglue-multirc", "race-high", "quail"
# qamrc: "crossfit:biomrc", "crossfit:duorc", "crossfit:quoref"
# other: "crossfit:kilt_ay2", "crossfit:xsum"
